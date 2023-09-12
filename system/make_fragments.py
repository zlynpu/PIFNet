import torch
import math
import os, sys
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME

pyexample_path = '/data1/zhangliyuan/code/IMFNet_exp'
sys.path.append(pyexample_path)

from system.optimize_posegraph import optimize_posegraph_for_fragment
from util.color_map import read_bin, get_fov_mask
from util.calibration import get_calib_from_file
import matplotlib.image as image
from util.uio import process_image
from scripts.benchmark_util import run_ransac
from util.misc import extract_features
from model import load_model
from util.pointcloud import make_open3d_point_cloud,make_open3d_feature_from_numpy

def odometry_to_positions(odometry):
    T_w_cam0 = odometry.reshape(3, 4)
    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
    return T_w_cam0

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

def pose_estimation(data_path, s, t, model, config):

    R = np.array([
          7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
          -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
      ]).reshape(3, 3)
    T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    velo2cam = np.hstack([R, T])
    velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
    
    device = torch.device("cuda")
    drive = 9
    pose =os.path.join(data_path, "dataset/poses/%02d.txt"%drive)
    icp_path = os.path.join(data_path, "icp")
    source = os.path.join(data_path, "dataset/sequences/%02d/velodyne/%06d.bin"%(drive,s))
    target = os.path.join(data_path, "dataset/sequences/%02d/velodyne/%06d.bin"%(drive,t))
    camera_path = os.path.join(data_path, "dataset/sequences/%02d/calib.txt"%drive)
    result = get_calib_from_file(camera_path)
    extrinsic = result['Tr_velo2cam']
    intrinsic = result['P2']
    extrinsic = np.expand_dims(extrinsic, axis=0)
    intrinsic = np.expand_dims(intrinsic, axis=0)

    p_points = read_bin(source)
    q_points = read_bin(target)

    pc1=o3d.geometry.PointCloud()
    pc2=o3d.geometry.PointCloud()

    pc1.points=o3d.utility.Vector3dVector(p_points) 
    pc2.points=o3d.utility.Vector3dVector(q_points) 

    p_xyz = np.asarray(pc1.points)
    q_xyz = np.asarray(pc2.points)
    source_image = source.replace(".bin", ".png")
    target_image = target.replace(".bin", ".png")


    p_image = image.imread(source_image)
    image_shape = np.asarray(p_image.shape)
    image_shape = np.expand_dims(image_shape,axis=0)
    if (p_image.shape[0] != config.image_H or p_image.shape[1] != config.image_W):
        p_image = process_image(image=p_image, aim_H=config.image_H, aim_W=config.image_W)
    p_image = np.transpose(p_image, axes=(2, 0, 1))
    p_image = np.expand_dims(p_image, axis=0)

    q_image = image.imread(target_image)
    if (q_image.shape[0] != config.image_H or q_image.shape[1] != config.image_W):
        q_image = process_image(image=q_image, aim_H=config.image_H, aim_W=config.image_W)
    q_image = np.transpose(q_image, axes=(2, 0, 1))
    q_image = np.expand_dims(q_image, axis=0)
    
    
    key = '%d_%d_%d' % (drive, s, t)
    filename = icp_path + '/' + key + '.npy'
    poses = np.genfromtxt(pose)
    all_odometry = poses[[s, t]]
    positions = [odometry_to_positions(odometry) for odometry in all_odometry]
    if not os.path.exists(filename):
        # work on the downsampled xyzs, 0.05m == 5cm
        _, sel0 = ME.utils.sparse_quantize(p_xyz / 0.05, return_index=True)
        _, sel1 = ME.utils.sparse_quantize(q_xyz / 0.05, return_index=True)

        M = (velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
             @ np.linalg.inv(velo2cam)).T
        xyz0_t = apply_transform(p_xyz[sel0], M)
        pcd0 = make_open3d_point_cloud(xyz0_t)
        pcd1 = make_open3d_point_cloud(q_xyz[sel1])
        reg = o3d.pipelines.registration.registration_icp(
            pcd0, pcd1, 0.2, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        pcd0.transform(reg.transformation)
        # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
        M2 = M @ reg.transformation
        # o3d.draw_geometries([pcd0, pcd1])
        # write to a file
        np.save(filename, M2)
    else:
        M2 = np.load(filename)
    T_gth = M2

    p_xyz_down, p_feature = extract_features(
        model,
        xyz=p_xyz,
        rgb=None,
        normal=None,
        voxel_size=config.voxel_size,
        device=device,
        skip_check=True,
        image=p_image,
        image_shape=image_shape,
        extrinsic=extrinsic,
        intrinsic=intrinsic
    )

    q_xyz_down, q_feature = extract_features(
        model,
        xyz=q_xyz,
        rgb=None,
        normal=None,
        voxel_size=config.voxel_size,
        device=device,
        skip_check=True,
        image=q_image,
        image_shape=image_shape,
        extrinsic=extrinsic,
        intrinsic=intrinsic
    )

    # get the evaluation metrix
    p_xyz_down = make_open3d_point_cloud(p_xyz_down)
    q_xyz_down = make_open3d_point_cloud(q_xyz_down)

    p_feature = p_feature.cpu().detach().numpy()
    p_feature = make_open3d_feature_from_numpy(p_feature)
    q_feature = q_feature.cpu().detach().numpy()
    q_feature = make_open3d_feature_from_numpy(q_feature)
    T_ransac = run_ransac(
        p_xyz_down,
        q_xyz_down,
        p_feature,
        q_feature,
        config.voxel_size
    )

    rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[:3, 3])
    rre = np.arccos((np.trace(T_ransac[:3, :3].T @ T_gth[:3, :3]) - 1) / 2)

    if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
        success = 1
    else:
        success = 0

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        pc1, pc2, config.voxel_size*1.5, T_ransac)

    return [success, T_ransac, info,rte,rre]


def register_one_rgbd_pair(data_path, s, t, model, config):
        
    [success, trans, info,rte,rre] = pose_estimation(data_path, s, t, model, config)
    return [success, trans, info,rte,rre]


def make_posegraph_for_fragment(out_path, data_path, n_pc, model, config):
    error_file_raw = os.path.join(out_path, "error_raw.txt")
    error_raw = np.zeros((n_pc,2))
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(trans_odometry))
    for s in range(0, n_pc-1):
        eid = min(s + 6, n_pc)
        for t in range(s + 1, eid):
            # odometry
            if t == s + 1:
                print(
                    "matching between frame : %d and %d"
                    % (s*2, t*2))
                [success, trans,
                 info,rte,rre] = register_one_rgbd_pair(data_path, s*2, t*2, model,config)
                error_raw[s, 0] = rte
                error_raw[s, 1] = rre
                print("RTE:%.5f RRE:%.5f"%(rte,rre))
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        trans_odometry_inv))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(s - 0,
                                                             t - 0,
                                                             trans,
                                                             info,
                                                             uncertain=False))

            # keyframe loop closure
            else:
                print(
                    "matching between frame : %d and %d"
                    % (s*2, t*2))
                [success, trans,
                 info,rte,rre] = register_one_rgbd_pair(data_path, s*2, t*2, model,config)

                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            s - 0, t - 0, trans, info, uncertain=True))
                    print("success!RTE:%.5f RRE:%.5f"%(rte,rre))
    o3d.io.write_pose_graph(
        os.path.join(out_path, "pose_graph_exp.json"),
        pose_graph)
    np.savetxt(error_file_raw, error_raw)


def integrate_frames_for_fragment(data_path, pose_graph_name, icp_path):
    drive = 9
    error_opt = np.zeros((795,2))
    gt_odometry = np.identity(4)
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    pcd_global = o3d.geometry.PointCloud()
    pcd_frame = o3d.geometry.PointCloud()

    camera_path = os.path.join(data_path, "dataset/sequences/%02d/calib.txt"%drive)
    result = get_calib_from_file(camera_path)
    extrinsic = result['Tr_velo2cam']
    intrinsic = result['P2']

    for i in range(len(pose_graph.nodes)):
        print("integrate  frame (%d of %d)." %(i + 1, len(pose_graph.nodes)))
        key = '%d_%d_%d' % (drive, i*2, i*2+2)
        filename = icp_path + '/' + key + '.npy'
        T_gth = np.load(filename)
        gt_odometry = np.dot(T_gth, gt_odometry)

        frame_name = os.path.join(data_path, "dataset/sequences/%02d/velodyne/%06d.bin"%(drive,i*2+2))
        point_in_lidar = read_bin(frame_name)

        image_name = frame_name.replace(".bin", ".png")
        color_image = image.imread(image_name)
        h, w = color_image.shape[:2]
        point_in_image, mask = get_fov_mask(point_in_lidar, extrinsic, intrinsic, h, w)
        valid_points = point_in_lidar[mask]
        
        colors = color_image[point_in_image[:, 1].astype(np.int32),
                         point_in_image[:, 0].astype(np.int32)]  # N x 3
        
        pcd_frame.points = o3d.utility.Vector3dVector(valid_points)
        pcd_frame.colors = o3d.utility.Vector3dVector(colors)
        pose = pose_graph.nodes[i].pose
        trans = np.linalg.inv(pose)

        rte = np.linalg.norm(trans[:3, 3] - T_gth[:3, 3])
        rre = np.arccos((np.trace(trans[:3, :3].T @ T_gth[:3, :3]) - 1) / 2)

        error_opt[i, 0] = rte
        error_opt[i, 1] = rre
        pcd_frame.transform(pose)
        pcd_global = pcd_global + pcd_frame
    pcd_name = "/data1/zhangliyuan/code/IMFNet_exp/system/results/fragments/exp3.pcd"
    o3d.io.write_point_cloud(pcd_name, pcd_global)

    error_file_opt = "/data1/zhangliyuan/code/IMFNet_exp/system/results/pose_graph/odometry9/error_opt.txt"
    np.savetxt(error_file_opt, error_opt)
    
def read_pose(pose_graph_name, position_name):
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    pose_file = np.zeros((795,12))
    for i in range(len(pose_graph.nodes)):
        pose = pose_graph.nodes[i].pose
        # trans = np.linalg.inv(pose)
        position = pose[:3,:].reshape(1,12)
        # print(position.shape)
        pose_file[i, :]=position
    np.savetxt(position_name, pose_file)

def read_gt_pose(icp_path, position_name):
    drive = 8
    pose_file = np.zeros((795,12))
    gt_odometry = np.identity(4)
    for i in range(795):
        key = '%d_%d_%d' % (drive, i*2, i*2+2)
        filename = icp_path + '/' + key + '.npy'
        T_gth = np.load(filename)
        gt_odometry = np.dot(T_gth, gt_odometry)
        gt_position = np.linalg.inv(gt_odometry)
        pose_file[i, :] =  gt_position[:3, :].reshape(1,12)
    np.savetxt(position_name, pose_file)

if __name__ == "__main__":
    # load the model
    # checkpoint_path = "/data1/zhangliyuan/code/IMFNet_exp/output/kitti/exp6/best_val_checkpoint_epoch_82_success_0.99.pth"
    # checkpoint = torch.load(checkpoint_path)
    # config = checkpoint['config']

    # num_feats = 1
    # Model = load_model(config.model)
    # model = Model(
    #     num_feats,
    #     config.model_n_out,
    #     bn_momentum=0.05,
    #     normalize_feature=config.normalize_feature,
    #     conv1_kernel_size=config.conv1_kernel_size,
    #     D=3,
    #     config=config
    # )
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    # device = torch.device("cuda")
    # model = model.to(device)

    out_path = "/data1/zhangliyuan/code/IMFNet_exp/system/results/pose_graph/odometry9"
    pose_graph_name = "/data1/zhangliyuan/code/IMFNet_exp/system/results/pose_graph/odometry9/pose_graph_exp.json"
    data_path = "/data1/zhangliyuan/code/IMFNet_exp/dataset/Kitti/Kitti"
    icp_path = "/data1/zhangliyuan/code/IMFNet_exp/dataset/Kitti/Kitti/icp"
    position_name = "/data1/zhangliyuan/code/IMFNet_exp/system/results/pose_graph/odometry9/position_raw.txt"
    # make_posegraph_for_fragment(out_path=out_path, data_path=data_path, n_pc=795, model=model, config=config)

    # integrate_frames_for_fragment(data_path=data_path, pose_graph_name=pose_graph_name, icp_path=icp_path)
    read_pose(pose_graph_name=pose_graph_name, position_name=position_name)
    # read_gt_pose(icp_path, position_name)

