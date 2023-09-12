import os
import sys

import matplotlib.image as image
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

ROOT_DIR = os.path.abspath('/data1/zhangliyuan/code/IMFNet_exp')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from util.calibration import get_calib_from_file

def read_bin(bin_path, intensity=False):
    """
    读取kitti bin格式文件点云
    :param bin_path:   点云路径
    :param intensity:  是否要强度
    :return:           numpy.ndarray `N x 3` or `N x 4`
    """
    lidar_points = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    if not intensity:
        lidar_points = lidar_points[:, :3]
    return lidar_points


def image2camera(point_in_image, intrinsic):
    """
    图像系到相机系反投影
    :param point_in_image: numpy.ndarray `N x 3` (u, v, z)
    :param intrinsic: numpy.ndarray `3 x 3` or `3 x 4`
    :return: numpy.ndarray `N x 3` (x, y, z)
    u = fx * X/Z + cx
    v = fy * Y/Z + cy
    X = (u - cx) * Z / fx
    Y = (v - cy) * z / fy
       [[fx, 0,  cx, -fxbi],
    K=  [0,  fy, cy],
        [0,  0,  1 ]]
    """
    if intrinsic.shape == (3, 3):  # 兼容kitti的P2, 对于没有平移的intrinsic添0
        intrinsic = np.hstack((intrinsic, np.zeros((3, 1))))

    u = point_in_image[:, 0]
    v = point_in_image[:, 1]
    z = point_in_image[:, 2]
    x = ((u - intrinsic[0, 2]) * z - intrinsic[0, 3]) / intrinsic[0, 0]
    y = ((v - intrinsic[1, 2]) * z - intrinsic[1, 3]) / intrinsic[1, 1]
    point_in_camera = np.vstack((x, y, z))
    return point_in_camera


def lidar2camera(point_in_lidar, extrinsic):
    """
    雷达系到相机系投影
    :param point_in_lidar: numpy.ndarray `N x 3`
    :param extrinsic: numpy.ndarray `4 x 4`
    :return: point_in_camera numpy.ndarray `N x 3`
    """
    point_in_lidar = np.hstack((point_in_lidar, np.ones(shape=(point_in_lidar.shape[0], 1)))).T
    point_in_camera = np.matmul(extrinsic, point_in_lidar)[:-1, :]  # (X, Y, Z)
    point_in_camera = point_in_camera.T
    return point_in_camera


def camera2image(point_in_camera, intrinsic):
    """
    相机系到图像系投影
    :param point_in_camera: point_in_camera numpy.ndarray `N x 3`
    :param intrinsic: numpy.ndarray `3 x 3` or `3 x 4`
    :return: point_in_image numpy.ndarray `N x 3` (u, v, z)
    """
    point_in_camera = point_in_camera.T
    point_z = point_in_camera[-1]

    if intrinsic.shape == (3, 3):  # 兼容kitti的P2, 对于没有平移的intrinsic添0
        intrinsic = np.hstack((intrinsic, np.zeros((3, 1))))

    point_in_camera = np.vstack((point_in_camera, np.ones((1, point_in_camera.shape[1]))))
    point_in_image = (np.matmul(intrinsic, point_in_camera) / point_z)  # 向图像上投影
    point_in_image[-1] = point_z
    point_in_image = point_in_image.T
    return point_in_image


def lidar2image(point_in_lidar, extrinsic, intrinsic):
    """
    雷达系到图像系投影  获得(u, v, z)
    :param point_in_lidar: numpy.ndarray `N x 3`
    :param extrinsic: numpy.ndarray `4 x 4`
    :param intrinsic: numpy.ndarray `3 x 3` or `3 x 4`
    :return: point_in_image numpy.ndarray `N x 3` (u, v, z)
    """
    point_in_camera = lidar2camera(point_in_lidar, extrinsic)
    point_in_image = camera2image(point_in_camera, intrinsic)
    return point_in_image


def get_fov_mask(point_in_lidar, extrinsic, intrinsic, h, w):
    """
    获取fov内的点云mask, 即能够投影在图像上的点云mask
    :param point_in_lidar:   雷达点云 numpy.ndarray `N x 3`
    :param extrinsic:        外参 numpy.ndarray `4 x 4`
    :param intrinsic:        内参 numpy.ndarray `3 x 3` or `3 x 4`
    :param h:                图像高 int
    :param w:                图像宽 int
    :return: point_in_image: (u, v, z)  numpy.ndarray `N x 3`
    :return:                 numpy.ndarray  `1 x N`
    """
    point_in_image = lidar2image(point_in_lidar, extrinsic, intrinsic)
    front_bound = point_in_image[:, -1] > 0
    point_in_image[:, 0] = np.round(point_in_image[:, 0])
    point_in_image[:, 1] = np.round(point_in_image[:, 1])
    u_bound = np.logical_and(point_in_image[:, 0] >= 0, point_in_image[:, 0] < w)
    v_bound = np.logical_and(point_in_image[:, 1] >= 0, point_in_image[:, 1] < h)
    uv_bound = np.logical_and(u_bound, v_bound)
    mask = np.logical_and(front_bound, uv_bound)
    return point_in_image[mask], mask


def get_point_in_image(point_in_lidar, extrinsic, intrinsic, h, w):
    """
    把雷达点云投影到图像上, 且经过筛选.  用这个就可以了.
    :param point_in_lidar:   雷达点云 numpy.ndarray `N x 3`
    :param extrinsic:        外参 numpy.ndarray `4 x 4`
    :param intrinsic:        内参 numpy.ndarray `3 x 3` or `3 x 4`
    :param h:                图像高 int
    :param w:                图像宽 int
    :return: point_in_image  (u, v, z)  numpy.ndarray `M x 3`  筛选掉了后面的点和不落在图像上的点
    :return: depth_image     numpy.ndarray `image_h x image_w` 深度图
    """
    point_in_image, mask = get_fov_mask(point_in_lidar, extrinsic, intrinsic, h, w)
    depth_image = np.zeros(shape=(h, w), dtype=np.float32)
    depth_image[point_in_image[:, 1].astype(np.int32), point_in_image[:, 0].astype(np.int32)] = point_in_image[:, 2]
    return point_in_image, depth_image

if __name__ == '__main__':
    image_path = '/data1/zhangliyuan/code/IMFNet_exp/dataset/Kitti/Kitti/dataset/sequences/04/velodyne/000000.png'
    bin_path = '/data1/zhangliyuan/code/IMFNet_exp/dataset/Kitti/Kitti/dataset/sequences/04/velodyne/000000.bin'
    calib_path = '/data1/zhangliyuan/code/IMFNet_exp/dataset/Kitti/Kitti/dataset/sequences/04/calib.txt'
    point_in_lidar = read_bin(bin_path)
    color_image = image.imread(image_path)
    result = get_calib_from_file(calib_path)
    intrinsic = result['P2']
    extrinsic = result['Tr_velo2cam']                      # 外参
    h, w = color_image.shape[:2]  # 图像高和宽

    point_in_image, mask = get_fov_mask(point_in_lidar, extrinsic, intrinsic, h, w)
    valid_points = point_in_lidar[mask]

    # 获取颜色
    colors = color_image[point_in_image[:, 1].astype(np.int32),
                         point_in_image[:, 0].astype(np.int32)]  # N x 3
    color_pcd = o3d.geometry.PointCloud()
    color_pcd.points = o3d.utility.Vector3dVector(valid_points)
    color_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud('/data1/zhangliyuan/code/IMFNet_exp/result/color_pcd.pcd',color_pcd)


