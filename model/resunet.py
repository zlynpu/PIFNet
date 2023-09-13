# coding = utf-8
import os
import sys
import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

ROOT_DIR = os.path.abspath('/data1/zhangliyuan/code/IMFNet_exp')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from model.common import get_norm

from model.residual_block import get_block
from model.Img_Encoder import ImageEncoder
from model.Img_Decoder import ImageDecoder
from model.Local_Fusion import Local_Atten_Fusion_Conv
from model.attention_fusion import AttentionFusion

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample
import math

class ResUNet2(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]

  # IMG_CHANNELS = [None, 64, 128, 256, 512]
  IMG_CHANNELS = [None, 0, 0, 0, 0]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=None,
               conv1_kernel_size=None,
               D=3,
               config=None):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE

    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    IMG_CHANNELS = self.IMG_CHANNELS

    FUSION_IMG_CHANNELS = [64, 64, 32]
    FUSION_POINT_CHANNELS = [64, 128, 64]

    self.normalize_feature = normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)
    self.block1 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)
    self.block2 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)
    self.block3 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)
    self.block4 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    # fusion attention
    self.attention_fusion = AttentionFusion(
        dim=128,  # the image channels
        depth=0,  # depth of net (self-attention - Processing的数量)
        latent_dim=CHANNELS[4],  # the PC channels
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=8,  # number of heads for latent self attention, 8
        cross_dim_head=int(CHANNELS[4]/2),  # number of dimensions per cross attention head
        latent_dim_head=int(CHANNELS[4]/2),  # number of dimensions per latent self attention head
    )

    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4] + IMG_CHANNELS[1],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3] + IMG_CHANNELS[2],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2] + IMG_CHANNELS[3],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        dimension=D)

    # image
    self.img_encoder = ImageEncoder()
    self.img_decoder = ImageDecoder()

    #local fusion
    self.fusion_attention = nn.ModuleList()
    for i in range(len(FUSION_IMG_CHANNELS)): 
      self.fusion_attention.append(Local_Atten_Fusion_Conv(inplanes_I=FUSION_IMG_CHANNELS[i], 
                                                           inplanes_P=FUSION_POINT_CHANNELS[i],
                                                           outplanes=FUSION_POINT_CHANNELS[i]))

    # self.fusion_conv1 = torch.nn.Conv1d(128, 64, 1)
    # self.fusion_bn1 = torch.nn.BatchNorm1d(64)

    # self.fusion_conv2 = torch.nn.Conv1d(192, 128, 1)
    # self.fusion_bn2 = torch.nn.BatchNorm1d(128)

    # self.fusion_conv3 = torch.nn.Conv1d(96, 64, 1)
    # self.fusion_bn3 = torch.nn.BatchNorm1d(64)
    

  def forward(self, x, image, image_shape, extrinsic, intrinsic):
    # I1,I2,I3,I_global = self.img_encoder(image)
    I0,I1,I2 = self.img_encoder(image)
    image_fusion = self.img_decoder(I0,I1,I2)

    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = self.block1(out_s1)
    out = MEF.relu(out_s1)
    

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out_s2 = self.block2(out_s2)
    out = MEF.relu(out_s2)
    fusion2 = self.local_fusion(coord=out.C, feature=out.F, image_feature=I0, image_shape=image_shape, extrinsic=extrinsic, intrinsic=intrinsic, id=0)
    # (B, 64, N)
    fusion2 = fusion2.reshape(-1,fusion2.shape[-1])
    # (64, N*B)
    out._F = fusion2.permute(1,0)
  
    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out_s4 = self.block3(out_s4)
    out = MEF.relu(out_s4)
    fusion4 = self.local_fusion(coord=out.C, feature=out.F, image_feature=I1, image_shape=image_shape, extrinsic=extrinsic, intrinsic=intrinsic, id=1)
    # fusion4 = F.relu(self.fusion_bn2(self.fusion_conv2(gather4)))
    fusion4 = fusion4.reshape(-1,fusion4.shape[-1])
    # 
    out._F = fusion4.permute(1,0)

    out_s8 = self.conv4(out)
    out_s8 = self.norm4(out_s8)
    out_s8 = self.block4(out_s8)
    out = MEF.relu(out_s8)

    # fusion-attention
    out._F = self.transformer(images=I2, F=out.F,xyz = out.C)

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out = self.block4_tr(out)
    out_s4_tr = MEF.relu(out)

    out = ME.cat(out_s4_tr, out_s4)
    # out._F = self.af(I1,I_global,out.F,af_flag=1)
    del out_s4_tr
    del out_s4

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out = self.block3_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat(out_s2_tr, out_s2)
    # out._F = self.af(I2,I_global,out.F,af_flag=2)
    del out_s2_tr
    del out_s2

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out = self.block2_tr(out)
    out_s1_tr = MEF.relu(out)

  
    out = ME.cat(out_s1_tr, out_s1)
    # out._F = self.af(I3, I_global, out.F, af_flag=3)
    del out_s1_tr
    del out_s1

    out = self.conv1_tr(out)
    out = MEF.relu(out)
    # print('final',out.F.shape)
    fusion_final = self.local_fusion(coord=out.C, feature=out.F, image_feature=image_fusion, image_shape=image_shape, extrinsic=extrinsic, intrinsic=intrinsic, id=2)
    # fusion_final = F.relu(self.fusion_bn3(self.fusion_conv3(gather_final)))
    fusion_final = fusion_final.reshape(-1,fusion_final.shape[-1])
    out._F = fusion_final.permute(1,0)

    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coordinate_map_key=out.coordinate_map_key,
          coordinate_manager=out.coordinate_manager
      )
    else:
      return out

  def transformer(self,images,F,xyz):

      # batch ----- batch
      lengths = []
      max_batch = torch.max(xyz[:, 0])
      for i in range(max_batch + 1):
          length = torch.sum(xyz[:, 0] == i)
          lengths.append(length)
      # batch ----- batch

      ps = []
      start = 0
      end = 0
      for length,image in zip(lengths,images):

          # pc ------- pc
          end += length
          P_att = torch.unsqueeze(F[start:end, :], dim=0)  # [B,M,C]
          # pc ------- pc

          # image ------- image
          image = torch.unsqueeze(image,dim=0)
          B,C,H,W = image.shape
          image = image.view(B,C,H*W)
          image = image.permute(0,2,1) # [B,H*W,C]
          # image ------- image

          # fusion attention
          P_att = self.attention_fusion(image,queries_encoder = P_att)
          P_att = torch.squeeze(P_att)
          start += length
          ps.append(P_att)
          # fusion attention

      F = torch.cat(ps, dim=0)

      return F
  
  def local_fusion(self, coord, feature, image_feature, image_shape, extrinsic, intrinsic, id, voxel_size=0.3):
    # batch ----- batch
    lengths = []
    max_batch = torch.max(coord[:, 0])
    for i in range(max_batch + 1):
        length = torch.sum(coord[:, 0] == i)
        lengths.append(length)

    device = torch.device('cuda')
    fusion_feature_batch = []
    start = 0
    end = 0
    batch_id = 0
    
    for length in lengths:
        end += length
        point_coord = coord[start:end, 1:] * voxel_size
        point_feature = feature[start:end, :].unsqueeze(0)
        point_feature = point_feature.permute(0,2,1)

        point_z = point_coord[:, -1]
        # print('pointcoord',point_coord.shape)
        one = torch.ones((point_coord.shape[0],1)).to(device)
        # print('one',one.shape)
        point_in_lidar = torch.cat([point_coord, one],1).t()
        # print('point_in_lidar',point_in_lidar.shape)

        point_in_camera = extrinsic[batch_id, :, :].mm(point_in_lidar)
        point_in_image = intrinsic[batch_id, :, :].mm(point_in_camera)/point_z
        point_in_image = point_in_image.t()
        point_in_image[:, -1] = point_z

        point_in_image[:, 0] = point_in_image[:, 0] * 2 / image_shape[batch_id, 0] - 1
        point_in_image[:, 1] = point_in_image[:, 1] * 2 / image_shape[batch_id, 1] - 1

        point_in_image = point_in_image.unsqueeze(0)

        feature_map = image_feature[batch_id, :, :, :].unsqueeze(0)
        # print('feature_map_shape:',feature_map.shape)
        xy = point_in_image[:, :, :-1].unsqueeze(1)
        # print('xy:',xy.shape)
        img_feature = grid_sample(feature_map, xy)
        img_feature = img_feature.squeeze(2)
        
        fusion_feature = self.fusion_attention[id](point_features=point_feature, img_features=img_feature)
        # fusion_feature = torch.cat([img_feature, point_feature], dim=1)
        # print('fusion_feature:', fusion_feature.shape)

        fusion_feature_batch.append(fusion_feature)
        start += length
        batch_id += 1
    
    fusion_feature_batch = torch.cat(fusion_feature_batch,2)
    # print('fusion_feature_batch:',fusion_feature_batch.shape)
    return fusion_feature_batch


class ResUNetBN2(ResUNet2):
  NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 128, 256]
  TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetIN2(ResUNet2):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2B(ResUNetBN2B):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2C(ResUNetBN2C):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2D(ResUNetBN2D):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2E(ResUNetBN2E):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'
