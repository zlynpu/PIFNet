import torch
import numpy as np
import MinkowskiEngine as ME
import time

def _hash(arr, M):
  if isinstance(arr, np.ndarray):
    N, D = arr.shape
  else:
    N, D = len(arr[0]), len(arr)

  hash_vec = np.zeros(N, dtype=np.int64)
  for d in range(D):
    if isinstance(arr, np.ndarray):
      hash_vec += arr[:, d] * M**d
    else:
      hash_vec += arr[d] * M**d
  return hash_vec


def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True,
                     image=None,
                     image_shape=None,
                     extrinsic=None,
                     intrinsic=None
                     ):
  '''
  xyz is a N x 3 matrix
  rgb is a N x 3 matrix and all color must range from [0, 1] or None
  normal is a N x 3 matrix and all normal range from [-1, 1] or None

  if both rgb and normal are None, we use Nx1 one vector as an input

  if device is None, it tries to use gpu by default

  if skip_check is True, skip rigorous checks to speed up

  model = model.to(device)
  xyz, feats = extract_features(model, xyz)
  '''
  if is_eval:
    model.eval()

  if not skip_check:
    assert xyz.shape[1] == 3

    N = xyz.shape[0]
    if rgb is not None:
      assert N == len(rgb)
      assert rgb.shape[1] == 3
      if np.any(rgb > 1):
        raise ValueError('Invalid color. Color must range from [0, 1]')

    if normal is not None:
      assert N == len(normal)
      assert normal.shape[1] == 3
      if np.any(normal > 1):
        raise ValueError('Invalid normal. Normal must range from [-1, 1]')

  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  feats = []
  if rgb is not None:
    # [0, 1]
    feats.append(rgb - 0.5)

  if normal is not None:
    # [-1, 1]
    feats.append(normal / 2)

  if rgb is None and normal is None:
    feats.append(np.ones((len(xyz), 1)))

  feats = np.hstack(feats)

  # Voxelize xyz and feats
  coords = np.floor(xyz / voxel_size)
  coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
  # coords,inds = coords[:5000],inds[:5000]
  # Convert to batched coords compatible with ME
  coords = ME.utils.batched_coordinates([coords])
  return_coords = xyz[inds]

  feats = feats[inds]

  feats = torch.tensor(feats, dtype=torch.float32)
  coords = torch.tensor(coords, dtype=torch.int32)



  stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

  image = torch.as_tensor(image,dtype=torch.float32,device=device)
  image_shape = torch.as_tensor(image_shape,dtype=torch.float32,device=device)
  extrinsic = torch.as_tensor(extrinsic,dtype=torch.float32,device=device)
  intrinsic = torch.as_tensor(intrinsic,dtype=torch.float32,device=device)

  # start = time.time()
  F = model(stensor,image,image_shape,extrinsic,intrinsic).F
  # end = time.time()
  # t = end - start

  return return_coords,F
