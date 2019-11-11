import torch
import numpy as np
from depth_transformation_utils import Rotation, Intrinsics, Translation
from depth_map import DepthMap
from loss import DepthMapTransformation, ReconstructionLoss

H, W = 10, 10
intrinsics = Intrinsics(torch.FloatTensor([H, W, H/2, W/2, 0.]).reshape(1, 5))
rotation = Rotation(torch.FloatTensor([0, 0, 0]).reshape(1,3))
translation = Translation(torch.rand(1, 3, H, W))
image = torch.rand(1, 3, H, W)
depth = DepthMap(torch.rand(1, H, W))
depth.transform(translation, rotation, intrinsics)

frameA = DepthMapTransformation(image, depth, intrinsics, rotation, translation)

intrinsics = Intrinsics(torch.FloatTensor([H, W, H/2, W/2, 0.]).reshape(1, 5))
rotation = Rotation(torch.FloatTensor([0, 0, 0]).reshape(1,3))
translation = Translation(torch.rand(1, 3, H, W))
image = torch.rand(1, 3, H, W)
depth = DepthMap(torch.rand(1, H, W))
depth.transform(translation, rotation, intrinsics)

frameB = DepthMapTransformation(image, depth, intrinsics, rotation, translation)

rc_loss = ReconstructionLoss(frameA, frameB)
loss = rc_loss()
print(loss)