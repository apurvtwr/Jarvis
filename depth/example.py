import torch
import numpy as np
from depth_transformation_utils import Rotation, Intrinsics, Translation
from depth_map import DepthMap

H, W = 10, 10
intrinsics = Intrinsics(torch.FloatTensor([H, W, H/2, W/2, 0.]).reshape(1, 5))
rotation = Rotation(torch.FloatTensor([np.pi/2, 0, 0]).reshape(1,3))
translation = Translation(torch.zeros(1, 3, H, W))
depth = DepthMap(torch.ones(1, H, W))

depth.transform(translation, rotation, intrinsics)
print(depth.depth)