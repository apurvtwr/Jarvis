import torch
from torch.autograd import Variable
import numpy as np
from depth_transformation_utils import Rotation, Intrinsics, Translation
from depth_map import DepthMap
from loss import DepthMapTransformation, ReconstructionLoss

device = 'cuda'
def getFrame() :

    H, W = 10, 10
    intrinsics = Variable(torch.FloatTensor([H, W, H/2, W/2, 0.]).reshape(1, 5), requires_grad = True)
    intrinsics = Intrinsics(intrinsics.to(device))

    rotation = Rotation(Variable(torch.FloatTensor([0, 0, 0]).reshape(1,3), requires_grad = True).to(device))
    translation = Translation(Variable(torch.rand(1, 3, H, W), requires_grad = True).to(device))
    image = Variable(torch.rand(1, 3, H, W), requires_grad = True).to(device)
    depth = DepthMap(Variable(torch.rand(1, H, W), requires_grad = True).to(device))
    depth.transform(translation, rotation, intrinsics)
    return DepthMapTransformation(image, depth, intrinsics, rotation, translation)

frameA = getFrame()


frameB = getFrame()
rc_loss = ReconstructionLoss(frameA, frameB)
loss = rc_loss()
loss.backward()