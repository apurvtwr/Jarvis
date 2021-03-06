"""
Post conversion translation and rotation we want to compute what 
the image will look like. to do this we will need the corresponding
grid location from where to sample to get each pixel value
"""

import torch
import torch.nn.functional as F

def field_sampling(field, grid) :
    """
    field [N, C, H, W] original field to sample from. 
    grid [N, 2, H, W] is grid of x,y position for each location to sample from 
    """
    no_channel_case = False
    if len(field.shape) == 3 :
        # No channel case 
        no_channel_case = True
        N, H, W = field.shape
        x_coord = (2 * grid[:, 0, :, :]/(W-1)) - 1
        y_coord = (2 * grid[:, 1, :, :]/(H-1)) - 1

        coord_original = torch.stack((x_coord, y_coord), dim=1)
        coord = coord_original.permute(0, 2, 3, 1)
        return F.grid_sample(field.reshape(N, 1, H, W), coord, align_corners=False)[:, 0, :, :]

    else :
        N, C, H, W = field.shape
        x_coord = (2 * grid[:, 0, :, :]/(W-1)) - 1
        y_coord = (2 * grid[:, 1, :, :]/(H-1)) - 1
        coord = torch.stack((x_coord, y_coord), dim=1).permute(0, 2, 3, 1)
        return F.grid_sample(field, coord, align_corners=False)