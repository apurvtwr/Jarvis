"""
Paper : https://arxiv.org/pdf/1904.04998.pdf
"""
import torch
import torch.nn.functional as F
from sampler import field_sampling

class DepthMapTransformation(object) :

    """Summary
    
    Attributes:
        depth_map (TYPE): Description
        image (TYPE): Description
        intrinsics (TYPE): Description
        rotation (TYPE): Description
        translation (TYPE): Description
    """
    
    def __init__(self, image, depth_map, intrinsics, rotation, translation) :
        """Summary
        
        Args:
            image (TYPE): Description
            depth_map (TYPE): Description
            intrinsics (TYPE): Description
            rotation (TYPE): Description
            translation (TYPE): Description
        """
        self.depth_map = depth_map
        self.image = image
        self.intrinsics = intrinsics
        self.rotation = rotation
        self.translation = translation


class ReconstructionLoss(object) :
    """
                |----------------------------|
    Frame A --> | K, RGB, R_AB, T_AB, Z_A    | --> Frame B
                |----------------------------|
    
    Attributes:
        transformation_AB (TYPE): Description
        transformation_BA (TYPE): Description
    """

    def __init__(self, transformation_AB, transformation_BA) :
        """Summary
        
        Args:
            transformation_AB (TYPE): Description
            transformation_BA (TYPE): Description
        """
        self.transformation_AB = transformation_AB
        self.transformation_BA = transformation_BA

    def __call__(self) :
        """
        Since the loss is not symmetric going from frame A to frame B
        we have to make sure we compute the loss both ways 
        sum it up and the use that as the final loss
        """
        lossAB = self.loss(self.transformation_AB, self.transformation_BA)
        lossBA = self.loss(self.transformation_BA, self.transformation_AB)

    @classmethod
    def loss(cls, frame1, frame2) :
        """
        there are 3 types of loss to compute:
            i) RGB : L1 loss
            ii) motion field cyclic consistency loss : L1 loss
            iii) Structural symmetry loss
            The first thing we want to compute is what pixels in frame1
            end up being closer camera post transformation than frame2
        
        Args:
            frame1 (TYPE): Description
            frame2 (TYPE): Description
        """
        frame2_resampled_depth = field_sampling(frame2.depth_map.depth, frame1.grid)
        frame2_resampled_image = field_sampling(frame2.image, frame1.grid)
        frame1_closer = (frame2_resampled_depth > frame1.depth_map.depth).float() * frame1.mask

        rgb_loss = cls.rgb_loss(frame1, frame2_resampled_image, frame1_closer)
        motion_field_loss = cls.motion_field_loss(frame1, frame2, frame1_closer)
        ssim_loss = cls.ssim_loss(frame1, frame2_resampled_depth, frame1_closer)

    @classmethod
    def motion_field_loss(frame1, frame2, frame1_closer) :
        """Summary
        
        Args:
            frame1 (TYPE): Description
            frame2 (TYPE): Description
            frame1_closer (TYPE): Description
        """
        pass

    @classmethod
    def ssim_loss(cls, frame1, frame2_resampled_depth, 
        frame2_resampled_image, frame1_closer) :
        """
        We are going to compute the RMSE of depth error,
        then for each pixel compute the weight with which to normalise
        the ssim as :
        weight = depth_mse/(depth_mse + (depth_error ** 2))

        Case average high depth_error:
            depth_mse ~ depth_error => weight = 1/2

        Case correct depth_error :
            if depth_mse >> depth_error, weight = 1
            if depth_mse << depth_error, weight = 0
        Args:
            frame1 (TYPE): Description
            frame2_resampled_depth (TYPE): Description
            frame1_closer (TYPE): Description
        """
        N, H, W = frame1_closer.shape
        depth_error_MSE = weighted_average(
            torch.pow(frame2_resampled_depth - frame1.depth_map.depth, 2),
            frame1_closer
            ) + 1e-4

        depth_weight = torch.div(
            depth_error_MSE,
            (torch.pow(frame2_resampled_depth - frame1.depth_map.depth, 2) + 
                depth_error_MSE) * frame1.mask).detach()

        ssim_error, avg_weight = weighted_ssim(frame2_resampled_image, 
            frame1.image, depth_weight)


    @classmethod
    def rgb_loss(cls, frame1, frame2_resampled_image, frame1_closer) :
        """
        Computes RGB loss 
        
        Args:
            frame1 (TYPE): Description
            frame2_resampled_image (TYPE): Description
            frame1_closer (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        N, H, W = frame1_closer.shape
        rgb_diff = (frame1.image - frame2_resampled_image) * frame1_closer.reshape(N, 1, H, W).expand(N, 3, H, W)
        return torch.mean(rgb_diff)

def weighted_ssim(frame1_image, frame2_image, weight, weight_epsilon = 0.01) :

    N, H, W = frame1_depth.shape
    c2 = 9e-6
    avg_weight = average_pool3x3(weight.reshape(N, 1, H, W))[:, 0, :, :] 
    weight_plus_eps = weight + weight_epsilon
    inv_avg_weight = 1.0/(avg_weight + weight_epsilon)

    def weighted_avg_pool_3x3(z) :
        n, c, h, w = z.shape
        weighted_z = z * weight_plus_eps.reshape(n, 1, h, w).expand(n, c, h, w)
        wighted_avg = average_pool3x3(weighted_z)
        n, c, h, w = wighted_avg.shape
        return wighted_avg * inv_avg_weight.reshape(n, 1, h, w).expand(n,c,h,w)

    mean_x = weighted_avg_pool_3x3(frame1_image)
    mean_y = weighted_avg_pool_3x3(frame2_image)
    sigma_x = weighted_avg_pool_3x3(torch.pow(frame1_image, 2)) - torch.pow(mean_x, 2)
    sigma_y = weighted_avg_pool_3x3(torch.pow(frame2_image, 2)) - torch.pow(mean_y, 2)
    sigma_xy = weighted_avg_pool3x3(frame1_image * frame2_image) - mean_x * mean_y

    ssim_n = (2 * sigma_xy + c2)
    ssim_d = (sigma_x + sigma_y + c2)
    result = torch.div(ssim_n, ssim_d)
    return result.clamp(min=0, max=1), avg_weight

def weighted_average(x, w, epsilon=1.) :
    """
    Computes Sum(x * w)/ (sum(w) + epsilon)
    Keeps the dimensions
    Args:
        x (TYPE): Description
        w (TYPE): Description
        epsilon (float, optional): Description
    
    Returns:
        TYPE: Description
    """
    weighted_sum = torch.sum(x * w, dim=(1,2), keepdim=True)
    sum_weights = torch.sum(w, dim=(1, 2), keepdim=True)
    return torch.div(weighted_sum, sum_weights + epsilon)


def average_pool3x3(w) :
    return F.avg_pool2d(w, kernel_size=3, stride=1)


