"""
Paper : https://arxiv.org/pdf/1904.04998.pdf
"""
import torch
import torch.nn.functional as F
from sampler import field_sampling

class Loss(object) :
    rgb_weight = 0.85
    rot_weight = 1e-2
    trans_weight = 1e-1
    distance_weight = 0.
    ssim_weight = 3.
    
    def __init__(self, rgb_loss, rot_error, 
        trans_error, ssim_error) :
        self.rgb_loss = rgb_loss
        self.rot_error = rot_error
        self.trans_error = trans_error
        self.ssim_error = ssim_error
        self.distance_loss = None

    def loss(self) :
        return self.rgb_weight * self.rgb_loss.value + self.rot_weight * self.rot_error.value + self.trans_weight * self.trans_error.value + self.ssim_weight * self.ssim_error.value + self.distance_weight * self.distance_loss.value

class DIST_Loss() :
    def __init__(self, distance_lossAB, distance_lossBA) :
        self.distance_lossAB = distance_lossAB
        self.distance_lossBA = distance_lossBA
        
    @property
    def value(self) :
        return torch.mean(self.distance_lossAB + self.distance_lossBA)
    
class SSIM_Loss() :
    def __init__(self, ssim_error, avg_weight) :
        self.ssim_error = ssim_error
        self.avg_weight = avg_weight
    
    @property
    def value(self) :
        return 1 - torch.mean(self.ssim_error * self.avg_weight)

class RGB_Loss() :
    def __init__(self, rgb_diff, frame_resampled) :
        self.rgb_diff = rgb_diff
        self.frame_resampled = frame_resampled
        
    @property
    def value(self) :
        return torch.mean(self.rgb_diff)

class ROT_Loss() :
    def __init__(self, R, R1, R2) :
        self.R = R
        self.R1 = R1
        self.R2 = R2
    
    @property
    def value(self) :
        N, H, W, _, _ = self.R.shape
        eye = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(N, H, W, 3, 3).to(self.R.device)
        rot_diff = torch.mean(torch.pow(self.R - eye, 2), dim=[3, 4])
        rot_scale = torch.mean(torch.pow(self.R1 - eye, 2), dim=[3, 4]) + torch.mean(torch.pow(self.R2 - eye, 2), dim=[3, 4]) + 1e-24
        rot_error = torch.mean(torch.div(rot_diff, rot_scale))
        return rot_error

class TRANS_Loss() :
    def __init__(self, T, T1, T2, frame1_closer):
        self.T = T
        self.T1 = T1
        self.T2 = T2
        self.frame1_closer = frame1_closer
        
    @property
    def value(self):
        mag_T = torch.sum(torch.pow(self.T, 2), dim=-1)
        mag_T1 = torch.sum(torch.pow(self.T1, 2), dim=-1)
        mag_T2 = torch.sum(torch.pow(self.T2, 2), dim=-1)

        trans_error = torch.mean(torch.div(self.frame1_closer * mag_T, 
            mag_T1 + mag_T2 + 1e-24))
        return trans_error
    

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

    def __init__(self, transformation_AB, transformation_BA, distance) :
        """Summary
        
        Args:
            transformation_AB (TYPE): Description
            transformation_BA (TYPE): Description
        """
        self.transformation_AB = transformation_AB
        self.transformation_BA = transformation_BA
        self.distance = distance

    def __call__(self) :
        """
        Since the loss is not symmetric going from frame A to frame B
        we have to make sure we compute the loss both ways 
        sum it up and the use that as the final loss
        """
        lossAB = self.loss(self.transformation_AB, self.transformation_BA)
        lossBA = self.loss(self.transformation_BA, self.transformation_AB)
        distance_loss = self.distance_loss(self.transformation_AB, self.transformation_AB, self.distance)
        lossAB.distance_loss = distance_loss
        lossBA.distance_loss = distance_loss
        return lossAB, lossBA

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
        frame2_resampled_depth = field_sampling(frame2.depth_map.depth, frame1.depth_map.grid)
        frame2_resampled_image = field_sampling(frame2.image, frame1.depth_map.grid)
        frame1_closer = (frame2_resampled_depth >= frame1.depth_map.depth).float() * frame1.depth_map.mask

        rgb_loss = cls.rgb_loss(frame1, frame2_resampled_image, frame1_closer)
        rot_err, trans_err = cls.motion_field_loss(frame1, frame2, frame1_closer)
        ssim_loss = cls.ssim_loss(frame1, frame2_resampled_depth, 
            frame2_resampled_image, frame1_closer)
        return Loss(rgb_loss, rot_err, trans_err, ssim_loss)

    @classmethod
    def distance_loss(cls, frame1, frame2, distance) :
        distance1 = frame1.translation.bg_motion
        distance2 = frame2.translation.bg_motion
        diff1 = torch.abs(distance1 - distance)/ (torch.abs(distance) + torch.abs(distance1) + 1e-4)
        diff2 = torch.abs(distance2 - distance)/ (torch.abs(distance) + torch.abs(distance2) + 1e-4)
        return DIST_Loss(diff1, diff2)
        
    
    @classmethod
    def motion_field_loss(cls, frame1, frame2, frame1_closer) :
        """
        Building a 4D transform matrix from each rotation and translation, and
        multiplying the two, we'd get:
        
        (  R2   t2) . (  R1   t1)  = (R2R1    R2t1 + t2)
        (0 0 0  1 )   (0 0 0  1 )    (0 0 0       1    )
        
        Where each R is a 3x3 matrix, each t is a 3-long column vector, and 0 0 0 is
        a row vector of 3 zeros. We see that the total rotation is R2*R1 and the t
        total translation is R2*t1 + t2.
        
        Args:
            frame1 (TYPE): Description
            frame2 (TYPE): Description
            frame1_closer (TYPE): Description
        """
        N, H, W = frame1_closer.shape
        R1 = frame1.rotation.value.reshape(N, 1, 1, 3, 3).expand(N, H, W, 3, 3)
        R2 = frame2.rotation.value.reshape(N, 1, 1, 3, 3).expand(N, H, W, 3, 3)
        T1 = frame1.translation.value.permute(0, 2, 3, 1)
        T2 = frame2.translation.value.permute(0, 2, 3, 1)
        
        R = torch.matmul(R2, R1)
        T = torch.einsum('bijkl,bijl->bijk', R2, T1) + T2
        rot_error = ROT_Loss(R, R1, R2)
        
        trans_error = TRANS_Loss(T, T1, T2, frame1_closer)
    
        return rot_error, trans_error


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

        depth_weight = (torch.div(
            depth_error_MSE,
            (torch.pow(frame2_resampled_depth - frame1.depth_map.depth, 2) + 
                depth_error_MSE)) * frame1.depth_map.mask).detach()
        ssim_error, avg_weight = weighted_ssim(frame2_resampled_image, 
            frame1.image, depth_weight)
        n, _, h, w = ssim_error.shape
        return SSIM_Loss(ssim_error, avg_weight.reshape(n, 1, h, w))


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
        rgb_diff = torch.abs(frame1.image - frame2_resampled_image) * frame1_closer.reshape(N, 1, H, W).expand(N, 3, H, W)
        return RGB_Loss(rgb_diff, frame2_resampled_image)

def weighted_ssim(frame1_image, frame2_image, weight, weight_epsilon = 0.01) :

    N, H, W = weight.shape
    c2 = 9e-6
    avg_weight = average_pool(weight.reshape(N, 1, H, W))[:, 0, :, :] 
    weight_plus_eps = weight + weight_epsilon
    inv_avg_weight = 1.0/(avg_weight + weight_epsilon)

    def weighted_avg_pool(z) :
        n, c, h, w = z.shape
        weighted_z = z * weight_plus_eps.reshape(n, 1, h, w).expand(n, c, h, w)
        wighted_avg = average_pool(weighted_z)
        n, c, h, w = wighted_avg.shape
        return wighted_avg * inv_avg_weight.reshape(n, 1, h, w).expand(n,c,h,w)

    mean_x = weighted_avg_pool(frame1_image)
    mean_y = weighted_avg_pool(frame2_image)
    sigma_x = weighted_avg_pool(torch.pow(frame1_image, 2)) - torch.pow(mean_x, 2)
    sigma_y = weighted_avg_pool(torch.pow(frame2_image, 2)) - torch.pow(mean_y, 2)
    sigma_xy = weighted_avg_pool(frame1_image * frame2_image) - mean_x * mean_y

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


def average_pool(w) :
    return F.avg_pool2d(w, kernel_size=3, stride=1)


