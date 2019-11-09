"""
A depth map is the tensor representation [N, H, W] of depth per pixel.
This class encapsulates the transformation of depth with camera
motion (translation and rotation) and foreground motion.
"""
import torch

class DepthMap(object):
    """
    An object of this class will hold the depth values per pixel
    after it has been transformed by rotation and translation
    vectors corresponding to the rotation of the camera and
    translation of camera (background) and potentially moving
    objects (foreground).
    Obviously not all pixels that were previously in frame will be
    in frame after these transformation, so an additional map of xy
    in frame will be present to help only penalise points that
    should be in frame post transformation
    """

    def __init__(self, depth):
        """
        Initialises depth map and visibility post transformation
        
        Args:
            depth (torch.Tensor): FloatTensor of shape [N, H, W]
            xy_inframe (TYPE): ByteTensor of shape [N, H, W]
        """
        self.__depth = depth
        self.__grid = None

    @property
    def grid(self):
        """
        Represents the (x,y) position of every pixel 
        in the depth map
        
        Returns:
            TYPE: Description
        """
        if self.__grid is None :
            N, H, W = self.depth.shape
            device = self.depth.device
            x = torch.linspace(0, W-1, W)
            y = torch.linspace(0, H-1, H)
            y, x = torch.meshgrid(y, x)
            self.__grid = torch.ones((N, 3, H, W))
            self.__grid[:, 0, :, :] = x
            self.__grid[:, 1, :, :] = y
            self.__grid = self.__grid.to(device)

        return self.__grid

    @grid.setter
    def grid(self, grid) :
        self.__grid = grid.clone()

    @property
    def depth(self):
        """
        Depth of every pixel post transformation
        
        Returns:
            torch.FloatTensor: FloatTensor of shape [N, H, W]
        """
        return self.__depth.clone()

    @depth.setter
    def depth(self, depth) :
        self.__depth = depth

    def transform(self, translation, rotation, intrinsics) :
        """
        Remember this transformation is Rotation first then translation
        z' p' = K * R * K.inv() * z * p + K * T   
        Args:
            translation (TYPE): translation object of shape [N, 3]
            rotation (TYPE): Description
            intrinsics (TYPE): Description
        """
        N, H, W = self.depth.shape
        K = intrinsics.value # [N, 3, 3]
        R = rotation.value  # [N, 3, 3]
        T = translation.value # [N, 3, H, W]
        z_p_original = self.depth.reshape(N, 1, H, W) * self.grid # [N, 3, H, W]

        K_R_Kinv = torch.einsum('bij,bjk,bkl->bil', K, R, K.inverse())
        z_p_rotation = torch.einsum('bij,bjkl->bikl', K_R_Kinv, z_p_original)
        z_p_translation = torch.einsum('bij,bjkl->bikl', K, T)

        z_p_final = z_p_rotation + z_p_translation
        z_final = z_p_final[:, 2:, :, :].clone()
        grid_final = torch.div(z_p_final, z_final)

        """
        if translation is such that the point goes out of the frame or
        worse goes behind the camera we have to make sure we are capturing 
        that in grid motion computation, oor := out of range
        """
        x_in_r = (0 <= grid_final[:, 0, :, :]) * (W-1 >= grid_final[:, 0, :, :])
        y_in_r = (0 <= grid_final[:, 1, :, :]) * (H-1 >= grid_final[:, 1, :, :])
        z_in_r = (0 < z_final[:, 0, :, :])
        oor = ~(x_in_r * y_in_r * y_in_r)
        oor = oor.reshape(N, 1, H, W).expand(N, 3, H, W)
        grid_final[oor] = -1 # places where we are out of range, set the grid to -1 
        self.grid = grid_final
        self.depth = z_final[:, 0, :, :]


