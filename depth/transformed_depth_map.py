"""
A depth map is the tensor representation [N, H, W] of depth per pixel.
This class encapsulates the transformation of depth with camera
motion (translation and rotation) and foreground motion.
"""
import torch

class Intrinsics(object) :

    """Summary
    """
    
    def __init__(self, intrinsics) :
        """Summary
        
        Args:
            intrinsics (FloatTensor): [N, 5] fx, fy, x0, y0, skew
        """
        self.fx = intrinsics[:, 0].reshape((-1, 1, 1))
        self.fy = intrinsics[:, 1].reshape((-1, 1, 1))
        self.x0 = intrinsics[:, 2].reshape((-1, 1, 1))
        self.y0 = intrinsics[:, 3].reshape((-1, 1, 1))
        self.s = intrinsics[:, 4].reshape((-1, 1, 1))
        self._value = None

    @property
    def value(self):
        if self._value is None :
            """
                --                   --
                |   fx      s   x0    |
             H  |   0       fy  y0    |
                |   0       0   1     |
                --                   --    
            """
            row_1 = torch.cat((self.fx, self.s, self.x0), dim=-1)
            row_2 = torch.cat((torch.zeros_like(self.fy), self.fy, self.y0), dim=-1)
            row_3 = torch.cat((torch.zeros_like(self.fx), 
                torch.zeros_like(self.fx), torch.ones_like(self.fx)), dim=-1)
            self._value = torch.cat((row_1, row_2, row_3), dim=-2)
        return self._value


class Rotation(object) :

    """Summary
    """
    
    def __init__(self, rotation_angles) :
        """
        Args:
            rotation_angles (FloatTensor): [N, 3]
        """
        self._theta_x = rotation_angles[:, 0]
        self._phi_y = rotation_angles[:, 1]
        self._psi_z = rotation_angles[:, 2]
        self.__rotation = None

    @property
    def value(self) :
        """Summary
        
        Returns:
            TYPE: Description
        """
        if self.__rotation is None :
            self.__rotation = torch.matmul(self.rotation_x, torch.matmul(self.rotation_y, self.rotation_z))

        return self.__rotation

    @property
    def rotation_x(self):
        """
            __                                 __
            | 1     0               0           |
        R_x | 0     cos(theta)      -sin(theta) |
            | 0     sin(theta)      cos(theta)  |
            --                                 --
        
        Returns:
            TYPE: Description
        """
        theta = self._theta_x.reshape((-1, 1, 1))
        row_1 = torch.cat((torch.ones_like(theta), torch.zeros_like(theta), torch.zeros_like(theta)), dim=-1)
        row_2 = torch.cat((torch.zeros_like(theta), torch.cos(theta), -torch.sin(theta)), dim=-1)
        row_3 = torch.cat((torch.zeros_like(theta), torch.sin(theta), torch.cos(theta)), dim=-1)
        R_x = torch.cat((row_1, row_2, row_3), dim=-2)
        return R_x

    @property
    def rotation_y(self):
        """
            __                                 __
            | cos(phi)     0        sin(phi)    |
        R_y | 0             1         0         |
            | -sin(phi)     0      cos(phi)     |
            --                                 --
        
        Returns:
            TYPE: Description
        """
        phi = self._phi_y.reshape((-1, 1, 1))
        row_1 = torch.cat((torch.cos(phi), torch.zeros_like(phi), torch.sin(phi)), dim=-1)
        row_2 = torch.cat((torch.zeros_like(phi), torch.ones_like(phi), torch.zeros_like(phi)), dim=-1)
        row_3 = torch.cat((-torch.sin(phi), torch.zeros_like(phi), torch.cos(phi)), dim=-1)
        R_y = torch.cat((row_1, row_2, row_3), dim=-2)
        return R_y

    @property
    def rotation_z(self):
        """
            __                                 __
            | cos(psi)     -sin(psi)       0    |
        R_z | sin(psi)      cos(psi)       0    |
            | 0                 0          1    |
            --                                 --
        
        Returns:
            TYPE: Description
        """
        psi = self._psi_z.reshape((-1, 1, 1))
        row_1 = torch.cat((torch.cos(psi), -torch.sin(psi), torch.zeros_like(psi)), dim=-1)
        row_2 = torch.cat((torch.sin(psi), torch.cos(psi), torch.zeros_like(psi)), dim=-1)
        row_3 = torch.cat((torch.zeros_like(psi), torch.zeros_like(psi), torch.ones_like(psi)), dim=-1)
        R_z = torch.cat((row_1, row_2, row_3), dim=-2)
        return R_z

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

    def __init__(self, depth, xy_inframe):
        """
        Initialises depth map and visibility post transformation
        
        Args:
            depth (torch.Tensor): FloatTensor of shape [N, H, W]
            xy_inframe (TYPE): ByteTensor of shape [N, H, W]
        """
        self.__depth = depth
        self.__xy_inframe = xy_inframe

        @property
        def depth(self):
            """
            Depth of every pixel post transformation
            
            Returns:
                torch.FloatTensor: FloatTensor of shape [N, H, W]
            """
            return self.__depth

        @property
        def xy_inframe(self):
            """
            Byte tensor representing if the pixel is still in frame
            post transformation.
            
            Returns:
                torch.ByteTensor: 1 if pixel still in frame, else 0
            """
            return self.__xy_inframe

        def transform(self, translation, rotation, intrinsics) :
            """Summary
            
            Args:
                translation (TYPE): Description
                rotation (TYPE): Description
                intrinsics (TYPE): Description
            """


