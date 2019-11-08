"""
Utility functions to get TransformedMap from model outputs
"""

import torch


class Intrinsics(object) :

    """Summary
    
    Attributes:
        fx (TYPE): Description
        fy (TYPE): Description
        s (TYPE): Description
        x0 (TYPE): Description
        y0 (TYPE): Description
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
        """
        Tensor of shape [N, 3, 3] where N is the batch size 
        
        Returns:
            TYPE: Description
        """
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

class Translation(object) :
    def __init__(self, translation) :
        if len(translation.shape) == 2 :
            N, C = translation.shape
            assert C == 3, "Translation should be 3 dimensional"
            self.__translation = translation.reshape(N, 3, 1, 1)
        else :
            N, C, H, W = translation.shape
            assert C == 3, "Translation must be 3 dimensional"
            self.__translation = translation.clone()

    @property
    def shape(self):
        return self.value.shape
    
    @property
    def value(self):
        return self.__translation
    

    def __add__(self, other) :
        N, _, _, _ = self.shape
        if other.shape == self.shape :
            return Translation(other.value + self.value)
        elif other.shape == torch.Size([N, 3, 1, 1]) :
            N, C, H, W = self.shape
            return Translation(other.value.expand(N, C, H, W) + self.value)
        else :
            N, C, H, W = other.shape
            return Translation(self.value.expand(N, C, H, W) + other.value)

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
        """
        [N, 3, 3] Rotation tensor
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



