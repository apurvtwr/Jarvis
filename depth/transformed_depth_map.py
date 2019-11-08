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

    def __init__(self, depth, xy_inframe):
        """
        Initialises depth map and visibility post transformation
        
        Args:
            depth (torch.Tensor): FloatTensor of shape [N, H, W]
            xy_inframe (TYPE): ByteTensor of shape [N, H, W]
        """
        self.__depth = depth
        self.__xy_inframe = xy_inframe
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
            self.__grid = grid

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
            """
            z' p' = K * R * K.inv() * z * p + K * T   
            Args:
                translation (TYPE): Description
                rotation (TYPE): Description
                intrinsics (TYPE): Description
            """
            K = intrinsics.value
            R = rotation.value
            T = translation.value
            torch.einsum('bij,bjk,bkl', K, )


