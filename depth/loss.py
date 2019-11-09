
class DepthMapTransformation(object) :

    def __init__(self, depth_map, intrinsics, rotation, translation) :
        self.depth_map = depth_map
        self.intrinsics = intrinsics
        self.rotation = rotation
        self.translation = translation


class ReconstructionLoss(object) :
    """
                |-----------------------|
    Frame A --> | K, R_AB, T_AB, Z_A    | --> Frame B
                |-----------------------|
    """
    def __init__(self, transformation_AB, transformation_BA) :
        self.transformation_AB = transformation_AB
        self.transformation_BA = transformation_BA