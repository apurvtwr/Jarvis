"""
Utility functions to get TransformedMap from model outputs
"""


def from_motion_vector(depth, translation, rotation_angles, intrinsics):
    """
    Given a depth tensor [N, H, W], translation tensor [N, 3, H, W], rotation
    [N, 3] and intrinsics [N, 4], this function creates a TransformedDepthMap
    of new depth values and pixel locations that are still in camera frame.
    Args:
        depth (FloatTensor): [N, H, W] depth of every pixel before
        transforms
        translation (FloatTensor): [N, 3, H, W] translations along x, y, z
        rotation_angles (FloatTensor): [N, 3] euler angle of rotations
        intrinsics (FloatTensor): [N, 4] Camera Intrinsics
    """
    pass


