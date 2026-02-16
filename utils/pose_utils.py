import torch
import torch.nn.functional as F


def rot6d2mat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    """
    device = x.device
    B = x.shape[0]

    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    rotMat = torch.stack((b1, b2, b3), dim=-1)  # [B,3,3]
    return rotMat


def mat2rot6d(rotMat):
    """Convert 3x3 rotation matrix to 6D rotation representation.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    """
    return rotMat[:, :, :2].reshape(rotMat.shape[0], 6)
