import torch
import torch.nn.functional as F


def img_feature_sampler(feat, uv):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = F.grid_sample(feat, uv, align_corners=True, padding_mode='zeros')  # [B, C, N, 1]
    
    return samples[:, :, :, 0]  # [B, C, N]


def index_to_world(index, f=1, z_min=101/128, z_max=117/128):
    '''
    transform the index to the world coordinate

    param index: [B, 3, N], default value in [-1, 1]
    return: [B, 3, N]
    '''
    points = torch.zeros_like(index)
    points[:, 0, :] = index[:, 1, :]
    points[:, 1, :] = index[:, 2, :]
    points[:, 2, :] = index[:, 0, :]
    points[:, 2, :] = (points[:, 2, :] + 1) / 2 * (z_max - z_min) + z_min
    points[:, 2, :] = points[:, 2, :] * f
    # points[:, 2, :] *= -1
    points[:, 1, :] *= -1
    points[:, 0, :] *= -1
    
    return points


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    calibrations = calibrations.repeat(points.shape[0], 1, 1)
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz


def query(points, calibs, projection, im_feat, transforms=None):
    '''
    Given 3D points, query the network predictions for each point.
    Image features should be pre-computed before this call.
    store all intermediate features.
    query() function may behave differently during training/testing.
    :param points: [B, 3, N] sampling space coordinates of points
    :param calibs: [B, 3, 4] calibration matrices for each image
    :param transforms: Optional [B, 2, 3] image space coordinate transforms
    :return: [B, Res, N] predictions for each point
    '''
    b, c, d, h, w = points.shape
    points = torch.flatten(points, start_dim=2)
    world_points = index_to_world(points)
    xyz = projection(world_points, calibs, transforms)
    xy = xyz[:, :2, :]
    
    point_local_feat = img_feature_sampler(im_feat, xy)
    
    return point_local_feat.reshape(b, -1, d, h, w)
