import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF 


def circle_mask(shape, sharpness = 40):
    """Return a circular mask of a given shape"""
    assert shape[1] == shape[2], "circle_mask received a bad shape: " + shape

    diameter = shape[1]
    x = np.linspace(-1, 1, diameter)
    y = np.linspace(-1, 1, diameter)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = (xx**2 + yy**2) ** sharpness

    mask = 1 - np.clip(z, -1, 1)
    mask = np.expand_dims(mask, axis=0)
    mask = np.broadcast_to(mask, shape)
    mask = torch.tensor(mask, dtype=torch.float)
    return mask


def affine_coeffs(scale, angle, x_shift, y_shift):
    """
    If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], 
    then it maps the output point (x, y) to a transformed input point 
    (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), 
    where k = c0 x + c1 y + 1. 
    The transforms are inverted compared to the transform mapping input points to output points.
    """
    rot = angle / 90. * (math.pi/2)

    # Standard rotation matrix
    # (use negative rot because nn.functional.affine_grid will do the inverse)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # Scale everything by inverse because nn.functional.affine_grid will do the inverse 
    cos /= scale
    sin /= scale
    x_shift /= -scale
    y_shift /= -scale

    coeffs = torch.stack((cos, -sin, x_shift, sin, cos, y_shift)).t().view(-1, 2, 3)
    return coeffs


def apply_patch(data, patch, transforms):
    bsz = data.size(0)
    height = data.size(2)
    
    mask = circle_mask(data.shape[1:]).cuda()
    mask_batch = mask.repeat(bsz, 1, 1, 1)
    patch_batch = patch.repeat(bsz, 1, 1, 1)


    coeffs = affine_coeffs(*transforms)
    grid = F.affine_grid(coeffs, data.size()).cuda()

    mask_t = F.grid_sample(mask_batch, grid)
    patch_t = F.grid_sample(patch_batch, grid)

    out = data * (1 - mask_t) + patch_t * mask_t
    return out


def tensor_to_pil(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), clip=False):
    image = image.clone()
    out = TF.normalize(image, mean=[0, 0, 0], std=[1/std[0], 1/std[1], 1/std[2]])
    out = TF.normalize(out, mean=[-mean[0], -mean[1], -mean[2]], std=[1, 1, 1])
    if clip:
        out = out * circle_mask(out.size()).cuda()
    out = TF.to_pil_image(out.detach().cpu())
    return out


def sample_transform(batch_size, min_scale, max_scale, max_angle):
    scale = torch.empty(batch_size).uniform_(min_scale, max_scale)
    angle = torch.empty(batch_size).uniform_(-max_angle, max_angle)
    x_shift = torch.empty(batch_size)
    y_shift = torch.empty(batch_size)

    for i in range(batch_size):
        s = scale[i]
        # Want to keep abs(shift) + scale <= 1 to keep all of patch inside image
        x_shift[i] = np.random.uniform(s - 1, 1 - s)
        y_shift[i] = np.random.uniform(s - 1, 1 - s)

    return scale, angle, x_shift, y_shift

