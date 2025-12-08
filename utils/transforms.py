# -------------------------------------------------------------
# Author: Zuzana Moravčíková
# Project: Deep-Learning-Based Segmentation of Tunneling Nanotubes in Volumetric Bioimage Data
# Year: 2025
# Contact: 514286@mail.muni.cz
# License: MIT
# -------------------------------------------------------------

from typing import Optional, Sequence, Union, Tuple

import bio_volumentations as bv
import numpy as np
import torch
import torch.nn.functional as F


def pad_to_multiple(volume: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """Pad a volume so that height and width are divisible by a given value.

    Padding is centered and uses zeros.

    Parameters
    ----------
    volume: Tensor of shape ``(1, Z, H, W)``.
    multiple: Target divisibility requirement for H and W.

    Returns
    -------
    padded: Tensor with H and W divisible by ``multiple``.
    """
    _, z, h, w = volume.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = F.pad(volume, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return padded


class CenterCropToDivisible(bv.DualTransform):
    """Center-crop so that height/width become divisible by ``divisor``.

    Works on arrays of shape ``(Z, Y, X)`` or ``(C, Z, Y, X)``.
    Ensures non-empty output even when dimensions are smaller than ``divisor``.

    Parameters
    ----------
    divisor: Target divisor for H and W.
    always_apply: Whether to apply the transform unconditionally.
    p: Probability of applying the transform.
    """

    def __init__(self, divisor: int = 16, always_apply: bool = True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.divisor = divisor

    def apply(self, img, **params):
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        elif img.ndim != 4:
            raise ValueError(f"Expected 3D or 4D image, got {img.ndim}D")
        _, z, h, w = img.shape
        new_h = (h // self.divisor) * self.divisor
        new_w = (w // self.divisor) * self.divisor
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return img[:, :, top:top + new_h, left:left + new_w]


class CenterCropPixels(bv.DualTransform):
    """Center-crop a fixed number of pixels along height and width.

    Works for arrays with the last two axes as ``(Y, X)`` and supports 2D–5D
    inputs.

    Parameters
    ----------
    h_crop: Total number of pixels to remove from height.
    w_crop: Total number of pixels to remove from width.
    always_apply: Whether to apply unconditionally.
    p: Probability of applying the transform.
    """

    def __init__(self, h_crop: int, w_crop: int,
                 always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)

        if h_crop < 0 or w_crop < 0:
            raise ValueError("h_crop and w_crop must be non-negative.")
        self.h_crop = h_crop
        self.w_crop = w_crop

    def apply(self, arr: np.ndarray, **params):
        h_total, w_total = self.h_crop, self.w_crop
        top = h_total // 2
        bottom = h_total - top
        left = w_total // 2
        right = w_total - left

        if arr.shape[-2] <= h_total or arr.shape[-1] <= w_total:
            raise ValueError(
                f"Requested crop ({h_total}, {w_total}) "
                f"is larger than the input size {arr.shape[-2:]}"
            )

        slices = [slice(None)] * arr.ndim
        slices[-2] = slice(top, arr.shape[-2] - bottom)
        slices[-1] = slice(left, arr.shape[-1] - right)
        return arr[tuple(slices)]


def to_torchio_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a Numpy volume to a TorchIO 4D tensor.

    Parameters
    ----------
    array: Input array with shape ``(Z, Y, X, C)`` or similar.

    Returns
    -------
    tensor: Torch tensor shaped ``(1, C, X, Y, Z)`` suited for TorchIO.
    """
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor.permute(0, 3, 2, 1)


def bv_train_transform() -> bv.Compose:
    """Return Bio-Volumentations augmentation pipeline for training.

    Includes mild intensity and geometric transforms with conservative
    magnitudes to preserve anatomical fidelity.

    Operations
    ----------
    - Gamma adjustment
    - Gaussian blur
    - Random flips
    - 90° rotation
    - Small affine transform
    """
    aug = bv.Compose([
        bv.RandomGamma(gamma_limit=(0.8, 1.2), p=0.6),
        bv.GaussianBlur(sigma=0.5, p=0.3),
        bv.RandomFlip(axes_to_choose=[1, 2, 3], p=0.5),
        bv.RandomRotate90(axes=[1], p=0.5),
        bv.RandomAffineTransform(
            angle_limit=(-30, 30, 0, 0, 0, 0),
            border_mode='constant', ival=0.0, mval=0.0, p=0.6),
    ])
    return aug


def bv_test_transform() -> bv.Compose:
    """Return an identity Bio-Volumentations transform for testing."""
    return bv.Compose([])


def center_crop_or_pad_after_transform(
        arr: Union["np.ndarray", "torch.Tensor"],
        out_size: Tuple[int, int, int],
        spatial_axes: Optional[Sequence[int]] = None,
        pad_value: Union[int, float, bool] = 0):
    """Center-crop or pad a 3D volume to an exact output size.

    Works with both NumPy arrays and Torch tensors. Extra non-spatial
    dimensions (e.g. batch or channels) are preserved.

    Parameters
    ----------
    arr: Input array or tensor.
    out_size: Desired ``(Z, Y, X)`` spatial size.
    spatial_axes: Which three axes correspond to spatial dimensions. If None, the last three axes are used for ndim ≥ 4.
    pad_value: Value used for padding.

    Returns
    -------
    out: Array/tensor of the same type as input with spatial shape exactly matching ``out_size``.
    """
    ndim = arr.ndim

    is_numpy = (np is not None) and isinstance(arr, np.ndarray)
    is_torch = (torch is not None) and isinstance(arr, torch.Tensor)
    if not (is_numpy or is_torch):
        raise TypeError("arr must be either a NumPy ndarray or a torch Tensor.")

    if spatial_axes is None:
        spatial_axes = (0, 1, 2) if ndim == 3 else (ndim - 3, ndim - 2, ndim - 1)
    if len(spatial_axes) != 3:
        raise ValueError("spatial_axes must contain exactly 3 axis indices.")
    spatial_axes = tuple(a if a >= 0 else ndim + a for a in spatial_axes)
    if len(set(spatial_axes)) != 3:
        raise ValueError("spatial_axes must refer to three distinct axes.")

    if is_numpy:
        arr_moved = np.moveaxis(arr, spatial_axes, (0, 1, 2))
    else:
        arr_moved = torch.movedim(arr, spatial_axes, (0, 1, 2))

    spatial_in = arr_moved.shape[:3]
    rest_shape = arr_moved.shape[3:]
    z_in, y_in, x_in = spatial_in
    z_out, y_out, x_out = out_size

    def centered_crop_range(in_sz: int, out_sz: int):
        if in_sz > out_sz:
            start = (in_sz - out_sz) // 2
            end = start + out_sz
        else:
            start, end = 0, in_sz
        return start, end

    z0, z1 = centered_crop_range(z_in, z_out)
    y0, y1 = centered_crop_range(y_in, y_out)
    x0, x1 = centered_crop_range(x_in, x_out)

    if is_numpy:
        arr_cropped = arr_moved[z0:z1, y0:y1, x0:x1, ...]
    else:
        arr_cropped = arr_moved[z0:z1, y0:y1, x0:x1, ...]

    z_cur, y_cur, x_cur = arr_cropped.shape[:3]

    def pad_before_after(cur: int, out_: int):
        if cur >= out_:
            return 0, 0
        total = out_ - cur
        before = total // 2
        after = total - before
        return before, after

    z_b, z_a = pad_before_after(z_cur, z_out)
    y_b, y_a = pad_before_after(y_cur, y_out)
    x_b, x_a = pad_before_after(x_cur, x_out)

    out_spatial_shape = (z_out, y_out, x_out)
    out_shape_full = out_spatial_shape + rest_shape

    if is_numpy:
        out_arr = np.full(out_shape_full, pad_value, dtype=arr.dtype)
        out_arr[z_b:z_b + z_cur, y_b:y_b + y_cur, x_b:x_b + x_cur, ...] = arr_cropped
        out_back = np.moveaxis(out_arr, (0, 1, 2), spatial_axes)
    else:
        if arr.dtype == torch.bool:
            fill_val = bool(pad_value)
        elif arr.dtype.is_floating_point:
            fill_val = float(pad_value)
        else:
            fill_val = int(pad_value)

        out_arr = torch.full(out_shape_full, fill_val, dtype=arr.dtype, device=arr.device)
        out_arr[z_b:z_b + z_cur, y_b:y_b + y_cur, x_b:x_b + x_cur, ...] = arr_cropped
        out_back = torch.movedim(out_arr, (0, 1, 2), spatial_axes)

    return out_back
