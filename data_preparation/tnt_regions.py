from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def quadrant_rect_coords(idx: int) -> Dict[str, int]:
    """Return rectangle coordinates for a fixed quadrant of a 512x512 image.
    Quadrants are defined as:
    - ``0``: top-left
    - ``1``: top-right
    - ``2``: bottom-left
    - ``3``: bottom-right
    The returned dictionary uses inclusive coordinates.

    Parameters
    ----------
    idx: Quadrant index in ``{0, 1, 2, 3}``.

    Returns
    -------
    dict:  Dictionary with keys ``"left"``, ``"right"``, ``"top"``, ``"bottom"``.
    """
    if not 0 <= idx <= 3:
        raise ValueError(f"Quadrant index must be in [0, 3], got {idx}.")

    if idx == 0:
        return{'left': 0, 'right': 255, 'top': 0, 'bottom': 255}
    elif idx == 1:
        return {'left': 256, 'right': 511, 'top': 0, 'bottom': 256}
    elif idx == 2:
        return {'left': 0, 'right': 255, 'top': 256, 'bottom': 511}
    elif idx == 3:
        return {'left': 256, 'right': 511, 'top': 256, 'bottom': 511}


def suggest_validation_coords(
        x_size: int,
        y_size: int,
        masked_part: int,
        img_width: int = 512,
        img_height: int = 512
        ) -> Dict[str, int]:
    """Suggest a validation rectangle that does not overlap a given quadrant.

    This scans the image canvas in a deterministic top-left to bottom-right
    fashion and returns the first rectangle that:
    - fits into the image bounds, and
    - does not overlap with the forbidden quadrant (given by ``masked_part``).

    Parameters
    ----------
    x_size: i Width of the desired rectangle.
    y_size: Height of the desired rectangle.
    masked_part: Quadrant index in ``{0, 1, 2, 3}`` that must not overlap with the suggested rectangle.
    img_width: Image width.
    img_height: Image height.

    Returns
    -------
    Dictionary with keys ``"left"``, ``"right"``, ``"top"``, ``"bottom"``. Coordinates are inclusive.
    """
    forbidden = quadrant_rect_coords(masked_part)

    if x_size > img_width or y_size > img_height:
        raise ValueError(
            f"Requested rect {x_size}x{y_size} does not fit into "
            f"image size {img_width}x{img_height}."
        )

    def overlaps(a: Dict[str, int], b: Dict[str, int]) -> bool:
        """Return True if rectangles a and b overlap (inclusive coords)."""
        return not (
            a["right"] < b["left"] or
            a["left"] > b["right"] or
            a["bottom"] < b["top"] or
            a["top"] > b["bottom"]
        )

    for top in range(0, img_height - y_size + 1):
        for left in range(0, img_width - x_size + 1):
            candidate = {
                "left": left,
                "right": left + x_size - 1,
                "top": top,
                "bottom": top + y_size - 1,
            }
            if not overlaps(candidate, forbidden):
                return candidate


def apply_rectangle_mask(
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        top: int = 0,
        bottom: int = 229,
        left: int = 0,
        right: int = 229,
        split: str = 'train'
        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Apply a rectangular mask consistently to images and masks.
    The behavior depends on the ``split``:
    - ``split == "train"``: the rectangle region is zeroed out (removed).
    - ``split == "test"``: only the rectangle region is kept (everything
      outside is zeroed out).

    Parameters
    ----------
    images: Input image volumes of shape ``(Z, Y, X)``.
    masks: Corresponding mask volumes of shape ``(Z, Y, X)``.
    top: Top coordinate (inclusive) of the rectangle.
    bottom: Bottom coordinate (inclusive) of the rectangle.
    left: Left coordinate (inclusive) of the rectangle.
    right: Right coordinate (inclusive) of the rectangle.
    split: Determines how masking is applied.

    Returns
    -------
    out_images: Masked image volumes.
    out_masks: Masked mask volumes.
    """
    out_images, out_masks = [], []

    for original_image, original_mask in zip(images, masks):
        assert original_image.shape == original_mask.shape

        if split == 'train':
            image = original_image.copy()
            mask = original_mask.copy()
            image[:, top:bottom + 1, left:right + 1] = 0
            mask[:, top:bottom + 1, left:right + 1] = 0

        elif split == 'test':
            image = np.zeros_like(original_image)
            mask = np.zeros_like(original_mask)
            image[:, top:bottom + 1, left:right + 1] = original_image[:, top:bottom + 1, left:right + 1]
            mask[:, top:bottom + 1, left:right + 1] = original_mask[:, top:bottom + 1, left:right + 1]

        else:
            raise ValueError("Split must be either 'train' or 'test'.")

        out_images.append(image)
        out_masks.append(mask)
    return out_images, out_masks


def expand_rectangle(
        rect: Dict[str, int],
        margin_xy: int | Tuple[int, int],
        max_y: int,
        max_x: int
        ) -> Dict[str, int]:
    """Expand a rectangle by a given margin and clamp to image bounds.

    Coordinates are inclusive and assumed in ``(Y, X)`` order.

    Parameters
    ----------
    rect: Original rectangle with keys ``"top"``, ``"bottom"``, ``"left"``,``"right"``.
    margin_xy: If an integer, the same margin is used for height and width.
               If a tuple, interpreted as ``(margin_y, margin_x)``.
    max_y: Image height (number of rows).
    max_x: Image width (number of columns).

    Returns
    -------
    Expanded and clamped rectangle with the same keys as ``rect``.
    """
    if isinstance(margin_xy, int):
        my, mx = margin_xy, margin_xy
    else:
        my, mx = margin_xy

    top = max(0, rect['top'] - my)
    bottom = min(max_y - 1, rect['bottom'] + my)
    left = max(0, rect['left'] - mx)
    right = min(max_x - 1, rect['right'] + mx)

    return {'top': top, 'bottom': bottom, 'left': left, 'right': right}


def make_roi_mask_like(example_zyx: np.ndarray, rect: Dict[str, int]) -> np.ndarray:
    """Create an ROI mask matching the shape of an example volume.

    Parameters
    ----------
    example_zyx : Example volume of shape ``(Z, Y, X)`` which defines the output shape.
    rect : Rectangle with keys ``"top"``, ``"bottom"``, ``"left"``, ``"right"``, using inclusive coordinates in YX.

    Returns
    -------
    ROI mask of shape ``(Z, Y, X)`` with value ``1`` inside ``rect`` and ``0`` elsewhere.
    """
    z, y, x = example_zyx.shape
    roi = np.zeros((z, y, x), dtype=np.uint8)
    roi[:, rect['top']:rect['bottom']+1, rect['left']:rect['right']+1] = 1
    return roi


def roi_bbox_zyx(roi01_zyx: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    """Compute the bounding box of a 3D ROI mask in ZYX order.

    The ROI is assumed to be binary (values in {0, 1}). The function
    returns end-exclusive indices. If the ROI is empty, the bounding box
    of the full volume is returned.

    Parameters
    ----------
    roi01_zyx: 3D ROI mask of shape ``[Z, Y, X]`` with values in ``{0, 1}``.

    Returns
    -------
    z0_z1_y0_y1_x0_x1: Tuple ``(z0, z1, y0, y1, x0, x1)`` with end-exclusive indices
                       defining the bounding box of the ROI.
    """
    assert roi01_zyx.ndim == 3, f"Expected [Z,Y,X], got {roi01_zyx.shape}"

    Z, Y, X = roi01_zyx.shape
    nz = (roi01_zyx > 0).nonzero(as_tuple=False)

    if nz.numel() == 0:
        return 0, Z, 0, Y, 0, X

    z0, z1 = int(nz[:, 0].min()), int(nz[:, 0].max()) + 1
    y0, y1 = int(nz[:, 1].min()), int(nz[:, 1].max()) + 1
    x0, x1 = int(nz[:, 2].min()), int(nz[:, 2].max()) + 1

    return z0, z1, y0, y1, x0, x1


def ensure_roi(y_b1zyx: torch.Tensor, roi_b1zyx: torch.Tensor | None) -> torch.Tensor:
    """Ensure a valid binary ROI mask for evaluation.

   Parameters
   ----------
   y_b1zyx: Prediction tensor used to infer the full-volume shape when no ROI is provided.
   roi_b1zyx: Optional ROI tensor. If given, it is binarized.

   Returns
   -------
   roi_b1zyx: Binary ROI tensor with values in ``{0, 1}``, dtype ``uint8``, shape ``[B, 1, Z, Y, X]``.
   """
    if roi_b1zyx is None:
        return torch.ones_like(y_b1zyx, dtype=torch.uint8)
    return (roi_b1zyx > 0).to(torch.uint8)


def crop_roi_batch(
        tensor_b1zyx: torch.Tensor,
        roi_b1zyx: torch.Tensor,
        detach: bool = True,
        strict: bool = True) -> torch.Tensor:
    """Crop a batched tensor to a single ROI bounding box shared across the batch.

        The function assumes that all samples in the batch share the same ROI
        bounding box (or that a single union bbox is acceptable). The crop is
        computed from the ROI mask and applied to the input tensor.

        Parameters
        ----------
        tensor_b1zyx: Input batched tensor of shape ``[B, 1, Z, Y, X]`` to be cropped.
        roi_b1zyx:  Batched ROI tensor of shape ``[B, 1, Z, Y, X]``. Non-zero values
                    indicate the ROI region for each sample.
        detach:  If ``True``, the returned tensor is detached from the computation graph.
                 If ``False``, the tensor remains connected.
        strict: If ``True``, all items in the batch must have the same ROI bounding box.
                If ``False``, a union bounding box across the batch is used.

        Returns
        -------
        out: Cropped batched tensor of shape ``[B, 1, Z', Y', X']`` with the same dtype and device as the input.
        """
    assert tensor_b1zyx.ndim == 5 and tensor_b1zyx.shape[1] == 1, \
        f"Expected [B,1,Z,Y,X], got {tensor_b1zyx.shape}"
    assert roi_b1zyx.shape == tensor_b1zyx.shape, \
        f"ROI shape {roi_b1zyx.shape} must match tensor shape {tensor_b1zyx.shape}"

    B = tensor_b1zyx.shape[0]

    z0, z1, y0, y1, x0, x1 = roi_bbox_zyx(roi_b1zyx[0, 0])

    if strict:
        for b in range(1, B):
            bz0, bz1, by0, by1, bx0, bx1 = roi_bbox_zyx(roi_b1zyx[b, 0])
            assert (bz0, bz1, by0, by1, bx0, bx1) == (z0, z1, y0, y1, x0, x1), \
                f"ROI bbox differs in batch (item 0 {z0,z1,y0,y1,x0,x1} vs item {b} {bz0,bz1,by0,by1,bx0,bx1})"
    else:
        for b in range(1, B):
            bz0, bz1, by0, by1, bx0, bx1 = roi_bbox_zyx(roi_b1zyx[b, 0])
            z0, z1 = min(z0, bz0),  max(z1, bz1)
            y0, y1 = min(y0, by0), max(y1, by1)
            x0, x1 = min(x0, bx0), max(x1, bx1)

    out = tensor_b1zyx[:, :, z0:z1, y0:y1, x0:x1]
    return out.detach() if detach else out
