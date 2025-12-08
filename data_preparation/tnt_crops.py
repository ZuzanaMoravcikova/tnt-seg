# -------------------------------------------------------------
# Author: Zuzana Moravčíková
# Project: Deep-Learning-Based Segmentation of Tunneling Nanotubes in Volumetric Bioimage Data
# Year: 2025
# Contact: 514286@mail.muni.cz
# License: MIT
# -------------------------------------------------------------

from typing import List, Tuple, Sequence, Literal
import random
import numpy as np
from skimage.measure import label, regionprops


def extract_instance_crops(
        image: np.ndarray,
        mask: np.ndarray,
        padding: Tuple[int, int, int] = (2, 50, 50),
        depth: Literal["tight", "all"] = "all",
        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract crops around connected components in a 3D mask.

    Connected components are found in ``mask`` via ``skimage.measure.label``,
    and for each component a padded bounding box is used to crop both
    ``image`` and a binary mask.

    Parameters
    ----------
    image : Input image volume of shape ``(Z, Y, X)``.
    mask : Input mask volume of shape ``(Z, Y, X)``.
    padding : Padding (``pz``, ``py``, ``px``) added around the component bounding box.
    depth : If ``"all"``, the crop spans all z-slices. If ``"tight"``,
            only the z-slices containing the component (with padding) are used.

    Returns
    -------
    List of ``(image_crop, binary_mask_crop)`` pairs.
    """
    if image.shape != mask.shape:
        raise ValueError(
            "Image and mask must have the same shape, got "
            f"{image.shape} and {mask.shape}."
        )

    crops = []

    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)

    pz, py, px = padding
    sz, sy, sx = mask.shape

    for region in regions:
        minz, miny, minx, maxz, maxy, maxx = region.bbox

        if depth == "all":
            z0, z1 = 0, sz
        else:
            z0 = max(minz - pz, 0)
            z1 = min(maxz + pz, sz)

        y0 = max(miny - py, 0)
        y1 = min(maxy + py, sy)
        x0 = max(minx - px, 0)
        x1 = min(maxx + px, sx)

        image_crop = image[z0:z1, y0:y1, x0:x1]
        mask_crop = (labeled_mask[z0:z1, y0:y1, x0:x1] > 0).astype(np.uint8)

        crops.append((image_crop, mask_crop))

    return crops


def extract_all_crops(
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        padding: Tuple[int, int, int] = (2, 50, 50),
        depth: Literal["tight", "all"] = "all",
        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract instance-centered crops for all image-mask pairs.

    Parameters
    ----------
    images: Image volumes.
    masks: Corresponding mask volumes.
    padding: Padding passed to :func:`extract_instance_crops`.
    depth : Depth behavior passed to :func:`extract_instance_crops`.

    Returns
    -------
    Concatenated list of crops for all image-mask pairs.
    """
    all_crops = []
    for image, mask in zip(images, masks):
        crops = extract_instance_crops(image, mask, padding, depth)
        all_crops.extend(crops)
    return all_crops


def sample_random_patches(
        image: np.ndarray,
        mask: np.ndarray,
        num_patches: int = 5,
        size_range_z: Tuple[int, int] = (7, 8),
        size_range_xy: Tuple[int, int] = (64, 256),
        min_positive_voxels: int = 10,
        max_attempts: int = 50,
        rng: random.Random = random
        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Sample random 3D patches that contain enough positive voxels.

    Each sampled patch must contain at least ``min_positive_voxels`` non-zero
    entries in the mask. If a valid patch is not found after
    ``max_attempts`` trials, that patch is skipped.

    Parameters
    ----------
    image: Image volume of shape ``(Z, Y, X)``.
    mask: Corresponding mask volume of shape ``(Z, Y, X)``.
    num_patches: Number of patches to sample.
    size_range_z: Inclusive range ``(min_z, max_z)`` for the patch depth in slices.
    size_range_xy: Inclusive range ``(min_xy, max_xy)`` for the patch height and width.
    min_positive_voxels: Minimum number of positive voxels required in the mask patch.
    max_attempts: Maximum attempts per patch before giving up.
    rng: Random number generator used for sampling,.

    Returns
    -------
    List of ``(image_patch, mask_patch)`` pairs.
    """
    z, y, x = image.shape
    zi_min, zi_max = size_range_z
    xy_min, xy_max = size_range_xy
    patches = []

    for _ in range(num_patches):
        attempts = 0
        while attempts < max_attempts:

            dz = rng.randint(zi_min, min(zi_max, z))
            dy = rng.randint(xy_min, min(xy_max, y))
            dx = rng.randint(xy_min, min(xy_max, x))

            z0 = rng.randint(0, z - dz)
            y0 = rng.randint(0, y - dy)
            x0 = rng.randint(0, x - dx)
            z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx

            img_crop = image[z0:z1, y0:y1, x0:x1]
            msk_crop = (mask[z0:z1, y0:y1, x0:x1] > 0).astype(np.uint8)

            if msk_crop.sum() >= min_positive_voxels:
                patches.append((img_crop, msk_crop))
                break
            attempts += 1
    return patches


def sample_random_patches_all(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    n_per_image: int = 50,
    size_range_z: Tuple[int, int] = (7, 8),
    size_range_xy: Tuple[int, int] = (64, 256),
    min_positive_voxels: int = 10,
    max_attempts: int = 50,
    rng: random.Random = random
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Sample random patches for each image-mask pair in a dataset.

    Parameters
    ----------
    images: Image volumes.
    masks: Corresponding mask volumes.
    n_per_image: Number of patches to sample per image.
    size_range_z: Depth range passed to :func:`sample_random_patches`.
    size_range_xy: XY range passed to :func:`sample_random_patches`.
    min_positive_voxels: Minimum number of positive voxels in a patch.
    max_attempts: Maximum attempts per patch.
    rng: Random number generator.

    Returns
    -------
    list of tuple of numpy.ndarray
        Concatenated list of sampled patches.
    """
    out = []
    for image, mask in zip(images, masks):
        per_image = []

        if n_per_image > 0:
            per_image.extend(
                sample_random_patches(
                    image=image,
                    mask=mask,
                    num_patches=n_per_image,
                    size_range_z=size_range_z,
                    size_range_xy=size_range_xy,
                    min_positive_voxels=min_positive_voxels,
                    max_attempts=max_attempts,
                    rng=rng,
                )
            )
        rng.shuffle(per_image)
        out.extend(per_image)

    return out


def crop_zeros(
        image: np.ndarray,
        mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Crop image and mask to the bounding box of non-zero regions.

    The bounding box is computed only from ``image``; the same spatial crop
    is applied to both ``image`` and ``mask``.

    Parameters
    ----------
    image: Image volume of shape ``(Z, Y, X)``.
    mask: Mask volume of shape ``(Z, Y, X)``.

    Returns
    -------
    cropped_image: Cropped image volume.
    cropped_mask: Cropped mask volume.
    """
    if image.shape != mask.shape:
        raise ValueError(
            "Image and mask must have the same shape, got "
            f"{image.shape} and {mask.shape}."
        )
    if image.ndim != 3:
        raise ValueError(
            f"Image must be 3D (Z, Y, X), but has shape {image.shape}."
        )

    non_zero = np.argwhere(image != 0)
    if non_zero.size == 0:
        return image, mask

    z_min, y_min, x_min = non_zero.min(axis=0)
    z_max, y_max, x_max = non_zero.max(axis=0) + 1

    cropped_image = image[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped_mask = mask[z_min:z_max, y_min:y_max, x_min:x_max]

    return cropped_image, cropped_mask


def crops_zeros_all(
        images: List[np.ndarray],
        masks: List[np.ndarray]
        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Apply :func:`crop_zeros` to all image-mask pairs.

    Parameters
    ----------
    images: Image volumes.
    masks: Corresponding mask volumes.

    Returns
    -------
    List of cropped image and mask pairs.
    """
    all_crops = []
    for image, mask in zip(images, masks):
        crop = crop_zeros(image, mask)
        all_crops.append(crop)
    return all_crops
