import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from skimage import io


def collect_file_paths(
        root_dir: str,
        images_subdir: str = '01',
        masks_subdir: str = '01_GT/SEG'
        ) -> Tuple[List[str], List[str], List[str]]:
    """Collect paired image and mask paths for TNT volumes.

    The function assumes a file structure of the form::

        root_dir/
          01/                # image folder
            tXXX.tif
          01_GT/SEG/         # mask folder
            maskXXX.tif
    where ``XXX`` is a three-digit identifier (e.g. ``001``).

    Parameters
    ----------
    root_dir: Root directory containing image and mask subdirectories
    images_subdir: Name of the subdirectory with image volumes
    masks_subdir : Name of the subdirectory with segmentation masks

    Returns
    -------
    names:  List of volume identifiers (e.g. ``["001", "002", ...]``).
    image_paths: Full paths to image volumes.
    mask_paths: Full paths to corresponding mask volumes.
    """
    images_dir = os.path.join(root_dir, images_subdir)
    masks_dir = os.path.join(root_dir, masks_subdir)

    names, images, masks = [], [], []

    for file in sorted(os.listdir(masks_dir)):
        if file.endswith('.tif'):
            cur_id = file[4:7]
            names.append(cur_id)
            masks.append(os.path.join(masks_dir, file))
            images.append(os.path.join(images_dir, f't{cur_id}.tif'))

    return names, images, masks


def compute_global_minmax(images_paths: Sequence[str]) -> Tuple[float, float]:
    """Compute dataset-wide min and max values for intensity normalization.

    Parameters
    ----------
    images_paths:  Paths to the image volumes.

    Returns
    -------
    Global minimum intensity.
    Global maximum intensity.
    """
    gmin = float("inf")
    gmax = float("-inf")

    for p in images_paths:
        v = io.imread(p).astype(np.float32)
        gmin = min(gmin, float(v.min()))
        gmax = max(gmax, float(v.max()))
        
    return gmin, gmax


def load_and_preprocess_one(
        path: str,
        dtype=np.float32,
        normalize: bool = True,
        use_global_stats: bool = True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        ) -> np.ndarray:
    """Load a 3D volume and optionally normalize intensities to ``[0, 1]``.

    Parameters
    ----------
    path: Path to the volume on disk.
    dtype: Target data type (used via ``astype``).
    normalize: If ``True``, normalize the volume to ``[0, 1]``.
    use_global_stats: If ``True``, expects ``min_value`` and ``max_value`` to be provided
                      and uses them for normalization; otherwise per-volume min/max are used.
    min_value: Global or per-volume minimum intensity used for normalization if``normalize=True``.
    max_value: Global or per-volume maximum intensity used for normalization if``normalize=True``.

    Returns
    -------
    Loaded (and optionally normalized) volume of shape ``(Z, Y, X)``.
    """
    volume = io.imread(path).astype(dtype)
    if normalize:
        if use_global_stats and min_value is not None and max_value is not None:
            min_value, max_value = min_value, max_value
        else:
            min_value, max_value = volume.min(), volume.max()
        volume = (volume - min_value) / (max_value - min_value)
    return volume


def get_images_and_masks(root_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load all TNT image and mask volumes for a dataset root.
    This uses global min/max over all images for intensity normalization.

    Parameters
    ----------
    root_path: Root directory containing images and masks.

    Returns
    -------
    images: List of image volumes of shape ``(Z, Y, X)``.
    masks: List of mask volumes of shape ``(Z, Y, X)`` (integer labels).
    """
    _, images_paths, masks_paths = collect_file_paths(root_path)
    gmin, gmax = compute_global_minmax(images_paths)

    images, masks = [], []

    for image_path, mask_path in zip(images_paths, masks_paths):
        image = load_and_preprocess_one(image_path, use_global_stats=True, min_value=gmin, max_value=gmax)
        mask = load_and_preprocess_one(mask_path, dtype=np.int32, normalize=False)
        images.append(image)
        masks.append(mask)

    return images, masks
