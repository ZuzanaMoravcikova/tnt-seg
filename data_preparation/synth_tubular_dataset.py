# -------------------------------------------------------------
# Author: Zuzana Moravčíková
# Project: Deep-Learning-Based Segmentation of Tunneling Nanotubes in Volumetric Bioimage Data
# Year: 2025
# Contact: 514286@mail.muni.cz
# License: MIT
# -------------------------------------------------------------

import os
from functools import partial
from typing import Sequence, Tuple, Optional, Union, List

import numpy as np
import pytorch_lightning as pl
import torch
import torchio as tio
from scipy.ndimage import binary_dilation, gaussian_filter
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------------------------------
# Tube synthesis
# -------------------------------------------------------------------------
def make_tube_template(
    image_shape: Tuple[int, int, int] = (16, 128, 128),
    radius_range: Tuple[float, float] = (1.0, 3.0),
    length_range: Tuple[int, int] = (16, 32),
    axes: Sequence[int] = (1, 2),
    structure_intensity: Tuple[float, float] = (0.7, 1.0),
    per_voxel: bool = True,
    rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
     Create a centered straight tube and return ``(template, mask)``.

     The tube is placed in the middle of the volume and oriented along
     one of the chosen ``axes``. A binary mask is generated for the tube
     and the intensity image is filled with background zeros and random
     values inside the tube.

     Parameters
     ----------
     image_shape: Shape of the 3D image in ``(Z, Y, X)``.
     radius_range: Minimum and maximum tube radius (in voxels).
     length_range: Minimum and maximum tube length (in voxels).
     axes: Sequence of allowed tube orientations.
     structure_intensity: Range ``(low, high)`` from which tube intensities are sampled.
     rng: Optional seed or :class:`numpy.random.Generator` instance.

     Returns
     -------
     template: image array of shape ``(Z, Y, X)`` with background
         equal to zero and non-zero values inside the tube.
     mask: array of shape ``(Z, Y, X)`` where voxels belonging
         to the tube are ``1`` and background is ``0``.
     """
    _rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    assert len(image_shape) == 3, "image_shape must be (Z, Y, X)"
    axes = list(axes)
    assert len(axes) > 0 and all(a in (0, 1, 2) for a in axes), "axes must be a subset of {0,1,2}"
    zdim, ydim, xdim = image_shape
    dims = (zdim, ydim, xdim)

    r_min, r_max = radius_range
    L_min, L_max = length_range
    assert 0 <= r_min <= r_max, "Invalid radius_range"
    assert 1 <= L_min <= L_max, "Invalid length_range"

    axis = int(_rng.choice(axes))
    radius = float(_rng.uniform(r_min, r_max))
    length = int(_rng.integers(L_min, L_max + 1))

    cz, cy, cx = zdim // 2, ydim // 2, xdim // 2
    centers = (cz, cy, cx)

    half_allow = min(centers[axis], dims[axis] - 1 - centers[axis])
    halfL = min(length // 2, half_allow)

    ortho_axes = [a for a in (0, 1, 2) if a != axis]
    r_allow = min(
        min(centers[ortho_axes[0]], dims[ortho_axes[0]] - 1 - centers[ortho_axes[0]]),
        min(centers[ortho_axes[1]], dims[ortho_axes[1]] - 1 - centers[ortho_axes[1]]),
    )
    radius = min(radius, float(r_allow))

    zz, yy, xx = np.ogrid[:zdim, :ydim, :xdim]

    if axis == 0:
        long_mask = np.abs(zz - cz) <= halfL
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    elif axis == 1:
        long_mask = np.abs(yy - cy) <= halfL
        r2 = (zz - cz) ** 2 + (xx - cx) ** 2
    else:
        long_mask = np.abs(xx - cx) <= halfL
        r2 = (zz - cz) ** 2 + (yy - cy) ** 2

    rad_mask = r2 <= (radius ** 2)
    mask = (long_mask & rad_mask).astype(np.uint8)

    low, high = structure_intensity
    assert low <= high, "structure_intensity must be (low, high) with low <= high"

    template = np.zeros(image_shape, dtype=np.float32)
    if per_voxel:
        vals = _rng.uniform(low, high, size=image_shape).astype(np.float32)
        template[mask.astype(bool)] = vals[mask.astype(bool)]
    else:
        val = float(_rng.uniform(low, high))
        template[mask.astype(bool)] = val

    return template, mask


# -------------------------------------------------------------------------
# Intensity transforms (PSF, noise, depth attenuation)
# -------------------------------------------------------------------------
def anisotropic_psf(
        image,
        sigma_xy=(0.6, 1.1),
        z_factor=(1, 1.2),
        max_sigma_frac=0.25,
        rng=None
        ) -> torch.Tensor:
    """
    Apply an anisotropic Gaussian blur to simulate an optical PSF.

    Parameters
    ----------
    image: Tensor of shape ``[C, Z, Y, X]`` with values in ``[0, 1]``.
    sigma_xy: Range for the in-plane Gaussian sigma (in voxels).
    z_factor: Multiplicative factor applied to ``sigma_xy`` to obtain
        the axial sigma ``sigma_z``.
    max_sigma_frac: Upper bound on ``sigma_z`` as a fraction of the depth ``Z``,
        to avoid blurring across the whole stack.
    rng: Optional seed for the internal random number generator.

    Returns
    -------
    Blurred tensor of the same shape and dtype as the input.
    """
    rng = np.random.default_rng(rng)
    sxy = float(rng.uniform(*sigma_xy))
    z_mult = float(rng.uniform(*z_factor))

    C, Z, Y, X = image.shape
    sz = sxy * z_mult
    sz = min(sz, max(0.5, max_sigma_frac * Z))

    device, dtype = image.device, image.dtype
    out = np.empty((C, Z, Y, X), dtype=np.float32)
    for c in range(C):
        arr = image[c].detach().cpu().numpy().astype(np.float32)
        arr = gaussian_filter(arr, sigma=(sz, sxy, sxy), mode="reflect")
        out[c] = arr
    return torch.from_numpy(out).to(device=device, dtype=dtype)


def depth_attenuation(
        image: torch.Tensor,
        alpha=(0.005, 0.03),
        rng=None
        ) -> torch.Tensor:
    """
    Simulate depth-dependent attenuation (e.g. scattering or absorption).

    Intensity decays exponentially with depth::

        I'(z) = I(z) * exp(-a * z)

    Parameters
    ----------
    image: Tensor of shape ``[C, Z, Y, X]``.
    alpha: Range for the attenuation coefficient ``a``.
    rng: Optional seed for the internal random number generator.

    Returns
    -------
    Attenuated image of the same shape and dtype.
    """
    rng = np.random.default_rng(rng)
    a = float(rng.uniform(*alpha))

    C, Z, Y, X = image.shape
    device, dtype = image.device, image.dtype
    z = torch.arange(Z, device=device, dtype=dtype)
    w = torch.exp(-a * z).view(1, Z, 1, 1)
    return image * w


def poisson_gaussian_hetero(
        image: torch.Tensor,
        poisson_scale: float = 255.0,
        sigma0_range=(0.01, 0.02),
        sigma1_range=(0.0,   0.04),
        rng=None
        ) -> torch.Tensor:
    """
    Apply Poisson (shot) noise and heteroscedastic Gaussian (read) noise.

    Parameters
    ----------
    image: Tensor of shape ``[C, Z, Y, X]`` with values in ``[0, 1]``.
    poisson_scale: Scaling factor
    sigma0_range: Range of the signal-independent noise floor.
    sigma1_range: Range of the signal-dependent term in the per-voxel sigma
    rng: Optional seed for the internal random number generator.

    Returns
    -------
    Noisy image of the same shape and dtype.
    """

    rng = np.random.default_rng(rng)
    device, dtype = image.device, image.dtype
    arr = image.detach().cpu().numpy().astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)

    scale = float(poisson_scale)
    arr = rng.poisson(arr * scale) / scale

    s0 = float(rng.uniform(*sigma0_range))
    s1 = float(rng.uniform(*sigma1_range))
    sigma_map = s0 + s1 * np.sqrt(np.clip(arr, 0.0, 1.0))

    arr = arr + rng.normal(0.0, 1.0, size=arr.shape).astype(np.float32) * sigma_map
    arr = np.clip(arr, 0.0, 1.0)

    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def soft_black_clamp(
        image: torch.Tensor,
        threshold: float = 0.01,
        knee: float = 0.02
    ) -> torch.Tensor:
    """
    Soft-threshold small values toward 0 to quiet background without hard edges.
    """
    t, k = threshold, knee
    x = image
    y = torch.zeros_like(x)

    mid = (x >= t) & (x < t + k)
    xm = (x[mid] - t) / k
    y[mid] = (3*xm**2 - 2*xm**3) * (t + k)
    y[x >= (t + k)] = x[x >= (t + k)]
    return torch.clamp(y, 0, 1)


# -------------------------------------------------------------------------
# TorchIO augmentation pipeline
# -------------------------------------------------------------------------
def get_augmentations(
    pad_before_affine: Optional[tuple[int, int, int]] = None,
    crop_target_shape: Optional[tuple[int, int, int]] = None,

    enable_elastic: bool = True,
    elastic_num_control_points: int = 9,
    elastic_max_displacement: Union[tuple[int, int, int], int, float] = (0, 5, 5),
    elastic_image_interpolation: str = 'linear',
    elastic_p: float = 1.0,

    enable_affine: bool = True,
    affine_scales: Union[tuple[float, float], tuple[float, float, float]] = (1.0, 1.0),
    affine_degrees: tuple[float, ...] = (-45, 45, -30, 30, -30, 30),
    affine_translation: Union[int, tuple[int, int, int]] = (0, 10, 10),
    affine_isotropic: bool = False,
    affine_center: Optional[str] = 'image',
    affine_image_interpolation: str = 'linear',
    affine_p: float = 1.0,

    enable_anisotropy: bool = False,
    anisotropy_axes: tuple[int, ...] = (0,),
    anisotropy_downsampling: Union[tuple[float, float], float] = (2, 4),
    anisotropy_scalars_only: bool = True,
    anisotropy_image_interpolation: str = 'linear',
    anisotropy_p: float = 1.0,

    enable_psf: bool = True,
    psf_sigma_xy: tuple[float, float] = (0.6, 1.1),
    psf_z_factor: tuple[float, float] = (1.0, 1.2),
    psf_max_sigma_frac: float = 0.25,
    psf_rng: Optional[int] = None,

    enable_poisson_hetero: bool = True,
    poisson_scale: float = 255.0,
    poisson_sigma0_range: tuple[float, float] = (0.02, 0.04),
    poisson_sigma1_range: tuple[float, float] = (0.0, 0.06),
    poisson_rng: Optional[int] = None,

    enable_soft_black_clamp: bool = True,
    clamp_threshold: float = 0.01,
    clamp_knee: float = 0.02,

    enable_bias_field: bool = True,
    bias_coefficients: Union[float, tuple[float, float]] = 0.3,
    bias_order: int = 3,
    bias_p: float = 1.0,

    enable_gamma: bool = True,
    log_gamma: tuple[float, float] = (-0.1, 0.25),
    gamma_p: float = 1.0,

    enable_blur: bool = True,
    blur_std: Union[float, tuple[float, float]] = (0.3, 0.6),
    blur_p: float = 0.25,

    enable_depth_attenuation: bool = True,
    depth_alpha: tuple[float, float] = (0.005, 0.03),
    depth_rng: Optional[int] = None,
    ) -> tio.Transform:
    """
    Build a TorchIO :class:`~torchio.transforms.Compose` object for
    geometric and intensity augmentations.

    The default configuration is designed to create plausible 3D
    microscopy-like volumes from the synthetic tube templates.
    """
    transforms: list[tio.Transform] = []

    if pad_before_affine is not None:
        transforms.append(tio.Pad(pad_before_affine))

    if enable_elastic and elastic_p > 0:
        transforms.append(
            tio.RandomElasticDeformation(
                num_control_points=elastic_num_control_points,
                max_displacement=elastic_max_displacement,
                image_interpolation=elastic_image_interpolation,
                p=elastic_p,
            )
        )

    if enable_affine and affine_p > 0:
        transforms.append(
            tio.RandomAffine(
                scales=affine_scales,
                degrees=affine_degrees,
                translation=affine_translation,
                isotropic=affine_isotropic,
                center=affine_center,
                image_interpolation=affine_image_interpolation,
                p=affine_p,
            )
        )

    if crop_target_shape is not None:
        transforms.append(tio.CropOrPad(crop_target_shape))

    if enable_anisotropy and anisotropy_p > 0:
        transforms.append(
            tio.RandomAnisotropy(
                axes=anisotropy_axes,
                downsampling=anisotropy_downsampling,
                scalars_only=anisotropy_scalars_only,
                image_interpolation=anisotropy_image_interpolation,
                p=anisotropy_p,
            )
        )

    if enable_psf:
        transforms.append(
            tio.Lambda(
                function=partial(
                    anisotropic_psf,
                    sigma_xy=psf_sigma_xy,
                    z_factor=psf_z_factor,
                    max_sigma_frac=psf_max_sigma_frac,
                    rng=psf_rng,
                ),
                types_to_apply=[tio.INTENSITY],
            )
        )

    if enable_poisson_hetero:
        transforms.append(
            tio.Lambda(
                function=partial(
                    poisson_gaussian_hetero,
                    poisson_scale=poisson_scale,
                    sigma0_range=poisson_sigma0_range,
                    sigma1_range=poisson_sigma1_range,
                    rng=poisson_rng,
                ),
                types_to_apply=[tio.INTENSITY],
            )
        )

    if enable_soft_black_clamp:
        transforms.append(
            tio.Lambda(
                function=partial(soft_black_clamp, threshold=clamp_threshold, knee=clamp_knee),
                types_to_apply=[tio.INTENSITY],
            )
        )

    if enable_bias_field and bias_p > 0:
        transforms.append(tio.RandomBiasField(coefficients=bias_coefficients, order=bias_order, p=bias_p))

    if enable_gamma and gamma_p > 0:
        transforms.append(tio.RandomGamma(log_gamma=log_gamma, p=gamma_p))

    if enable_blur and blur_p > 0:
        transforms.append(tio.RandomBlur(std=blur_std, p=blur_p))

    if enable_depth_attenuation:
        transforms.append(
            tio.Lambda(
                function=partial(depth_attenuation, alpha=depth_alpha, rng=depth_rng),
                types_to_apply=[tio.INTENSITY],
            )
        )

    return tio.Compose(transforms)


# -------------------------------------------------------------------------
# Helper: random crop after augmentations
# -------------------------------------------------------------------------
def _random_crop_with_presence(
    img: np.ndarray,
    msk: np.ndarray,
    target_shape: tuple[int, int, int],
    rng: np.random.Generator,
    min_visible_frac: float = 0.05,
    min_visible_voxels: int | None = None,
    max_tries: int = 25,
    edge_prob: float = 0.6,
    edge_margin_frac: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly crop a ``[Z, Y, X]`` volume ensuring some positive mask.

    The crop is biased toward the volume edges with probability
    ``edge_prob``, which encourages tubes to appear partially near the
    borders, better matching realistic TNT appearances.

    Parameters
    ----------
    img: Image volume of shape ``(Z, Y, X)``.
    msk: Binary mask of the same shape as ``img``.
    target_shape: Desired crop size in ``(Z, Y, X)``.
    rng: random number generator.
    min_visible_frac: Minimum fraction of the total positive voxels
    min_visible_voxels: Absolute minimum number of positive voxels in the crop
    max_tries: Maximum number of random crops to attempt before falling back.
    edge_prob: Probability of biasing the crop toward edges.
    edge_margin_frac: Fraction of the maximum offset that defines the "edge region".

    Returns
    -------
    Cropped image and mask of shape ``target_shape``.
    """
    Z, Y, X = img.shape
    tz, ty, tx = target_shape
    assert tz <= Z and ty <= Y and tx <= X, "Target crop larger than source. Increase canvas/padding."

    msum = int(msk.sum())
    if min_visible_voxels is None:
        thresh = max(1, int(msum * float(min_visible_frac)))
    else:
        thresh = max(1, int(min_visible_voxels))

    max_off = (Z - tz, Y - ty, X - tx)

    def sample_offset(axis: int, prefer_edge: bool) -> int:
        m = max_off[axis]
        if m <= 0:
            return 0
        if prefer_edge and rng.random() < 1.0:
            jitter = max(1, int(edge_margin_frac * m))
            if rng.random() < 0.5:
                return int(rng.integers(0, jitter + 1))
            else:
                return int(m - rng.integers(0, jitter + 1))
        return int(rng.integers(0, m + 1))

    for _ in range(max_tries):
        prefer_edge = rng.random() < edge_prob
        z0 = sample_offset(0, prefer_edge)
        y0 = sample_offset(1, prefer_edge)
        x0 = sample_offset(2, prefer_edge)

        crop_m = msk[z0:z0+tz, y0:y0+ty, x0:x0+tx]
        if int(crop_m.sum()) >= thresh:
            crop_i = img[z0:z0+tz, y0:y0+ty, x0:x0+tx]
            return crop_i, crop_m

    pos = np.argwhere(msk > 0)

    if pos.size > 0:
        cz, cy, cx = pos[int(rng.integers(0, len(pos)))]
        z0 = int(np.clip(cz - tz // 2, 0, Z - tz))
        y0 = int(np.clip(cy - ty // 2, 0, Y - ty))
        x0 = int(np.clip(cx - tx // 2, 0, X - tx))
    else:
        z0 = (Z - tz) // 2
        y0 = (Y - ty) // 2
        x0 = (X - tx) // 2

    crop_i = img[z0:z0+tz, y0:y0+ty, x0:x0+tx]
    crop_m = msk[z0:z0+tz, y0:y0+ty, x0:x0+tx]
    return crop_i, crop_m


class SynthTntsDataset_2(Dataset):
    """
    Synthetic tubular dataset with TorchIO-based augmentations.

    For each index, the dataset:

    1. Generates a straight tube on a (possibly larger) 3D canvas.
    2. Wraps the image and mask into a TorchIO :class:`~torchio.Subject`.
    3. Applies geometric and intensity augmentations on the full canvas.
    4. Randomly crops down to the target ``image_shape``, ensuring
       that the tube is at least partially present in the crop.
    5. Optionally dilates the mask and optionally saves intermediate
       and final volumes to disk.
    6. Returns image and mask tensors of shape ``(1, Z, Y, X)``.

    This dataset is intended for pretraining on synthetic tubular
    structures resembling TNTs.

    Parameters
    ----------
    n_samples: Number of samples provided by the dataset.
    image_shape: Final crop size in ``(Z, Y, X)``.
    radius_range, length_range, axes, structure_intensity, per_voxel:
        Parameters forwarded to :func:`make_tube_template`.
    randomize_position: If ``True``, generate the tube on a larger canvas and then take
        a crop; this allows tubes to appear near volume borders.
    canvas_pad: Number of voxels added on each side of the target shape when building the larger canvas
    min_visible_frac, min_visible_voxels, max_crop_attempts,
    edge_prob, edge_margin_frac: Parameters controlling the random
        cropping strategy that guarantees some visible structure.
    augmentations: Optional TorchIO transform to apply. If ``None``,
        a default pipeline is created via :func:`get_augmentations`.
    aug_kwargs: Optional keyword arguments passed to :func:`get_augmentations`
    mask_dtype: Output dtype of the mask tensor
    base_seed: Base seed for sample-specific RNG
    save_samples: If ``True``, saves intermediate and final volumes
    samples_root: Root directory used when ``save_samples=True``.
    mask_dilation_iters: Number of binary dilation iterations applied to the mask after
        augmentation, to make the structures slightly thicker.
    """
    def __init__(
        self,
        n_samples: int,
        image_shape: Tuple[int, int, int] = (7, 128, 128),

        radius_range: Tuple[float, float] = (0.5, 1.5),
        length_range: Tuple[int, int] = (8, 60),
        axes: Sequence[int] = (1, 2),
        structure_intensity: Tuple[float, float] = (0.0, 1.0),
        per_voxel: bool = True,

        randomize_position: bool = True,
        canvas_pad: Tuple[int, int, int] = (3, 60, 60),

        min_visible_frac: float = 0.05,
        min_visible_voxels: Optional[int] = None,
        max_crop_attempts: int = 25,
        edge_prob: float = 0.6,
        edge_margin_frac: float = 0.25,

        augmentations: Optional[tio.Transform] = None,
        aug_kwargs: Optional[dict] = None,

        mask_dtype: torch.dtype = torch.long,
        base_seed: Optional[int] = None,
        save_samples: bool = False,
        samples_root: str = "synth_samples",
        mask_dilation_iters: int = 1,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.image_shape = image_shape

        self.tube_cfg = dict(
            image_shape=image_shape,
            radius_range=radius_range,
            length_range=length_range,
            axes=axes,
            structure_intensity=structure_intensity,
            per_voxel=per_voxel,
        )
        self.randomize_position = randomize_position
        self.canvas_pad = tuple(int(v) for v in canvas_pad)

        self.min_visible_frac = float(min_visible_frac)
        self.min_visible_voxels = min_visible_voxels
        self.max_crop_attempts = int(max_crop_attempts)
        self.edge_prob = float(edge_prob)
        self.edge_margin_frac = float(edge_margin_frac)

        if augmentations is not None:
            self.aug = augmentations
        else:
            aug_kwargs = (aug_kwargs or {}).copy()
            aug_kwargs.setdefault("pad_before_affine", None)
            aug_kwargs.setdefault("crop_target_shape", None)
            self.aug = get_augmentations(**aug_kwargs)

        self.mask_dtype = mask_dtype
        self.base_seed = base_seed

        self.mask_dilation_iters = mask_dilation_iters

        self.save_samples = save_samples
        self.samples_root = samples_root

        if self.save_samples:
            self.templates_dir = os.path.join(self.samples_root, "templates")
            self.final_dir = os.path.join(self.samples_root, "final")
            os.makedirs(self.templates_dir, exist_ok=True)
            os.makedirs(self.final_dir, exist_ok=True)

    def __len__(self) -> int:
        return self.n_samples

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        if self.base_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.base_seed + idx)

    def __getitem__(self, idx: int):
        rng = self._rng_for_index(idx)

        if self.randomize_position:
            tz, ty, tx = self.image_shape
            pz, py, px = self.canvas_pad
            canvas_shape = (tz + 2 * pz, ty + 2 * py, tx + 2 * px)
            tube_cfg = dict(self.tube_cfg)
            tube_cfg["image_shape"] = canvas_shape
            template_np_full, mask_np_full = make_tube_template(rng=rng, **tube_cfg)
        else:
            template_np_full, mask_np_full = make_tube_template(rng=rng, **self.tube_cfg)

        if self.save_samples:
            np.save(
                os.path.join(self.templates_dir, f"img_{idx}.npy"),
                template_np_full.astype(np.float32),
            )
            np.save(
                os.path.join(self.templates_dir, f"mask_{idx}.npy"),
                mask_np_full.astype(np.uint8),
            )
        image_t_full = torch.from_numpy(template_np_full[np.newaxis]).float()  # [1, Zc, Yc, Xc]
        mask_t_full = torch.from_numpy(mask_np_full[np.newaxis].astype(np.int16))  # [1, Zc, Yc, Xc]
        subject_full = tio.Subject(
            image=tio.ScalarImage(tensor=image_t_full),
            mask=tio.LabelMap(tensor=mask_t_full),
        )
        out = self.aug(subject_full) if self.aug is not None else subject_full

        img_full = out.image.tensor[0].cpu().numpy().astype(np.float32)  # [Zc, Yc, Xc]
        msk_full = out.mask.tensor[0].cpu().numpy().astype(np.uint8)  # [Zc, Yc, Xc]

        if self.mask_dilation_iters > 0:
            msk_bin = msk_full > 0
            selem = np.ones((3, 3, 3), dtype=bool)
            msk_bin_dilated = binary_dilation(
                msk_bin,
                structure=selem,
                iterations=self.mask_dilation_iters,
            )
            msk_full = msk_bin_dilated.astype(np.uint8)

        tz, ty, tx = self.image_shape
        Zc, Yc, Xc = img_full.shape
        if tz <= Zc and ty <= Yc and tx <= Xc:
            img_c, msk_c = _random_crop_with_presence(
                img_full,
                msk_full,
                self.image_shape,
                rng,
                min_visible_frac=self.min_visible_frac,
                min_visible_voxels=self.min_visible_voxels,
                max_tries=self.max_crop_attempts,
                edge_prob=self.edge_prob,
                edge_margin_frac=self.edge_margin_frac,
            )
        else:
            raise ValueError(
                f"Augmented volume smaller than target: got {(Zc, Yc, Xc)} vs target {self.image_shape}. "
                f"Increase canvas_pad or remove internal crops in augmentations."
            )
        if self.save_samples:
            np.save(
                os.path.join(self.final_dir, f"img_{idx}.npy"),
                img_c.astype(np.float32),
            )
            np.save(
                os.path.join(self.final_dir, f"mask_{idx}.npy"),
                msk_c.astype(np.uint8),
            )
        image_t = torch.from_numpy(img_c[np.newaxis]).float()
        mask_t = torch.from_numpy(msk_c[np.newaxis].astype(np.int16))

        return image_t.contiguous(), mask_t.to(dtype=self.mask_dtype).contiguous()


# -------------------------------------------------------------------------
# Lightning DataModule
# -------------------------------------------------------------------------
class SynthTntsDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the synthetic tubular dataset.

    This module instantiates :class:`SynthTntsDataset_2` for training,
    validation, and testing, and exposes the standard Lightning
    dataloader methods.

    Parameters
    ----------
    train_samples, val_samples, test_samples: Number of synthetic samples in each split.
    batch_size: Batch size for all dataloaders.
    num_workers: Number of worker processes used by the dataloaders.
    pin_memory: If ``True``, pin memory in dataloaders.
    persistent_workers: Whether to keep worker processes alive across epochs.
    drop_last_train: If ``True``, drop last incomplete batch in the training dataloader.
    image_shape: Common ``(Z, Y, X)`` crop size for all splits.
    radius_range, length_range, axes, structure_intensity, per_voxel:
        Parameters forwarded to :class:`SynthTntsDataset_2`.
    randomize_position, canvas_pad, min_visible_frac, min_visible_voxels,
    max_crop_attempts, edge_prob, edge_margin_frac:
        Crop and placement parameters forwarded to
        :class:`SynthTntsDataset_2`.
    aug: augmentation pipeline
    mask_dtype: Mask dtype used in all splits.
    base_seed: Base seed for the synthetic RNG.
    """
    def __init__(
        self,
        # ----- dataset sizes -----
        train_samples: int = 10000,
        val_samples:   int = 100,
        test_samples:  int = 100,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
        drop_last_train: bool = True,
        image_shape: tuple[int, int, int] = (7, 128, 128),
        radius_range: tuple[float, float] = (0.5, 1.5),
        length_range: tuple[int, int] = (8, 60),
        axes: tuple[int, ...] = (1, 2),
        structure_intensity: tuple[float, float] = (0.4, 1.0),
        per_voxel: bool = True,
        randomize_position: bool = True,
        canvas_pad: tuple[int, int, int] = (2, 32, 32),
        min_visible_frac: float = 0.05,
        min_visible_voxels: int | None = None,
        max_crop_attempts: int = 25,
        edge_prob: float = 0.6,
        edge_margin_frac: float = 0.25,
        aug=None,
        mask_dtype: torch.dtype = torch.float32,
        base_seed: int | None = None,
    ):
        super().__init__()
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last_train = drop_last_train

        self.image_shape = image_shape

        self.radius_range = radius_range
        self.length_range = length_range
        self.axes = axes
        self.structure_intensity = structure_intensity
        self.per_voxel = per_voxel

        self.randomize_position = randomize_position
        self.canvas_pad = canvas_pad
        self.min_visible_frac = min_visible_frac
        self.min_visible_voxels = min_visible_voxels
        self.max_crop_attempts = max_crop_attempts
        self.edge_prob = edge_prob
        self.edge_margin_frac = edge_margin_frac

        self.aug = aug

        self.mask_dtype = mask_dtype
        self.base_seed = base_seed if base_seed else 13
        self.seed_offset = [0, 100_000, 200_000]

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _build_dataset(self, n_samples: int, seed: int) -> SynthTntsDataset_2:
        return SynthTntsDataset_2(
            n_samples=n_samples,
            image_shape=self.image_shape,
            radius_range=self.radius_range,
            length_range=self.length_range,
            axes=self.axes,
            structure_intensity=self.structure_intensity,
            per_voxel=self.per_voxel,
            randomize_position=self.randomize_position,
            canvas_pad=self.canvas_pad,
            min_visible_frac=self.min_visible_frac,
            min_visible_voxels=self.min_visible_voxels,
            max_crop_attempts=self.max_crop_attempts,
            edge_prob=self.edge_prob,
            edge_margin_frac=self.edge_margin_frac,
            augmentations=self.aug,
            mask_dtype=self.mask_dtype,
            base_seed=seed,
        )

    def setup(self, stage: str | None = None) -> None:
        seed_train = self.base_seed + self.seed_offset[0]
        seed_val = self.base_seed + self.seed_offset[1]
        seed_test = self.base_seed + self.seed_offset[2]

        if stage is None or stage == "fit":
            self.train_ds = self._build_dataset(self.train_samples, seed_train)
            self.val_ds = self._build_dataset(self.val_samples, seed_val)

        if stage is None or stage in ("validate", "test"):
            if self.val_ds is None:
                self.val_ds = self._build_dataset(self.val_samples, seed_val)
            self.test_ds = self._build_dataset(self.test_samples, seed_test)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "Call setup('fit') before requesting the train_dataloader."
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last_train,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None, "Call setup('fit' or 'validate') before requesting the val_dataloader."
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None, "Call setup('test') before requesting the test_dataloader."
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
