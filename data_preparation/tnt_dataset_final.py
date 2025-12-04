import math
import random
from typing import List, Tuple, Optional, Literal

import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

from data_preparation.tnt_io import get_images_and_masks
from data_preparation.tnt_regions import quadrant_rect_coords, suggest_validation_coords, apply_rectangle_mask
from data_preparation.tnt_crops import extract_all_crops, crop_zeros, crops_zeros_all, sample_random_patches_all
from utils.transforms import center_crop_or_pad_after_transform

import napari


class TNTDatasetMasked(Dataset):
    """3D TNT dataset with rectangular masking and cropping strategies.

    This dataset:

    - Loads all image and mask volumes from a root directory.
    - Applies rectangular masking, either removing or isolating a region.
    - Builds crops differently depending on the ``split``:

      * ``"train"``: instance crops and optional random patches.
      * ``"val"``: a single non-zero crop per volume.
      * ``"test"``:
        - if ``masked_part`` is given: large context crop for input, plus
          ROI to evaluate only the smaller masked quadrant.
        - otherwise: a single non-zero crop per volume.

    The dataset returns tensors (as NumPy arrays) of shape ``(1, Z, Y, X)``
    for images, and ``(1, Z, Y, X)`` for masks, with optional ROI for test.

    Parameters
    ----------
    root_dir : Root directory of the dataset.
    split : Dataset split.
    padding : Padding used for instance crops in ``(pz, py, px)``.
    transform : Optional transform applied to image and mask.
                The signature should be ``transform(image=..., mask=...)`` and
                return a dictionary with keys ``"image"`` and ``"mask"``.
    instance_seg : If ``True``, keep instance labels in the mask. If ``False``, masks are binarized.
    random_seed: Seed for random patch sampling.
    random_patches_per_image : Number of random patches to sample per image in the training split.
    min_positive_voxels : Minimum positive voxels in random patches.
    random_size_z : Depth range for random patch sampling.
    random_size_xy : XY range for random patch sampling.
    final_zyx : If given, training crops are padded/cropped to this fixed ``(Z, Y, X)`` size.
    masked_part : Quadrant index in ``{0, 1, 2, 3}`` to apply special masking behavior.
    crop_depth : Depth behavior for instance crops.
    test_context_margin_xy : For test with ``masked_part``: margin (in Y and X) added around
                             the masked quadrant for the larger context crop.
    validation_size : Size ``(height, width)`` of validation rectangle.
    """

    def __init__(self,
                 root_dir: str,
                 split: str = "train",
                 padding: Tuple[int, int, int] = (2, 40, 40),
                 transform=None,
                 instance_seg: bool = False,
                 random_seed: int = 42,
                 random_patches_per_image: int = 0,
                 min_positive_voxels: int = 300,
                 random_size_z: Optional[Tuple[int, int]] = None,
                 random_size_xy: Optional[Tuple[int, int]] = None,
                 final_zyx: Optional[Tuple[int, int, int]] = None,
                 masked_part: Optional[int] = None,
                 crop_depth: Literal["tight", "all"] = "all",
                 test_context_margin_xy: int | Tuple[int, int] = 32,
                 validation_size: Optional[Tuple[int, int]] = None,
                 ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Argument 'split' must be one of 'train', 'val', 'test', got {split!r}."
            )
        self.root_dir = root_dir
        self.split = split
        self.padding = padding
        self.transform = transform
        self.instance_seg = instance_seg
        self.random_seed = random_seed
        self.random_patches_per_image = random_patches_per_image
        self.random_size_z = random_size_z
        self.random_size_xy = random_size_xy
        self.final_zyx = final_zyx
        self.test_context_margin_xy = test_context_margin_xy

        original_images, original_masks = get_images_and_masks(root_dir)

        masked_split = "train" if split in ("train", "val") else "test"

        coords_small = None
        if masked_part is not None:
            coords_small = quadrant_rect_coords(masked_part)
            masked_images_default, masked_masks_default = apply_rectangle_mask(
                original_images, original_masks,
                top=coords_small["top"], bottom=coords_small["bottom"],
                left=coords_small["left"], right=coords_small["right"],
                split=masked_split,
            )

        else:
            masked_images_default, masked_masks_default = apply_rectangle_mask(
                original_images, original_masks, split=masked_split
            )

        if masked_split == "train" and validation_size is not None and masked_part is not None:
            vh, vw = validation_size
            val_coords = suggest_validation_coords(vh, vw, masked_part)

            masked_images_default, masked_masks_default = apply_rectangle_mask(
                masked_images_default, masked_masks_default,
                top=val_coords["top"], bottom=val_coords["bottom"],
                left=val_coords["left"], right=val_coords["right"],
                split="train" if split == "train" else "test",
            )

        if split == "test" and masked_part is not None:
            H = original_images[0].shape[1]
            W = original_images[0].shape[2]

            if isinstance(self.test_context_margin_xy, int):
                my = mx = int(self.test_context_margin_xy)
            else:
                my, mx = self.test_context_margin_xy

            coords_big = {
                "top": max(0, coords_small["top"] - my),
                "bottom": min(H - 1, coords_small["bottom"] + my),
                "left": max(0, coords_small["left"] - mx),
                "right": min(W - 1, coords_small["right"] + mx),
            }

            images_big, _ = apply_rectangle_mask(
                original_images, original_masks,
                top=coords_big["top"], bottom=coords_big["bottom"],
                left=coords_big["left"], right=coords_big["right"],
                split="test",
            )
            _, masks_small = apply_rectangle_mask(
                original_images, original_masks,
                top=coords_small["top"], bottom=coords_small["bottom"],
                left=coords_small["left"], right=coords_small["right"],
                split="test",
            )

            rois_small = []
            for img_full in original_images:
                z, y, x = img_full.shape
                roi = np.zeros((z, y, x), dtype=np.uint8)
                roi[:, coords_small["top"]:coords_small["bottom"] + 1,
                    coords_small["left"]:coords_small["right"] + 1] = 1
                rois_small.append(roi)

            self.crops: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for img_big, msk_small, roi_small in zip(images_big, masks_small, rois_small):
                img_c, msk_c = crop_zeros(img_big, msk_small)
                _, roi_c = crop_zeros(img_big, roi_small)
                self.crops.append((img_c, msk_c, roi_c))

        elif split in ("val", "test"):
            self.crops = crops_zeros_all(masked_images_default, masked_masks_default)

        else:
            self.crops = extract_all_crops(
                masked_images_default, masked_masks_default,
                padding=padding, depth=crop_depth,
            )

        if split == "train" and self.random_patches_per_image > 0:
            rng = random.Random(self.random_seed)
            random_patches = sample_random_patches_all(
                masked_images_default, masked_masks_default,
                n_per_image=self.random_patches_per_image,
                size_range_z=self.random_size_z,
                size_range_xy=self.random_size_xy,
                min_positive_voxels=min_positive_voxels,
                max_attempts=50,
                rng=rng,
            )
            self.crops.extend(random_patches)

    def __len__(self) -> int:
        """Return the number of volumes available in the dataset."""
        return len(self.crops)

    def __getitem__(
            self,
            idx: int
            ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a single sample from the dataset.

        Depending on how the dataset was constructed, this may return:
        - For training and standard validation/test: ``(image, mask)``
        - For test with ``masked_part``:``(image, mask, roi)``
        where all arrays have shape ``(1, Z, Y, X)``.

        Parameters
        ----------
        idx : Sample index.

        Returns
        -------
        Either ``(image, mask)`` or ``(image, mask, roi)``, where
            `image`` and ``mask`` are float and integer volumes,
            respectively, and ``roi`` is a binary mask indicating the
            evaluation region (for test).
        """
        cur = self.crops[idx]

        if len(cur) == 3:
            image_crop, mask_crop, roi_crop = cur
        else:
            image_crop, mask_crop = cur
            roi_crop = None

        # Add channel dimension
        image_crop = image_crop[np.newaxis, ...]
        if not self.instance_seg:
            mask_crop = (mask_crop > 0).astype(np.int32)

        sample = {"image": image_crop, "mask": mask_crop}

        if self.transform is not None:
            sample = self.transform(image=sample["image"], mask=sample["mask"])

        image_out = sample["image"]
        mask_out = sample["mask"]

        if self.split in ("val", "test"):
            _, z, y, x = image_out.shape
            y_new = math.ceil(y / 16) * 16
            x_new = math.ceil(x / 16) * 16

            image_out = center_crop_or_pad_after_transform(image_out, (z, y_new, x_new))
            mask_out = center_crop_or_pad_after_transform(mask_out, (z, y_new, x_new))

            if roi_crop is not None:
                roi_out = center_crop_or_pad_after_transform(
                    roi_crop[np.newaxis, ...], (z, y_new, x_new)
                )

        elif self.final_zyx is not None:
            image_out = center_crop_or_pad_after_transform(image_out, self.final_zyx)
            mask_out = center_crop_or_pad_after_transform(mask_out, self.final_zyx)

            if roi_crop is not None:
                roi_out = center_crop_or_pad_after_transform(
                    roi_crop[np.newaxis, ...], self.final_zyx
                )

        mask_out = mask_out.astype(np.uint8)
        if mask_out.ndim == 3:
            mask_out = np.expand_dims(mask_out, axis=0)

        if roi_crop is not None:
            roi_out = (roi_out > 0).astype(np.uint8)
            return image_out, mask_out, roi_out

        return image_out, mask_out


class TntLightningDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the TNT dataset.

    This module encapsulates the creation of :class:`TNTDatasetMasked`
    instances for training, validation, and testing, and provides the
    respective dataloaders.

    Parameters
    ----------
    root_dir: Root directory of the TNT dataset.
    batch_size: Batch size for all dataloaders.
    num_workers: Number of worker processes for dataloaders.
    train_transform: Transform applied to training samples.
    test_transform: Transform applied to validation and test samples.
    padding: Padding for instance crops.
    random_seed: Seed used for random patch sampling.
    random_patches_per_image:  Number of random patches per image in training.
    min_positive_voxels: Minimum positive voxels in random patches.
    random_size_z: Depth range of random patches.
    random_size_xy: XY range of random patches.
    final_zyx: Fixed size ``(Z, Y, X)`` for training crops.
    masked_part: Quadrant index for special masking behavior.
    crop_depth: Depth behavior for instance crops.
    test_context_margin_xy: Margin for test context crops when ``masked_part`` is set.
    validation_size: Size of the additional validation rectangle to be masked out.
    """
    def __init__(self,
                 root_dir: str,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 train_transform=None,
                 test_transform=None,
                 padding: Tuple[int, int, int] = (3, 80, 80),
                 random_seed: int = 42,
                 random_patches_per_image: int = 75,
                 min_positive_voxels: int = 300,
                 random_size_z: Tuple[int, int] = (5, 8),
                 random_size_xy: Tuple[int, int] = (80, 128),
                 final_zyx: Optional[Tuple[int, int, int]] = None,
                 masked_part: Optional[int] = None,
                 crop_depth: Literal["tight", "all"] = "all",
                 test_context_margin_xy: int | Tuple[int, int] = 32,
                 validation_size: Optional[Tuple[int, int]] = (128, 128),
                 ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.padding = padding
        self.random_seed = random_seed
        self.random_patches_per_image = random_patches_per_image
        self.min_positive_voxels = min_positive_voxels
        self.random_size_z = random_size_z
        self.random_size_xy = random_size_xy
        self.final_zyx = final_zyx
        self.masked_part = masked_part
        self.crop_depth = crop_depth
        self.test_context_margin_xy = test_context_margin_xy
        self.validation_size = validation_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = TNTDatasetMasked(
                root_dir=self.root_dir,
                split="train",
                padding=self.padding,
                transform=self.train_transform,
                random_seed=self.random_seed,
                random_patches_per_image=self.random_patches_per_image,
                min_positive_voxels=self.min_positive_voxels,
                random_size_z=self.random_size_z,
                random_size_xy=self.random_size_xy,
                final_zyx=self.final_zyx,
                masked_part=self.masked_part,
                crop_depth=self.crop_depth,
                validation_size=self.validation_size,
            )

            self.val_dataset = TNTDatasetMasked(
                root_dir=self.root_dir,
                split="val",
                padding=self.padding,
                transform=self.test_transform,
                random_seed=self.random_seed,
                random_patches_per_image=0,
                min_positive_voxels=self.min_positive_voxels,
                random_size_z=self.random_size_z,
                random_size_xy=self.random_size_xy,
                final_zyx=self.final_zyx,
                masked_part=self.masked_part,
                crop_depth=self.crop_depth,
                test_context_margin_xy=self.test_context_margin_xy,
                validation_size=self.validation_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = TNTDatasetMasked(
                root_dir=self.root_dir,
                split="test",
                padding=self.padding,
                transform=self.test_transform,
                random_seed=self.random_seed,
                random_patches_per_image=0,
                min_positive_voxels=self.min_positive_voxels,
                random_size_z=self.random_size_z,
                random_size_xy=self.random_size_xy,
                final_zyx=self.final_zyx,
                masked_part=self.masked_part,
                crop_depth=self.crop_depth,
                test_context_margin_xy=self.test_context_margin_xy,
                validation_size=self.validation_size,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
#
# -----------------------------------------------------------------------------
if __name__ == "__main__":
# #     import multiprocessing as mp
# #     mp.freeze_support()  # safe on Windows; ok to omit if not freezing to EXE
# #     try:
# #         mp.set_start_method("spawn", force=True)  # explicit on Windows
# #     except RuntimeError:
# #         pass  # already set
# #
    from utils.transforms import bv_train_transform, bv_test_transform
# #
# #
    path = r"C:/muni/DP/180322_Sqh-mCh_Tub-GFP_16h_110/180322_Sqh-mCh Tub-GFP 16h_110"
#
    dm_1 = TntLightningDataModule(
        path, batch_size=1, num_workers=0,  # <-- 0 avoids multiprocessing while testing
        train_transform=bv_test_transform(), test_transform=bv_test_transform(), masked_part=0,
        final_zyx=(7, 128, 128)
    )
# #     # dm_2 = TntLightningDataModule(
# #     #     path, batch_size=1, num_workers=0,  # <-- same here
# #     #     train_transform=bv_train_transform(), test_transform=bv_test_transform(), masked_part=0,
# #     #     full_test_image=True, final_zyx=(7, 128, 128)
# #     # )
# #
    dm_1.setup()
# #     # dm_2.setup()
    test_loader_1 = dm_1.test_dataloader()
    train_loader_1 = dm_1.train_dataloader()
    val_loader_1 = dm_1.val_dataloader()
#
#     print(len(val_loader_1))
#     print(len(train_loader_1))
#     print(len(test_loader_1))
#
#     for batch in val_loader_1:
#         images, masks = batch
#         print(f"Image shape: {images.shape}, Mask shape: {masks.shape}") #, print(f"ROI shape: {roi.shape}"))
#         #
#         viewer = napari.Viewer()
#         viewer.add_image(images[0, 0].numpy(), name='image')
#         viewer.add_labels(masks[0, 0].numpy(), name='mask')
#         # viewer.add_labels(roi[0, 0].numpy(), name='roi')
#         napari.run()
#
#         break

