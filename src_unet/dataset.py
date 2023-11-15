import os
import numpy as np
import rasterio
from skimage import img_as_float

import torch
from torch.utils.data import Dataset


class Fit_Dataset(Dataset):

    def __init__(self,
                 dict_files,
                 num_classes=13,
                 use_augmentations=None,
                 config=None
                 ):
        self.list_imgs = np.array(dict_files["PATH_IMG"])
        self.list_msks = np.array(dict_files["PATH_LABELS"])
        self.use_augmentations = use_augmentations
        self.num_classes = num_classes
        self.config = config

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array

    def read_msk(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_msk:
            array = src_msk.read()[0]
            array[array > self.num_classes] = self.num_classes
            array = array - 1
            # array = np.stack([array == i for i in range(self.num_classes)], axis=0)
            return array

    def __len__(self):
        return len(self.list_imgs)

    def get_item_base_data(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)

        mask_file = self.list_msks[index]
        msk = self.read_msk(raster_file=mask_file)
        return img, msk

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)

        mask_file = self.list_msks[index]
        msk = self.read_msk(raster_file=mask_file)

        if self.use_augmentations is not None:
            sample = {"image": img.swapaxes(0, 2).swapaxes(0, 1), "mask": msk[None, :, :].swapaxes(0, 2).swapaxes(0, 1)}
            transformed_sample = self.use_augmentations(**sample)
            img, msk = transformed_sample["image"].swapaxes(0, 2).swapaxes(1, 2).copy(), \
                       transformed_sample["mask"].swapaxes(0, 2).swapaxes(1, 2).copy()[0]

        img = img_as_float(img)

        return {"img": torch.as_tensor(img, dtype=torch.float),
                "msk": torch.as_tensor(msk, dtype=torch.float)}


class Predict_Dataset(Dataset):

    def __init__(self,
                 dict_files,
                 num_classes=13, use_metadata=True
                 ):
        self.list_imgs = np.array(dict_files["PATH_IMG"])
        self.num_classes = num_classes
        self.use_metadata = use_metadata

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)
        img = img_as_float(img)

        return {"img": torch.as_tensor(img, dtype=torch.float),
                "id": '/'.join(image_file.split('/')[-4:])}
