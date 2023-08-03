import os
import typing

import torch
import torch.utils.data as data
import numpy as np
import imageio
import kornia.augmentation as augmentation
import kornia.filters as filters
from torchvision import io
from tqdm import tqdm

import dataset
import dataset.img_transform
from utils import srgb_to_linear, crop_boundary


def sf_paths(root):
    paths = {}
    for p in ('train', 'val'):
        paths[p] = {
            'img': os.path.join(
                root, 'FlyingThings3D_subset', p, 'image_clean', 'right'
            ),
            'disparity': os.path.join(
                root, 'FlyingThings3D_subset_disparity', p, 'disparity', 'right'
            )
        }
    return paths


class SceneFlow(data.Dataset):
    def __init__(
        self,
        sf_root: str,
        split: str,
        image_size: typing.Tuple[int, int],
        is_training: bool = True,
        random_crop: bool = False,
        augment: bool = False,
        padding: int = 0,
        n_depths: int = 16,
        ignore_incomplete: bool = True
    ):
        super().__init__()
        self.__dataset_path = sf_paths(sf_root)

        if split not in self.__dataset_path:
            raise ValueError(f'Wrong dataset: {split}; expected: {self.__dataset_path.keys()}')

        self.__transform = dataset.img_transform.RandomTransform(image_size, random_crop, augment)
        self.__centercrop = augmentation.CenterCrop(image_size)

        self.__records = []
        pfm_missing_ids = []
        img_dir = self.__dataset_path[split]['img']
        disparity_dir = self.__dataset_path[split]['disparity']
        for filename in sorted(os.listdir(img_dir)):
            if not filename.endswith('.png'):
                continue

            id_ = os.path.splitext(filename)[0]
            disparity_path = os.path.join(disparity_dir, f'{id_}.pfm')
            if not os.path.exists(disparity_path):
                pfm_missing_ids.append(id_)
                continue

            self.__records.append(id_)

        if pfm_missing_ids and not ignore_incomplete:
            raise ResourceWarning(f'Missing pfm files: {pfm_missing_ids}')

        self.__is_training = torch.tensor(is_training)
        self.__padding = padding
        self.__n_depths = n_depths
        self.split = split

    def __len__(self):
        return len(self.__records)

    def __getitem__(self, item: int) -> dataset.img_transform.ImageItem:
        id_ = self.__records[item]
        img_dir = self.__dataset_path[self.split]['img']
        disparity_dir = self.__dataset_path[self.split]['disparity']

        img, depthmap = self.__prepare(
            os.path.join(img_dir, f'{id_}.png'),
            os.path.join(disparity_dir, f'{id_}.pfm')
        )

        if self.__is_training:
            img, depthmap = self.__transform(img, depthmap)
        else:
            img, depthmap = self.__centercrop(img), self.__centercrop(depthmap)

        # SceneFlow's depthmap has some aliasing artifact. (dfd)
        depthmap = filters.gaussian_blur2d(depthmap, sigma=(0.8, 0.8), kernel_size=(5, 5))
        img, depthmap = img.squeeze(0), depthmap.squeeze(0)

        return dataset.ImageItem(id_, img, depthmap, torch.ones_like(depthmap))

    def __prepare(
        self, img_path: str, disparity_path: str
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        img = imageio.imread(img_path).astype(np.float32) / 255.
        disparity = np.flip(imageio.imread(
            disparity_path, format='pfm'
        ), axis=0).astype(np.float32)

        p = self.__padding
        img = np.pad(img, ((p, p), (p, p), (0, 0)), mode='reflect')
        disparity = np.pad(disparity, ((p, p), (p, p)), mode='reflect')

        img = torch.from_numpy(img).permute(2, 0, 1)
        disparity = torch.from_numpy(disparity)[None, ...]

        disparity -= disparity.min()
        depthmap = 1 - disparity / disparity.max()  # depth of the nearest is 0

        return img, depthmap


class PreCodedSceneFlow(SceneFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        root = args[0]
        self.__pre_path = sf_paths(root + '_pre')[self.split]['img']

    def __getitem__(self, index):
        original = super().__getitem__(index)

        path = os.path.join(self.__pre_path, f'{original[0]}.png')
        img = io.read_image(path).to(torch.float32) / 255
        return original[0], original[1], original[2], original[3], img

    @classmethod
    def prepare(cls, root_path, base, split, model, batch_size=1):
        path = sf_paths(root_path)[split]['img']
        os.makedirs(path, exist_ok=True)
        loader = data.DataLoader(base, batch_size)

        for batch in tqdm(loader, ncols=50, unit='batch'):
            coded, _ = model.image(batch[1].to('cuda'), batch[2].to('cuda'))
            coded = (coded * 255).to(torch.uint8).cpu()
            for i, id_ in enumerate(batch[0]):
                img_path = os.path.join(path, f'{id_}.png')
                io.write_png(coded[i], img_path)
