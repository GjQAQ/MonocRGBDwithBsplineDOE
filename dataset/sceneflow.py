import abc
from pathlib import Path
from typing import Literal, Dict, List, Iterator
import warnings

from kornia import filters
import torch
from torch import Tensor
import torch.utils.data as data

import utils

__all__ = [
    'SceneFlowDriving',
    'SceneFlowFlyingThings3D',
    'SceneFlowFlyingThings3DSubset',
    'SceneFlowMonkaa',
]

SF_FILTER = True


class SceneFlowBase(data.Dataset, metaclass=abc.ABCMeta):
    size: int
    _list: List[Path]

    resolution = (540, 960)
    depth_type = 'disp_scale'

    def __init__(self, view: Literal['left', 'right'] = 'right'):
        self.view = view

    def __len__(self):
        return len(self._list)

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        img_path = self._list[item]
        img = utils.imread(img_path)  # 3HW, float 0-1

        disp_path = self._disp_path(img_path)
        disp = utils.imread(disp_path)  # HW, float32
        if SF_FILTER:
            disp = filters.gaussian_blur2d(disp[None, None, ...], 5, (0.8, 0.8)).squeeze(0)
        if disp.max() < 1:
            disp /= disp.max()
        disp = torch.clamp(disp, min=1e-2)
        depth = 1. / disp
        # depth = disp  # todo
        # depth = disp - disp.min().item()
        # depth = depth / depth.max().item()
        # depth = 1. - depth
        return {
            'image': img,  # 3HW
            'depth': depth  # 1HW
        }

    def _get_list(self) -> List[Path]:
        img_paths = []
        missing = []
        for img_path in self._image_paths():
            disp_path = self._disp_path(img_path)
            if not disp_path.exists():
                missing.append(str(disp_path))
                continue
            img_paths.append(img_path)

        img_paths.sort()
        missing.sort()
        if len(missing) != 0:
            warnings.warn(RuntimeWarning(f'Missing disparity files: {missing}'))
        if len(img_paths) != self.size:
            warnings.warn(RuntimeWarning(
                f'Wrong sample number for dataset {self.__class__.__name__}: '
                f'{len(img_paths)} ({self.size} expected)'
            ))
        return img_paths

    @abc.abstractmethod
    def _disp_path(self, img_path: Path) -> Path:
        pass

    @abc.abstractmethod
    def _image_paths(self) -> List[Path]:
        pass


class SceneFlowCommon(SceneFlowBase):
    def __init__(
        self,
        root: utils.Pathlike,
        view: Literal['left', 'right'] = 'right',
        pass_name: Literal['clean', 'final'] = 'clean',
        file_format: Literal['png', 'webp'] = 'png'
    ):
        super().__init__(view)
        self.file_format = file_format
        self.pass_name = pass_name

        root = utils.makepath(root)
        image_dir_name = f'frames_{pass_name}pass'
        if file_format == 'webp':
            image_dir_name += '_webp'
        self._image_dir = root / image_dir_name
        self._disp_dir = root / 'disparity'
        self._list = self._get_list()

    def _disp_path(self, img_path: Path) -> Path:
        scene = img_path.parent.parent.name
        name = img_path.stem
        disp_path = self._disp_dir / scene / self.view / (name + '.pfm')
        return disp_path

    def _image_paths(self) -> Iterator[Path]:
        return self._image_dir.glob(f'*/{self.view}/*.{self.file_format}')


class SceneFlowDriving(SceneFlowCommon):
    def __init__(
        self,
        root: utils.Pathlike,
        view: Literal['left', 'right'] = 'right',
        pass_name: Literal['clean', 'final'] = 'clean',
        focal_length: Literal['15', '35', 'all'] = 'all',
        direction: Literal['forward', 'backward', 'all'] = 'all',
        rate: Literal['fast', 'slow', 'all'] = 'all',
        file_format: Literal['png', 'webp'] = 'png'
    ):
        super().__init__(root, view, pass_name, file_format)
        self.focal_length = '*' if focal_length == 'all' else (focal_length + 'mm_focallength')
        self.direction = '*' if direction == 'all' else ('scene_' + direction + 's')
        self.rate = '*' if rate == 'all' else rate

        if rate == 'fast':
            self.size = 300
        elif rate == 'slow':
            self.size = 800
        else:  # all
            self.size = 1100
        if direction == 'all':
            self.size *= 2
        if focal_length == 'all':
            self.size *= 2

    def _disp_path(self, img_path: Path) -> Path:
        components = img_path.parts
        fl, d, r = components[-5:-2]
        name = img_path.stem
        disp_path = self._disp_dir / fl / d / r / self.view / (name + '.pfm')
        return disp_path

    def _image_paths(self) -> Iterator[Path]:
        return self._image_dir.glob(
            f'{self.focal_length}/{self.direction}/{self.rate}/{self.view}/*.{self.file_format}'
        )


class SceneFlowMonkaa(SceneFlowCommon):
    size = 8664


class SceneFlowFlyingThings3D(SceneFlowCommon):
    _size = {
        'train': (7460, 7460, 7470),
        'test': (1440, 1470, 1460)
    }

    def __init__(
        self,
        root: utils.Pathlike,
        split: Literal['test', 'train'],
        view: Literal['left', 'right'] = 'right',
        pass_name: Literal['clean', 'final'] = 'clean',
        part: Literal['A', 'B', 'C', 'all'] = 'all',
        file_format: Literal['png', 'webp'] = 'png'
    ):
        super().__init__(root, view, pass_name, file_format)
        self.split = split
        self.part = part

        if part == 'all':
            self.size = sum(self._size[split])
        else:
            self.size = self._size[split][ord(part) - ord('A')]

    def _disp_path(self, img_path: Path) -> Path:
        components = img_path.parts
        split, part, scene = components[-5:-2]
        name = img_path.stem
        disp_path = self._disp_dir / split / part / scene / self.view / (name + '.pfm')
        return disp_path

    def _image_paths(self) -> Iterator[Path]:
        split = self.split.upper()
        part = '*' if self.part == 'all' else self.part
        return self._image_dir.glob(f'{split}/{part}/*/{self.view}/*.{self.file_format}')


class SceneFlowFlyingThings3DSubset(SceneFlowBase):
    _size = {'train': 21818, 'val': 4248}

    def __init__(
        self,
        root: utils.Pathlike,
        split: Literal['train', 'val'],
        view: Literal['left', 'right'] = 'right'
    ):
        super().__init__(view)
        self.split = split

        root = utils.makepath(root)
        self._root = root / 'FlyingThings3D_subset'
        self.size = self._size[split]
        self._list = self._get_list()

    def _disp_path(self, img_path: Path) -> Path:
        name = img_path.stem
        disp_path = self._root / self.split / 'disparity' / self.view / (name + '.pfm')
        return disp_path

    def _image_paths(self) -> Iterator[Path]:
        return self._root.glob(f'{self.split}/image_clean/{self.view}/*.png')
