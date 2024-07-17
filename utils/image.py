from typing import Tuple, List, Union, Any, Sequence

import torch
import torch.nn.functional
from torch import Tensor

__all__ = [
    'crop',
    'merge_patches',
    'pad',
    'padded_size',
    'scale',
    'slice_image',
    'slice_image_padded',
    'zoom',

    'Size2d',
]

Size2d = Union[int, Sequence[int], Tuple[int, int]]


def _size(sz: Size2d) -> Tuple[int, int]:
    if isinstance(sz, int):
        return sz, sz
    elif len(sz) == 1:
        return sz, sz
    else:
        return tuple(sz)


def padded_size(original_size: Size2d, cropping: Size2d) -> Tuple[int, int]:
    original_size = _size(original_size)
    cropping = _size(cropping)
    return original_size[0] + 2 * cropping[0], original_size[1] + 2 * cropping[1]


def zoom(image: Tensor, target_size: Size2d):
    if target_size is None:
        return image
    target_size = _size(target_size)
    diff_h = target_size[-2] - image.shape[-2]
    diff_w = target_size[-1] - image.shape[-1]
    pad_h = diff_h // 2
    pad_w = diff_w // 2
    return torch.nn.functional.pad(
        image, (pad_w, diff_w - pad_w, pad_h, diff_h - pad_h),
        mode='constant', value=0
    )


def crop(img: Tensor, cropping: Size2d) -> Tensor:
    cropping = _size(cropping)
    return img[..., cropping[0]:-cropping[0], cropping[1]:-cropping[1]]


def pad(image: Tensor, padding: Size2d, **kwargs) -> Tensor:
    padding = _size(padding)
    return torch.nn.functional.pad(image, (padding[1], padding[1], padding[0], padding[0]), **kwargs)


def scale(image: Tensor, target_size):
    return torch.nn.functional.interpolate(image, size=target_size)


def slice_image(
    img: Tensor,
    patches: Tuple[int, int],
    overlap: Tuple[int, int] = (0, 0),
    sequential: bool = False
) -> Tensor:
    """
    Slice an image into patches, either overlapping or not.
    :param img: One or more images of shape ... x H x W.
    :param patches: Number of patches in vertical and horizontal direction.
    :param overlap: Overlapping width in vertical and horizontal direction.
    :param sequential: Whether to arrange all patches into one dimension.
    :return: Image patches of shape patches[0] x patches[1] x ... x PH x PW
        or (patches[0] x patches[1]) x ... x PH x PW, depending on ``sequential``
        wherein PH and PW are height and width of each patch, respectively.
    """
    img_sz = img.shape[-2:]
    for i in (0, 1):
        if (img_sz[i] - overlap[i]) % patches[i] != 0:
            raise ValueError(f'patches must be divisible by the dimension of image minus overlap '
                             f'in each direction')
    non_overlap_sz = tuple([(img_sz[i] - overlap[i]) // patches[i] for i in (0, 1)])
    patch_sz = tuple([non_overlap_sz[i] + overlap[i] for i in (0, 1)])

    img_patches = [... for _ in range(patches[0])]
    for i in range(patches[0]):
        img_patches[i] = [... for _ in range(patches[1])]
        upper = non_overlap_sz[0] * i
        for j in range(patches[1]):
            left = non_overlap_sz[1] * j
            img_patches[i][j] = img[..., upper:upper + patch_sz[0], left:left + patch_sz[1]]
        img_patches[i] = torch.stack(img_patches[i], 0)

    if sequential:
        tensor = torch.cat(img_patches)
    else:
        tensor = torch.stack(img_patches)

    return tensor


def slice_image_padded(
    img: Tensor,
    patches: Tuple[int, int],
    padding: Tuple[int, int] = (0, 0),
    mode: str = 'constant',
    value: Any = 0,
    sequential: bool = False
) -> Tensor:
    """
    Slice an image into patches, each of which is padded with pixels from neighbouring patches.
    Paddings of marginal patches depend on the parameters ``mode`` and ``value``.
    :param img: One or more images of shape ... x H x W.
    :param patches: Number of patches in vertical and horizontal direction.
    :param padding: Padding width in vertical and horizontal direction.
    :param mode: See :func:``torch.nn.functional.pad``.
    :param value: See :func:``torch.nn.functional.pad``.
    :param sequential: Whether to arrange all patches into one dimension.
    :return: Image patches of shape patches[0] x patches[1] x ... x PH x PW
        or (patches[0] x patches[1]) x ... x PH x PW, depending on ``sequential``
        wherein PH and PW are height and width of each patch, respectively.
    """
    img = torch.nn.functional.pad(img, (padding[1], padding[1], padding[0], padding[0]), mode, value)
    img_patches = slice_image(img, patches, (2 * padding[0], 2 * padding[1]), sequential=sequential)
    return img_patches


def merge_patches(
    patches: Union[Tensor, List[List[Tensor]]],
    overlap: Tuple[int, int] = (0, 0),
    merge_method: str = 'average'
) -> Tensor:
    """
    Merge a set of patches into an image.
    :param patches: Either a tensor of shape PR x PC x ... x PH x PW or a 2d list composed of
        tensors of shape ... x PH x PW, wherein PR, PC are numbers of patches in vertical and
        horizontal direction and PH, PW are height and width of each patch, respectively.
    :param overlap: Overlapping width in vertical and horizontal direction.
    :param merge_method: The method used to determine the values of overlapping pixels.
        Options: average, slope
    :return: A resulted image of shape ... x H x W.
    """
    first = patches[0][0]
    if isinstance(patches, Tensor):
        patch_sz = patches.shape[-2:]
        patch_n = patches.shape[:2]
    else:
        patch_sz = first.shape[-2:]
        patch_n = (len(patches), len(patches[0]))
    img_sz = tuple([patch_n[i] * patch_sz[i] - (patch_n[i] - 1) * overlap[i] for i in (0, 1)])
    shape = first.shape[:-2] + img_sz

    img = torch.zeros(shape)
    img = img.to(first)
    if merge_method == 'average':
        img = _average_merge(img, patch_n, patch_sz, overlap, patches)
    elif merge_method == 'slope':
        img = _slope_merge(img, patch_n, patch_sz, overlap, patches)
    else:
        raise ValueError(f'Unknown merging method: {merge_method}')

    return img


def _average_merge(img, patch_n, patch_sz, overlap, patches):
    for i in range(patch_n[0]):
        upper = i * (patch_sz[0] - overlap[0])
        for j in range(patch_n[1]):
            left = j * (patch_sz[1] - overlap[1])
            img[..., upper:upper + patch_sz[0], left:left + patch_sz[1]] += patches[i][j]

    for i in range(1, patch_n[0]):
        upper = i * (patch_sz[0] - overlap[0])
        img[..., upper:upper + overlap[0], :] /= 2
    for j in range(1, patch_n[1]):
        left = j * (patch_sz[1] - overlap[1])
        img[..., left:left + overlap[1]] /= 2

    return img


def _slope_merge(img, patch_n, patch_sz, overlap, patches):
    shape = patches[0][0].shape
    i = torch.arange(shape[-2]).reshape(-1, 1) / (overlap[0] - 1)
    j = torch.arange(shape[-1]).reshape(1, -1) / (overlap[1] - 1)
    i, j = torch.broadcast_tensors(i, j)
    upper_m, lower_m, left_m, right_m = [
        torch.clamp(mask, 0, 1).to(patches[0][0]) for mask in (i, i.max() - i, j, j.max() - j)
    ]

    for i in range(patch_n[0]):
        upper = i * (patch_sz[0] - overlap[0])
        for j in range(patch_n[1]):
            left = j * (patch_sz[1] - overlap[1])

            p = patches[i][j]
            if i != 0:
                p = p * upper_m
            if i != patch_n[0] - 1:
                p = p * lower_m
            if j != 0:
                p = p * left_m
            if j != patch_n[1] - 1:
                p = p * right_m
            img[..., upper:upper + patch_sz[0], left:left + patch_sz[1]] += p

    return img
