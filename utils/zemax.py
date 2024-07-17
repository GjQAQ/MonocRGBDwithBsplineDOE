import torch
from torch import Tensor
import numpy as np
from scipy.interpolate import interpn


def read_wfm(path: str) -> Tensor:
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i < 16:
                continue
            row = line.strip().split()
            row = [float(s) for s in row]
            data.append(row)
    data = torch.tensor(data)
    return data


def wfm_interp(
    wfm_data: Tensor,
    exit_pupil_dia: float,
    sampling_x: Tensor,
    sampling_y: Tensor,
) -> Tensor:
    r = exit_pupil_dia / 2
    # In wavefront error data provided by Zemax, the columns with small index
    # and rows with large index correspond to small x and y coordinates, respectively
    x = np.linspace(-r, r, wfm_data.size(1))
    y = np.linspace(-r, r, wfm_data.size(0))[::-1]
    points = (x, y)
    wfm_data = torch.transpose(wfm_data, 0, 1)
    sampling_grid = torch.stack((sampling_x, sampling_y), -1).cpu().numpy()

    result = interpn(
        points, wfm_data.numpy(), sampling_grid,
        method='linear', bounds_error=False, fill_value=0
    )
    return torch.from_numpy(result)
