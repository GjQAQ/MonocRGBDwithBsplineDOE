r"""
In this module, we define B-spline surface as follows:
.. math::
    f(x,y)=\sum_{i=0}^{n-1}\sum_{j=0}^{n-1} a_{ij}B_{i,d}(x)B_{j,d}(y)
where :math:`d` is *degree*. Given a *knot sequence*

.. math::
    t_0\le t_1\le\ldots\le t_m
where :math:`m=n+d`, :math:`B_{i,d}(t)`: is defined as:

.. math::
    B_{i,0}(t)=\left\{\begin{array}{ll}1,t_i\le t<t_{i+1}\\0,\text{else}\end{array}\right.

    B_{i,k}(t)=\frac{t-t_i}{t_{i+k}-t_i}B_{i,k-1}(t)+\frac{t_{i+k+1}-t}{t_{i+k+1}-t_{i+1}}B_{i+1,k-1}(t),

    k=1,2,\ldots,d

Note that this follows the definition of B-spline in
`Scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html>`_
"""
from scipy.interpolate import BSpline
from scipy.linalg import lstsq
import torch
from torch import Tensor

__all__ = [
    'bspline_fit',
    'clamped_bspline_knots',
    'design_matrix'
]


def clamped_bspline_knots(n: int, d: int, t_0=0, t_m=1, **kwargs) -> Tensor:
    kv = torch.linspace(t_0, t_m, n - d - 1, **kwargs)
    kv = torch.cat([torch.zeros(d + 1), kv, torch.ones(d + 1)])
    return kv


def design_matrix(x: Tensor, knots: Tensor, d: int) -> Tensor:
    return torch.from_numpy(BSpline.design_matrix(x, knots, d).toarray().astype('float32'))


def bspline_fit(
    target: Tensor,
    x_knots: Tensor,
    x_degree: int,
    x_cps: int,
    y_knots: Tensor = None,
    y_degree: int = None,
    y_cps: int = None,
    return_cond: bool = False
):
    if y_knots is None:
        y_knots = x_knots
    if y_degree is None:
        y_degree = x_degree
    if y_cps is None:
        y_cps = x_cps
    x_size, y_size = target.shape

    x = torch.linspace(0, 1, x_size)
    y = torch.linspace(0, 1, y_size)
    b_x = design_matrix(x, x_knots, x_degree)
    b_y = design_matrix(y, y_knots, y_degree)

    mat = torch.zeros(x_size * y_size, x_cps * y_cps)
    for i in range(len(b_x)):
        for j in range(len(b_y)):
            mat[i * y_size + j] = torch.flatten(b_x[i][:, None] @ b_y[j][None, :])

    control_points, _, _, s = lstsq(mat, torch.flatten(target))
    control_points = torch.from_numpy(control_points).reshape(x_cps, y_cps)
    if return_cond:
        return control_points, s[0] / s[-1]
    else:
        return control_points
