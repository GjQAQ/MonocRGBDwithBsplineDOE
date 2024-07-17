import torch
from torch import Tensor

import optics
import utils
from domain import bspline, latticefocal

__all__ = ['BSplineApertureCamera']

INIT_GRID_SZ = 256


class BSplineApertureCamera(optics.ClassicCamera):
    u_matrix: torch.Tensor
    v_matrix: torch.Tensor

    def __init__(
        self,
        aperture_type='circular',
        grid_size=(50, 50),
        knot_vectors=None,
        degrees=(3, 3),
        init_type='default',
        **kwargs
    ):
        r"""
        Construct a camera model whose DOE surface is characterized as a B-spline surface.
        The parameters to be trained are control points :math:`c_{ij}, 0\leq i<N, 0\leq j<M`
        Control points are located evenly on aperture plane, i.e. the area

        :math:`[-D/2, D/2] \times [-D/2, D/2]`

        where D is the diameter of the aperture.
        When computing the height of point :math:`(u,v)` on aperture, the coordinates will be normalized:

        :math:`u'=(u+D/2)/D`

        :param grid_size: Size of control points grid :math:`(N,M)`
        :param knot_vectors: Knot vectors, default to which used in clamped B-spline
        :param degrees:
        :param requires_grad:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.aperture_type = aperture_type
        if aperture_type not in ('circular', 'square'):
            raise ValueError(f'Unknown aperture type: {aperture_type}')

        if knot_vectors is None:
            self.degrees = degrees
            knot_vectors = (
                bspline.clamped_bspline_knots(grid_size[0], degrees[0]),
                bspline.clamped_bspline_knots(grid_size[1], degrees[1])
            )
        else:
            self.degrees = (len(knot_vectors[0]) - grid_size[0] - 1, len(knot_vectors[1]) - grid_size[1] - 1,)
        self.grid_size = grid_size
        self.knot_vectors = knot_vectors

        if init_type == 'lattice_focal':
            init = self.lattice_focal_init()
        elif init_type == 'random':
            init = torch.randn(grid_size) * 1e-7
        else:
            init = torch.zeros(grid_size)
        self.control_points = torch.nn.Parameter(init)

        # buffered tensors used to compute heightmap in psf
        self.register_buffer('u_matrix', self.design_matrix(1), persistent=False)
        self.register_buffer('v_matrix', self.design_matrix(0), persistent=False)

    def heightmap(self):
        return self.__heightmap(
            self.u_matrix,
            self.v_matrix,
            self.control_points.unsqueeze(0)
        )  # n_wl x N_u x N_v

    @torch.no_grad()
    def lattice_focal_init(self):
        slope_range, n = self.prepare_lattice_focal_init()
        r = self.aperture_diameter / 2
        u = torch.linspace(-r, r, INIT_GRID_SZ)[None, ...]
        v = torch.linspace(-r, r, INIT_GRID_SZ)[..., None]
        smap, idx = latticefocal.slopemap(u, v, n, slope_range, self.aperture_diameter, fill='inscribe')
        hmap = latticefocal.slope2height(
            u, v, smap, idx, 12, self.focal_length, self.focal_depth, self.center_n
        )

        x_config, y_config = zip(self.knot_vectors, self.degrees, self.grid_size)
        control_points = bspline.bspline_fit(hmap, *x_config, *y_config)
        return control_points

    @torch.no_grad()
    def aberration(self, u, v, wavelength: float = None):
        wavelength = wavelength or self.design_wavelength
        c = self.control_points.cpu()[None, None, ...]

        r2 = u ** 2 + v ** 2
        scaled_u = self.scale_coordinate(u).squeeze(-2)  # 1 x omega_x x t1
        scaled_v = self.scale_coordinate(v).squeeze(-1)  # omega_y x 1 x t2
        u_mat = self.design_matrices(scaled_u, c.shape[-2], self.knot_vectors[0], self.degrees[0])
        v_mat = self.design_matrices(scaled_v, c.shape[-1], self.knot_vectors[1], self.degrees[1])
        h = self.__heightmap(u_mat, v_mat, c)

        phase = utils.heightmap2phase(h, wavelength, utils.refractive_index(wavelength, self.doe_material))
        phase = torch.transpose(phase, 0, 1)
        return self.apply_stop(torch.exp(1j * phase), r2=r2, x=u, y=v)

    @torch.no_grad()
    def heightmap_log(self, size, normalize=True):
        m = []
        axis = []
        for sz, kv, p in zip(size, self.knot_vectors, self.degrees):
            axis.append(torch.linspace(0, 1, sz))
            m.append(bspline.design_matrix(axis[-1], kv, p))

        u, v = torch.meshgrid(*axis, indexing='xy')
        h = self.__heightmap(*m, self.control_points.cpu())
        u = u - 0.5
        v = v - 0.5
        h = self.apply_stop(h, 0.5, x=u, y=v, r2=u ** 2 + v ** 2).unsqueeze(0)
        if normalize:
            h -= h.min()
            h /= h.max()
        return h

    def apply_stop(self, *args, **kwargs):
        if self.aperture_type == 'circular':
            return super().apply_stop(*args, **kwargs)
        else:
            return self.apply_square_stop(*args, **kwargs)

    def apply_square_stop(self, f, limit=None, **kwargs):
        x, y = kwargs['x'], kwargs['y']
        if limit is None:
            limit = self.aperture_diameter / 2
        return torch.where(
            torch.logical_and(torch.abs(x) < limit, torch.abs(y) < limit),
            f, torch.zeros_like(f)
        )

    @torch.no_grad()
    def design_matrix(self, dim):
        n, kv, p = self.psf_size[dim], self.knot_vectors[dim], self.degrees[dim]
        x = torch.flatten(self.u_grid if dim == 1 else self.v_grid, -2, -1)

        x = self.scale_coordinate(x)
        m = torch.stack([
            bspline.design_matrix(x[i].numpy(), kv, p)
            for i in range(x.shape[0])
        ])
        return m  # n_wl x N x n_ctrl

    def __heightmap(self, u, v, c) -> Tensor:
        h = torch.matmul(torch.matmul(u, c), torch.transpose(v, -1, -2))
        return self.wrap_profile(h)

    @staticmethod
    def design_matrices(x, c_n, kv, p):
        mat = torch.zeros(*x.shape, c_n)

        shape = x.shape
        mat = mat.reshape(-1, x.shape[-1], c_n)
        x = x.reshape(-1, x.shape[-1])
        for i in range(mat.shape[0]):
            mat[i] = bspline.design_matrix(x[i].numpy(), kv, p)
        return mat.reshape(*shape, c_n)
