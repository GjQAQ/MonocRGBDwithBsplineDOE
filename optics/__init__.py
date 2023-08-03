from optics.base import *
from optics.rsc import *
from optics.classic import *
from optics.bsac import *
from optics.zernike import *


register_camera('rs', RotationallySymmetricCamera)
register_camera('b-spline', BSplineApertureCamera)
register_camera('bspline', BSplineApertureCamera)
register_camera('zernike', ZernikeApertureCamera)
