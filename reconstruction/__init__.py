from .reconstructor import *
from .depthguided import *
from .restormer import *

register_model('plain', Reconstructor)
register_model('depth-guided', DepthGuided)
register_model('restormer', RestormerBased)
