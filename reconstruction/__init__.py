from .reconstructor import *
from .depthguided import *
from .volguided import *
from .restormer import *

register_model('plain', Reconstructor)
register_model('depth-guided', DepthGuided)
register_model('vol-guided', VolumeGuided)
register_model('restormer', RestormerBased)
