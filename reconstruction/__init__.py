from .unet_est import *
from .restormer import *

register_model('plain', UNetBased)
register_model('restormer', RestormerAlt)
