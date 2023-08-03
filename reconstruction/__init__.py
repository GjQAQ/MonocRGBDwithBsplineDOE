from .unet_est import *
from .restormer import *
# from .aoa import *

register_model('unet', UNetBased)
register_model('restormer', RestormerEstimator)
# register_model('aoa', AOARestormerEstimator)
