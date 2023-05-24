import torch


def compatible_load(ckpt_path: str):
    """This function is needed because some ckpt developed early lack some configuration items."""
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    hparams: dict = ckpt['hyper_parameters']
    hparams.setdefault('effective_psf_factor', 2)
    hparams.setdefault('dynamic_conv', False)
    hparams.setdefault('norm', 'BN')
    hparams.setdefault('aperture_type', 'circular')
    hparams.setdefault('reconstructor_type', 'plain')
    hparams.setdefault('depth_refine', False)
    hparams.setdefault('init_optics', '')
    hparams.setdefault('init_cnn', '')

    hparams['initialization_type'] = 'default'

    # if hparams.get('base_ckpt', None) is not None:
    #     hparams['init_optics'] = hparams['base_ckpt']
    #     hparams['init_cnn'] = hparams['base_ckpt']

    return ckpt, hparams
