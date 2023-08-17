import torch


def compatible_load(ckpt_path: str):
    """This function is needed because some ckpt developed early lack some configuration items."""
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    hparams: dict = ckpt['hyper_parameters']
    # hparams.setdefault('estimator_type', 'unet')
    # hparams.setdefault('unet_channels', None)
    # hparams.setdefault('network_lr', 1e-3)
    #
    # hparams['init_network'] = ''
    # hparams['init_optics'] = ''
    # hparams['initialization_type'] = 'default'

    return ckpt, hparams


def add_switch(parser, name, default, help_info):
    parser.add_argument(
        f'--{name}', type=lambda v: bool(int(v)),
        nargs='?', const=True, default=default, help=help_info
    )
