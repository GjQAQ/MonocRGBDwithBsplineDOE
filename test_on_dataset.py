from utils import eval_model, config_args

if __name__ == '__main__':
    eval_model(config_args().parse_args(), override={
        'image_sz': 256,
        'crop_width': 32,
        'padding': 0,
        'psf_effective_factor': 1
    })
