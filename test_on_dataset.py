from utils import eval_model, config_args

if __name__ == '__main__':
    eval_model(config_args().parse_args(), override={
        'image_sz': 192,
        # 'depth_forcing': True
    })
