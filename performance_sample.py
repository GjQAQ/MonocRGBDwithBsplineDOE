import json
import os.path

import numpy as np

from test_on_dataset import config_args, eval_model


def min_depth(d, s):
    return 2 * d / (2 + s)


def max_depth(d, s):
    return 2 * d / (2 - s)


if __name__ == '__main__':
    parser = config_args()
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--saving_dir', type=str, default='result/curve_data')
    args = parser.parse_args()

    s = np.linspace(0.5, 1.5, 11)
    d1 = min_depth(5 / 3, s).tolist()
    d2 = max_depth(5 / 3, s).tolist()

    result = {
        'measurements': [],
        'label': args.label,
        'slope_range': s.tolist(),
        'min_depth': d1,
        'max_depth': d2
    }
    m = ''
    for i in range(len(s)):
        m, r = eval_model(args, override={'min_depth': d1[i], 'max_depth': d2[i]})
        result['measurements'].append(r[-1][1:])
    result['metrics'] = m

    if not os.path.exists(args.saving_dir):
        os.mkdir(args.saving_dir)
    with open(os.path.join(args.saving_dir, f'{args.label}.json'), 'w') as f:
        json.dump(result, f)
