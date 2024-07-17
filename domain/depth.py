__all__ = [
    'center_depth',
    'depth2ips',
    'depth2slope',
    'depth_range',
    'ips2depth',
    'ips2slope',
    'slope2depth',
    'slope2ips',
    'slope_lim',
    'slope_range'
]


def depth2ips(depth, min_depth, max_depth):
    return max_depth * (depth - min_depth) / (depth * (max_depth - min_depth))


def ips2depth(ips, min_depth, max_depth):
    return max_depth * min_depth / (max_depth - ips * (max_depth - min_depth))


def depth2slope(depth, zero_slope):
    return (depth - zero_slope) / depth


def slope2depth(slope, zero_slope):
    return zero_slope / (1 - slope)


def ips2slope(ips, slope_limit):
    return slope_limit * (ips - 0.5)


def slope2ips(slope, slope_limit):
    return slope / slope_limit + 0.5


def center_depth(min_depth, max_depth):
    return 2 * min_depth * max_depth / (min_depth + max_depth)


def slope_lim(min_depth, max_depth):
    return (max_depth - min_depth) / (max_depth - min_depth)


def slope_range(min_depth, max_depth):
    return 2 * slope_lim(min_depth, max_depth)


def depth_range(_center_depth, _slope_range):
    _center_depth2 = 2 * _center_depth
    return _center_depth2 / (2 + _slope_range), _center_depth2 / (2 - _slope_range)
