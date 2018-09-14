from typing import Tuple, List, Dict, Set
from random import randrange, random, seed, choice, sample
from math import sqrt
import numpy as np

# seed(100)

DIRECTIONS = {
    0: (1, 0),
    45: (1, 1),
    90: (0, 1),
    135: (-1, 1),
    180: (-1, 0),
    225: (-1, -1),
    270: (0, -1),
    315: (1, -1),
    360: (1, 0)
}

DOTTED_LINE_DIFF_COLOR_CODE = 10101
DOTTED_LINE_SAME_COLOR_CODE = 1010
GRADUAL_COLOR_CHANGE_CODE = 987

# CLUSTER_TYPES = [DOTTED_LINE_DIFF_COLOR_CODE,
#                  DOTTED_LINE_SAME_COLOR_CODE]
#               ,
#              *[GRADUAL_COLOR_CHANGE_CODE] * 5]

CLUSTER_TYPES = [GRADUAL_COLOR_CHANGE_CODE]

MAX_DOTTED_RATE = 0.1
MAX_COLOR_CHANGE = 0.2
MAX_GREYNESS = 0.06

_RESTART_RATE = 0.01
_MAX_LEVEL_DIFF = 4
_LEVEL_COE = lambda l: 810 / pow(9, l) + 10
_POINT_AWAY_SCALE = 1


def generate_random_example(rows: int, cols: int, channels=3) -> Tuple[List[List[List[float]]], List[List[List[int]]]]:
    # might raise the same exception as in _gradual_color_change
    generated = False
    while not generated:
        try:
            lb = generate_random_label(rows, cols)
            ipt = generate_input(lb, rows, cols, channels)
            generated = True
        except IndexError:
            pass
    # print(lb)
    return _lst_form(ipt, rows, cols, channels), _lst_form(lb, rows, cols, 1)


def _flatten_channels(lst: List[List[list]]) -> List[list]:
    for i in range(0, len(lst)):
        lst[i] = list(np.array(lst[i]).flatten())
    return lst


def _convert_to_bool(lst: List[List[int]]) -> List[List[bool]]:
    for row in lst:
        for i in range(0, len(row)):
            if row[i] == 0:
                row[i] = False
            else:
                row[i] = True
    return lst


def generate_input(label: Dict[str, int], rows: int, cols: int, channels: int) -> Dict[str, float]:
    clusters, edge = _cluster_label(label, rows, cols)
    code_type = choice(CLUSTER_TYPES)
    if code_type == DOTTED_LINE_SAME_COLOR_CODE:
        return _dotted_line(clusters, edge, rows, cols, channels, False)
    elif code_type == DOTTED_LINE_DIFF_COLOR_CODE:
        return _dotted_line(clusters, edge, rows, cols, channels, True)
    else:
        # print(code_type == GRADUAL_COLOR_CHANGE_CODE)
        return _gradual_color_change(clusters, edge, rows, cols, channels)


def _dotted_line(clusters: List[Set[Tuple[int, int]]], edge: Set[Tuple[int, int]],
                 rows: int, cols: int, channels: int, diff_color: bool) -> Dict[str, float]:
    if diff_color:
        colors = [_random_color(channels) for _ in range(0, len(clusters))]
    else:
        colors = [_random_color(channels)] * len(clusters)

    dotted_rate = random() * MAX_DOTTED_RATE

    img = {}
    blacks = [random() * MAX_GREYNESS for _ in range(0, channels)]

    for cluster, color in zip(clusters, colors):
        for r, c in cluster:
            for chn in range(0, channels):
                img[input_tensor_key(r, c, chn)] = color[chn]
    for r, c in edge:
        dotted = random() < dotted_rate
        if not dotted:
            for chn in range(0, channels):
                img[input_tensor_key(r, c, chn)] = blacks[chn]
        else:
            color_p_opts = []
            for dr, dc in [DIRECTIONS[th] for th in DIRECTIONS.keys() if th != 360]:
                r1, c1 = r + dr, c + dc
                if (r1, c1) not in edge:
                    color_p_opts.append((r1, c1))
                    # p_same_color = (r1, c1)
                    # break
            try:
                p_same_color = choice(color_p_opts)
                for chn in range(0, channels):
                    img[input_tensor_key(r, c, chn)] = img[input_tensor_key(*p_same_color, chn)]
            except (IndexError, KeyError):
                # surrounded by boundary
                for chn in range(0, channels):
                    img[input_tensor_key(r, c, chn)] = blacks[chn]
    return img


def _gradual_color_change(clusters: List[Set[Tuple[int, int]]], edge: Set[Tuple[int, int]],
                          rows: int, cols: int, channels: int) -> Dict[str, float]:
    # print('gradual')
    chn0 = 0
    colors = [_random_color(channels) for _ in range(0, len(clusters))]
    centrals = [choice(tuple(cluster)) for cluster in clusters]
    change_rate = random() * MAX_COLOR_CHANGE
    scale = min(rows, cols)
    img = {}
    for color, central, cluster in zip(colors, centrals, clusters):
        r_c, c_c = central
        for r, c in cluster:
            change_coefficient = sqrt(((r - r_c) * (r - r_c) + (c - c_c) * (c - c_c)) / scale) \
                                 * change_rate
            for chn in range(0, channels):
                img[input_tensor_key(r, c, chn)] = (1 - change_coefficient) * color[chn]
    for r_e, c_e in edge:
        color_p_opts = []
        for dr, dc in [DIRECTIONS[th] for th in DIRECTIONS.keys() if th != 360]:
            r1, c1 = r_e + dr, c_e + dc
            try:
                img[input_tensor_key(r1, c1, chn0)]
                color_p_opts.append((r1, c1))
            except KeyError:
                pass
        p_color_same = choice(color_p_opts)
        # might raise IndexError here, handled at higher level
        for chn in range(0, channels):
            img[input_tensor_key(r_e, c_e, chn)] = img[input_tensor_key(*p_color_same, chn)]
    return img


def _cluster_label(label: Dict[str, int], rows: int, cols: int) -> \
        Tuple[List[Set[Tuple[int, int]]], Set[Tuple[int, int]]]:
    chn = 0
    # pts = set([(r, c) for r, c, _ in all_points_on_graph(rows, cols, 1) if label[input_tensor_key(r, c, chn)] == 0])
    pts = set()
    edge_pts = set()
    for r, c, _ in all_points_on_graph(rows, cols, 1):
        if label[input_tensor_key(r, c, chn)] == 0:
            pts.add((r, c))
        else:
            edge_pts.add((r, c))

    clusters = []

    offset_vec = [DIRECTIONS[k] for k in DIRECTIONS if k != 360]
    while len(pts) > 0:
        initial_pt = choice(tuple(pts))
        # initial_pt = sample(pts, 1)[0]
        cluster = set([initial_pt])
        stack = [initial_pt]
        while len(stack) > 0:
            r0, c0 = stack.pop()
            pts.remove((r0, c0))
            for dr0, dc0 in offset_vec:
                new_pt = r0 + dr0, c0 + dc0
                if 0 <= new_pt[0] < rows and 0 <= new_pt[1] < cols and \
                    new_pt not in cluster and \
                        label[input_tensor_key(*new_pt, chn)] != 1 and \
                        new_pt not in stack:
                    stack.append(new_pt)
                    cluster.add(new_pt)

        clusters.append(cluster)
    return clusters, edge_pts


def generate_random_label(rows: int, cols: int) -> Dict[str,int]:
    #                                                      same with second return value for generate_random_examples
    at_peripheral = lambda r, c: r == 0 or r == rows - 1 or c == 0 or c == cols - 1
    equal_point = lambda p1, p2: p1[0] == p2[0] and p1[1] == p2[1]
    line_points = [(randrange(0, rows), randrange(0, cols))]
    line_end1 = line_points[0]
    line_points.extend(_randomly_adj(line_points[0]))
    line_end2 = line_points[-1]

    end1_diff = _random_start_diffs(_random_level())
    end2_diff = _random_start_diffs(_random_level())

    while (not (at_peripheral(*line_end1) and at_peripheral(*line_end2))) and (not equal_point(line_end1, line_end2)):

        if line_end2 is None:
            line_end2 = line_end1

        if not at_peripheral(*line_end1):
            line_end1 = _update_position(end1_diff[0], line_end1)
        if not at_peripheral(*line_end2):
            line_end2 = _update_position(end2_diff[0], line_end2)

        line_points.append(line_end1)
        line_points.append(line_end2)

        end1_diff = _validate_diffs(_update_diff_level(end1_diff))
        end2_diff = _validate_diffs(_update_diff_level(end2_diff))

        # print(len(line_points), line_end1, line_end2)
    line_points = set(line_points)
    labels = {}
    for r in range(0, rows):
        for c in range(0, cols):
            chn = 0
            if (r, c) in line_points:
                value = 1
            else:
                value = 0
            labels[input_tensor_key(r, c, chn)] = value

    if len(line_points) >= min((rows, cols)) * 0.8:
        return labels
    else:
        return generate_random_label(rows, cols)


def _update_position(theta: float, p: Tuple[int, int]) -> Tuple[int, int]:
    angles = [t for t in DIRECTIONS if abs(t - theta) < 45]
    lower = min(angles)
    upper = max(angles)

    try:
        lower_prob = abs(theta - lower) / (upper - lower)
    except ZeroDivisionError:
        lower_prob = 1

    if random() < lower_prob:
        final_angle = lower
    else:
        final_angle = upper

    dr, dc = DIRECTIONS[final_angle]
    return p[0] + dr, p[1] + dc


def _validate_diffs(diffs: List[float]) -> List[float]:
    # if len(diffs) > 0 and diffs[0] >= 360:
    #     diffs[0] -= 360
    #     return _validate_diffs(diffs)
    # elif len(diffs) > 0 and diffs[0] < 0:
    #     diffs[0] += 360
    #     return _validate_diffs(diffs)
    # else:
    #     return diffs
    for i in range(0, len(diffs)):
        while not (0 <= diffs[i] < 360):
            if diffs[i] < 0:
                diffs[i] += 360
            else:
                diffs[i] -= 360
    return diffs


def _update_diff_level(diffs: List[float]) -> List[float]:
    to_restart = random() < _RESTART_RATE
    if to_restart:
        restart_level = _random_level()
        return _random_start_diffs(restart_level)
    else:
        lg = len(diffs)
        for i in range(2, lg + 1):
            diffs[lg - i] += diffs[lg - i + 1]
        return diffs


def _random_start_diffs(levels: int) -> List[float]:
    # the start of the list is the theta direction of moving
    if levels == 1:
        return [randrange(0, 360)]
    else:
        last_diff = _random_start_diffs(levels - 1)
        c = _LEVEL_COE(len(last_diff))
        last_diff = [c * d for d in last_diff]
        return [randrange(0, 360), *last_diff]


def _random_level():
    try:
        l = round(1/random())
    except ZeroDivisionError:
        l = 10000

    if l >= _MAX_LEVEL_DIFF:
        l = _MAX_LEVEL_DIFF

    if l <= 0:
        l = 1

    return l


def _randomly_adj(p: Tuple[int, int]) -> List[Tuple[int, int]]:
    choices = list(DIRECTIONS.keys())
    choices.remove(360)
    d = DIRECTIONS[choices.pop(randrange(0, len(choices)))]
    return [(p[0] + (mc + 1) * d[0], p[1] + (mc + 1) * d[1]) for mc in range(0, _POINT_AWAY_SCALE)]


def _random_color(channels: int) -> tuple:
    return tuple([random() for i in range(0, channels)])


def _lst_form(ipt: Dict[str, float], rows: int, cols: int, channels: int) -> List[List[list]]:
    return [[[ipt[input_tensor_key(r, c, chn)] for chn in range(0, channels)]
             for c in range(0, cols)] for r in range(0, rows)]


def _lst_form_label(lb: Dict[str, int], rows: int, cols: int) -> List[List[int]]:
    chn0 = 0
    return [[lb[input_tensor_key(r, c, chn0)] for c in range(0, cols)] for r in range(0, rows)]


def input_tensor_key(r: int, c: int, chn: int) -> str:
    return str(r) + ', ' + str(c) + ', ' + str(chn)


def all_points_on_graph(rows: int, cols: int, channels: int) -> Tuple[int, int, int]:
    for r in range(0, rows):
        for c in range(0, cols):
            for chn in range(0, channels):
                yield r, c, chn


if __name__ == '__main__':
    import time
    # s = time.time()
    # lb = generate_random_label(50, 50)
    # e = time.time()
    # print(e - s)
    #
    # s = time.time()
    # ipt = generate_input(lb, 50, 50, 3)
    # e = time.time()
    # print(e - s)
    s = time.time()
    success = 0
    for _ in range(0, 100):
        try:
            ipt, lb = generate_random_example(10, 10, 3)
            # print(ipt)
            # print(lb)
            # print()
            success += 1
        except IndexError:
            pass
    e = time.time()
    print(e - s, success)
    print(ipt)
    print(lb)

    # exp = generate_random_label(100, 100)
    # st_l = []
    # for r in range(0, 100):
    #     st_l.append('\t'.join([str(exp[input_tensor_key(r, c, 0)]) for c in range(0, 100)]))
    # with open('exp_img.txt', 'w') as f:
    #     f.write('\n'.join(st_l))
