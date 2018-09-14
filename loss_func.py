from math import pow, e
# import tensorflow as tf
import keras.backend as K
from typing import List

# NEGATIVE_ERROR_RATIO_POSITIVE_ERROR = 100
ERR_RATIO = 0.05


def asymmetric_loss_generator(neg_ratio_pos_err):
    def loss_func(last_input: List[list], output_holders: List[list]):
        return _asymmetric_case_loss(last_input, output_holders, neg_ratio_pos_err)
    return loss_func


def _asymmetric_case_loss(last_input: List[list], output_holders: List[list],
                          negative_ratio_positive):
    n = _asymmetry_coefficient(negative_ratio_positive)
    # loss = tf.reduce_mean(tf.add_n([
    #     tf.pow(e, (output_holders[tp] - last_input[tp]) * n) - n * (output_holders[tp] - last_input[tp]) - 1
    #     for tp in last_input
    # ]))

    # loss = tf.reduce_mean(tf.add_n([
    #     tf.pow(e, (output_holders[tp] - last_input[tp]) * n) - n * (output_holders[tp] - last_input[tp]) - 1
    #     for lb, rs in zip(_iterate_list(output_holders), _iterate_list())
    # ]))
    flattened_result = K.flatten(last_input)
    flattened_label = K.flatten(output_holders)
    x = flattened_label - flattened_result

    # print(x.__dict__)
    loss = K.mean(K.pow(e, n * x) - n * x - 1)
    return loss


# def _iterate_list(lst: list):
#     for item in lst:
#         if isinstance(item, list):
#             yield from _iterate_list(item)
#         else:
#             yield item


def _asymmetry_coefficient(c: float, err_ratio=ERR_RATIO) -> float:
    def fr(x: float):
        if x == 0:
            return 1
        else:
            return (pow(e, x) - x -1)/(pow(e, -x) + x - 1)

    def actual_f(x: float, n: float):
        return pow(e, x * n) - n * x - 1

    if c <= 0 or c == 1:
        raise ValueError
    elif c > 1:
        x0 = 0.5
        x1 = 10.0
        while fr(x1) <= c:
            x1 = x1 * 2
        while fr(x0) >= c:
            x0 = x0 / 2
        m = (x0 + x1) / 2
        while abs((fr(m) - c) / c) > err_ratio:
            if fr(m) < c:
                x0 = m
            else:
                x1 = m
            m = (x0 + x1) / 2

        print(m)
        print(fr(m))
        print(actual_f(1, m), actual_f(-1, m))
        return m
    else:
        result = -1 * _asymmetry_coefficient(1/c, err_ratio)
        print(result)
        print(fr(result))
        print(actual_f(1, result), actual_f(-1, result))
        return result