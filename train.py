from generate_example import generate_random_example
from create_model import create_model
from loss_func import asymmetric_loss_generator
from keras.initializers import RandomNormal
from keras.optimizers import adam
from keras.regularizers import l2
from keras.models import save_model
import numpy as np
from typing import List, Union
from segment_img import read_img, predict_whole_img, visualize_prediction


ROWS = 40
COLS = 40
CHANNELS = 3

K_I = RandomNormal(mean=0.0, stddev=0.2)
B_I = RandomNormal(mean=0.0, stddev=0.2)
K_R = l2(0.01)
B_R = l2(0.01)
KERNEL_SIZE = 3
STRIDE = 2
LAYERS = 1
CONV_ACTIVATION = 'tanh'
CONNECT_ACTIVATION = 'tanh'

BATCH_SIZE = 20
EPOCHS = 70

NEG_RATIO_POS = 100
LEARN_RATE = 0.001


def _example_generator():
    # yielded = 0
    while True:
        xs = []
        ys = []
        for _ in range(0, BATCH_SIZE):
            x, y = generate_random_example(ROWS, COLS, CHANNELS)
            xs.append(x)
            ys.append(y)
        #
        # yielded += 1
        # print('yield', yielded, 'batches')

        yield np.array(xs), np.array(ys)


def train(callback=lambda *args: None):
    model = create_model(rows=ROWS, cols=COLS, channels=CHANNELS, kernel_initializer=K_I, bias_initializer= B_I,
                         kernel_regularizer=K_R, bias_regularizer=B_R, kernel_size=KERNEL_SIZE, strides=STRIDE,
                         conv_activation=CONV_ACTIVATION, connected_activation=CONNECT_ACTIVATION, layers=LAYERS)
    loss_f = asymmetric_loss_generator(NEG_RATIO_POS)
    opt = adam(lr=LEARN_RATE)

    model.compile(optimizer=opt, loss=loss_f)

    model.fit_generator(_example_generator(), steps_per_epoch=BATCH_SIZE, epochs=EPOCHS, use_multiprocessing=True,
                        workers=8)
    callback(model)

if __name__ == '__main__':
    def label_write_to_file(fn: str, lst: List[List[Union[int, float, list]]], normalize=False):
        l_str = []
        avg = 1
        average_func = lambda any_list: sum(any_list) / len(any_list)
        if normalize:
            avg = average_func(list(np.array(lst).flatten()))
        if avg == 0:
            avg = 1
        for row in lst:
            try:
                row_str = ["{0:.2f}".format(float(num / abs(avg))) for num in row]
                l_str.append('\t'.join(row_str))
            except TypeError:
                try:
                    row_str = ["{0:.2f}".format(float(num[0] / abs(avg))) for num in row]
                    l_str.append('\t'.join(row_str))
                except (TypeError, IndexError):
                    print(type(row))
                    print(type(row[0]))
                    l_str.append("TYPE ERROR")

        if normalize:
            l_str.append("\naverage " + str(avg))

        with open(fn, 'w') as f:
            f.write('\n'.join(l_str))

    def main_callback(model):
        # model.save('trained.h5')
        save_model(model, 'trained.h5')
        # row_f = 250
        # col_f = 150
        test_img_rows = 353
        test_img_cols = 546
        file_path = 'test0.png'

        image_to_test, img_file = read_img(file_name=file_path)
        test_prediction = predict_whole_img(model, image_to_test,
                                            test_img_rows, test_img_cols, ROWS, COLS)
        label_write_to_file('prediction.txt', test_prediction, True)
        visualize_prediction(test_prediction, img_file, test_img_rows, test_img_cols, CHANNELS, 'pred_vis.png')
        # for i in range(0, 5):
        #     x, y = generate_random_example(ROWS, COLS, CHANNELS)
        #
        #     pred = model.predict(np.array([x]))[0]
        #     label_write_to_file('lb' + str(i) + '.txt', y)
        #     label_write_to_file('prd' + str(i) + '.txt', pred, True)
        # print(m.get_weights())
        # print(encoder.get_weights())
        # print(decoder.get_weights())

    train(main_callback)
