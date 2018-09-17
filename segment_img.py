from typing import List
import cv2
import numpy as np
from PIL import Image


def read_img(file_name) -> List[List[List[float]]]:
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    rows, cols, channels = img.shape
    # print(img.shape)
    img_list = [[[0.0 for _ in range(0, channels)] for _ in range(0, cols)] for _ in range(0, rows)]
    for r in range(0, rows):
        for c in range(0, cols):
            for chn in range(0, channels):
                # print(type(img[r, c, chn]), type(float(img[r, c, chn])), float(img[r, c, chn]))
                img_value = float(img[r, c, chn])
                img_list[r][c][chn] = img_value / 255.0
    return img_list, img


def predict(model, img: List[List[List[float]]], row_from, row_to, col_from, col_to) -> List[List[List[int]]]:
    cropped_img = [[img[r][c] for c in range(col_from, col_to)] for r in range(row_from, row_to)]
    input_data = np.array([cropped_img])
    # model.predict()
    # not done yet
    prediction = model.predict(input_data)[0]
    return prediction


def predict_whole_img(model, img: List[List[List[float]]], rows, cols, row_unit, col_unit, crop_out_bound=True,
                      fill_in=None) \
        -> List[List[List[int]]]:
    # key is the tuple start_row and start_col, value is a List[List[List[int]]]
    if fill_in is None:
        fill_in = [1, 1, 1]
    predict_dic = {}
    # print(len(img))
    # print(len(img[0]))
    # print(len(img[0][0]))
    # print(row_unit, col_unit)
    for row_start in range(0, rows, row_unit):
        for col_start in range(0, cols, col_unit):

            if row_start + row_unit > rows or col_start + col_unit > cols:
                if crop_out_bound:
                    # print('out bound', row_start + row_unit, rows, col_start + col_unit, cols)
                    to_predict = None
                    continue
                else:
                    to_predict = [[img[r][c] if r < rows and c < cols else fill_in
                                   for c in range(col_start, col_start + col_unit)]
                                  for r in range(row_start, row_start + row_unit)]
            else:
                to_predict = [[img[r][c] for c in range(col_start, col_start + col_unit)]
                              for r in range(row_start, row_start + row_unit)]
            if to_predict is not None:
                try:
                    predict_dic[(row_start, col_start)] = predict(model, to_predict, 0, row_unit, 0, col_unit)
                except Exception as e:
                    print(row_start, col_start, row_unit, len(to_predict), rows, cols)
                    raise e
    result = []
    for r in range(0, rows):
        row_result = []
        # if r + row_unit > rows:
        #     continue
        for c in range(0, cols):
            closest_r_s = r - r % row_unit
            closest_c_s = c - c % col_unit
            try:
                row_result.append(predict_dic[(closest_r_s, closest_c_s)][r - closest_r_s][c-closest_c_s])
            except KeyError:
                pass
        result.append(row_result)
    return result


def visualize_prediction(prediction: List[List[List[int]]], original_img, rows, cols, channels, visualize_file):
    # print(type(original_img))
    # print(original_img.shape)
    # new_img_data = []
    for r in range(0, rows):
        # row = []
        for c in range(0, cols):
            # col = []
            for chn in range(0, channels):
                try:
                    original_img[r][c][chn] = min(abs(prediction[r][c][0]) * 255, 255)
                    # col.append(min(int(abs(prediction[r][c][0]) * 255), 255))
                except IndexError:
                    # print(r, c, chn, 'index error')
                    pass
                    # print(r, c, chn)
                    # print(rows, cols, channels)
                    # raise e
            # row.append(col)
            # new_img_data.append(tuple(col))
        # new_img_data.append(row)

    # img_type = 'RGB' if channels == 3 else 'L'
    cv2.imwrite(visualize_file, original_img)
    # new_img = Image.new(img_type, (cols, rows), 'white')
    # try:
    #     new_img.putdata(new_img_data)
    # except Exception as e:
    #     print(new_img_data)
    #     raise e
    # new_img.save(visualize_file)
    # cv2.imshow(visualize_file, original_img)

if __name__ == '__main__':
    # img_lst = read_img('WHATEVER U WANT TO PUT HERE')
    # print(img_lst)
    read_img('test.png')
