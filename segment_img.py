from typing import List
import cv2
import numpy as np


def read_img(file_name) -> List[List[List[float]]]:
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    rows, cols, channels = img.shape
    img_list = [[[0.0 for _ in range(0, channels)] for _ in range(0, cols)] for _ in range(0, rows)]
    for r in range(0, rows):
        for c in range(0, cols):
            for chn in range(0, channels):
                # print(type(img[r, c, chn]), type(float(img[r, c, chn])), float(img[r, c, chn]))
                img_value = float(img[r, c, chn])
                img_list[r][c][chn] = img_value / 255.0
            #     if img_value / 255.0 < 0.9:
            #         print(r, c, chn)
            # # imgdic[(r, c)] = tuple(img[r, c])
    return img_list


def predict(model, img: List[List[List[float]]], row_from, row_to, col_from, col_to) -> List[List[List[int]]]:
    cropped_img = [[img[r][c] for c in range(col_from, col_to)] for r in range(row_from, row_to)]
    input_data = np.array([cropped_img])
    # model.predict()
    # not done yet
    prediction = model.predict(input_data)[0]
    return prediction

if __name__ == '__main__':
    # img_lst = read_img('WHATEVER U WANT TO PUT HERE')
    # print(img_lst)
    pass
