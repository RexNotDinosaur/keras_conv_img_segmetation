from keras.models import load_model
from segment_img import read_img, visualize_prediction, predict_whole_img
from loss_func import asymmetric_loss_generator


def main(model_path='trained.h5', tests=5, row_unit=40, col_unit=40, neg_ratio_pos=100):
    model = load_model(model_path, custom_objects={'loss_func': asymmetric_loss_generator(neg_ratio_pos)})
    print('loaded')
    test_paths = ['test' + str(i) + '.png' for i in range(0, tests)]
    visualize_paths = ['pred_vis'+str(i) + '.png' for i in range(0, tests)]
    for test_path, vis_path in zip(test_paths, visualize_paths):
        img_data, img_file = read_img(test_path)

        rows, cols, channels = img_file.shape
        prediction = predict_whole_img(model, img_data, rows, cols, row_unit, col_unit, crop_out_bound=False)
        visualize_prediction(prediction, img_file, rows, cols, channels, vis_path)
        # print(img_file.shape)
        # print(len(prediction), len(prediction[0]))

if __name__ == '__main__':
    main()
