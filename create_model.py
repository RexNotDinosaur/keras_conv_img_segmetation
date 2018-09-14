from keras.models import Model
from keras.layers import Conv2D, LocallyConnected2D, Input, UpSampling2D, Reshape
from keras.initializers import Constant
from keras.regularizers import l2


FILTER_BASE = 8


def create_model(rows: int, cols: int, channels: int, layers: int,
                 kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, kernel_size,
                 conv_activation, connected_activation, strides):
    max_filters = pow(FILTER_BASE, layers)
    input_img = Input(shape=(rows, cols, channels))
    conv2d_layers = []
    for layer_num in range(0, layers):
        if len(conv2d_layers) == 0:
            prev_input = input_img
        else:
            prev_input = conv2d_layers[-1]

        conv2d_layer = Conv2D(filters=max(max_filters // pow(2, layer_num), 1), strides=(strides, strides),
                              padding='same',
                              kernel_size=(kernel_size, kernel_size),
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              activation=conv_activation)
        layer_output = conv2d_layer(prev_input)
        conv2d_layers.append(layer_output)

    img_code = conv2d_layers[-1]

    # locally_connected_encoding = LocallyConnected2D(filters=max(max_filters // pow(2, layers), 1), strides=(1, 1),
    #                                                 kernel_size=(kernel_size, kernel_size),
    #                                                 kernel_initializer=kernel_initializer,
    #                                                 bias_initializer=bias_initializer,
    #                                                 kernel_regularizer=kernel_regularizer,
    #                                                 bias_regularizer=bias_regularizer,
    #                                                 activation=connected_activation)
    # img_code = locally_connected_encoding(conv2d_output)
    # img_code = conv2d_output

    upsampling_layers = []
    for layer_num in range(layers - 1, -1, -1):
        if len(upsampling_layers) == 0:
            prev_code = img_code
        else:
            prev_code = upsampling_layers[-1]
        upsample_layer = UpSampling2D(size=(strides, strides))
        upsampled = upsample_layer(prev_code)

        conv2d_layer = Conv2D(filters=max(max_filters // pow(2, layer_num), 1), strides=(1, 1),
                              padding='same',
                              kernel_size=(kernel_size, kernel_size),
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              activation=conv_activation)
        layer_output = conv2d_layer(upsampled)
        upsampling_layers.append(layer_output)

    conv_out = upsampling_layers[-1]
    locally_connected_decoding = Conv2D(filters=1, strides=(1, 1),
                                        kernel_size=(kernel_size, kernel_size),
                                        kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activation=connected_activation, padding='same')

    decoded = locally_connected_decoding(conv_out)

    # decoded = Reshape(target_shape=(rows, cols))(decoded)
    # print(decoded.shape)
    # decoded = conv_out
    _, decoded_r, decoded_c, _ = decoded._keras_shape
    assert decoded_r == rows
    assert decoded_c == cols
    model = Model(inputs=[input_img], outputs=decoded)

    return model


if __name__ == "__main__":
    create_model(20, 20, 3, 2, Constant(0.0), Constant(0.0), l2(0.0), l2(0.0), 3, 'sigmoid', 'sigmoid', 2)


