import tensorflow as tf



#------------------------UTILS-------------------
def batchnorm(inputs, training):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=training, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def gen_conv(batch_input, out_channels, strides=1):
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=3, strides=strides, padding="same", kernel_initializer=initializer)

def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)



def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)




def conv_bn_relu(x, filters, training):
    conv = gen_conv(x, filters)
    bn = batchnorm(conv, training)
    return tf.nn.relu(bn)

def down_block(input, ngf,  training, pool_size):
    x = tf.layers.max_pooling2d(input, 2, 2)
    temp = conv_bn_relu(x, ngf, training)

    bn = batchnorm(gen_conv(temp, ngf), training)
    bn += x
    if pool_size == 4:
        bn = tf.layers.max_pooling2d(bn, 2, 2)
    act = tf.nn.relu(bn)
    print(act.shape)
    return bn, act

def up_block(act, bn, ngf, use_drop,training):
    bn_shape = tf.shape(bn)
    h, w = bn_shape[1], bn_shape[2]
    x = tf.image.resize_images(
        act,
        (h, w),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        align_corners=False
    )
    temp = tf.concat([bn, x], axis=-1)
    temp = conv_bn_relu(temp, ngf, training)
    bn = batchnorm(gen_conv(temp, ngf), training)
    output = tf.nn.relu(bn)
    if use_drop:
        output = tf.nn.dropout(output, keep_prob=0.5)
    return output


#--------------------------------GENERATOR
def get_generator_deepunet(generator_inputs, generator_outputs_channels, ngf, use_drop=True, training=True):
    '''
        generator_inputs:512x512x3
        outputs: for line and for text
    '''
    assert generator_outputs_channels is not None
    x = conv_bn_relu(generator_inputs, 64, training)
    print(generator_inputs.shape)
    net = conv_bn_relu(x, ngf, training)
    bn1 = batchnorm(gen_conv(net, ngf), training=training)
    act1 = tf.nn.relu(bn1)
    bn2, act2 = down_block(act1, ngf, pool_size=4, training=training)
    bn3, act3 = down_block(act2, ngf, pool_size=4, training=training)
    bn4, act4 = down_block(act3, ngf, pool_size=2, training=training)
    bn5, act5 = down_block(act4, ngf, pool_size=2, training=training)
    bn6, act6 = down_block(act5, ngf, pool_size=2, training=training)
    bn7, act7 = down_block(act6, ngf, pool_size=2, training=training)
    temp = up_block(act6, bn7, ngf, use_drop=use_drop, training=training)
    temp = up_block(temp, bn6, ngf, use_drop=use_drop, training=training)
    temp = up_block(temp, bn5, ngf, use_drop=use_drop, training=training)
    temp = up_block(temp, bn4, ngf, use_drop=use_drop, training=training)
    temp = up_block(temp, bn3, ngf, use_drop=use_drop, training=training)
    temp = up_block(temp, bn2, ngf, use_drop=use_drop, training=training)
    temp = up_block(temp, bn1, ngf, use_drop=use_drop, training=training)
    logits = gen_conv(temp, generator_outputs_channels)
    return logits



def get_generator_unet(generator_inputs, generator_outputs_channels, ngf, use_drop=True, training=True):
    '''
        generator_inputs:512x512x3
        outputs: for line and for text
    '''
    layers = []
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    scope_name = "encoder_1"
    with tf.variable_scope(scope_name):
        output = gen_conv(generator_inputs, ngf, strides=2)
        layers.append(output)
        print(scope_name, layers[-1].shape[1:])
    layer_specs = [
        ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        scope_name = "encoder_%d" % (len(layers) + 1)
        with tf.variable_scope(scope_name):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, strides=2)
            output = batchnorm(convolved, training)
            layers.append(output)
            print(scope_name, layers[-1].shape[1:])
    layer_specs = [
        (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        scope_name = "decoder_%d" % (skip_layer + 1)
        with tf.variable_scope(scope_name):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output, training=training)

            if dropout > 0.0 and use_drop:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)
            print(scope_name, layers[-1].shape[1:])
    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    scope_name = "outputs"
    with tf.variable_scope(scope_name):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        logits = gen_deconv(rectified, generator_outputs_channels)
        print(scope_name, logits.shape)
    return logits