import tensorflow as tf



def batchnorm(inputs, training):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=training, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def gen_conv(batch_input, out_channels):
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=3, strides=(1, 1), padding="same", kernel_initializer=initializer)

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


def up_block(act, bn, ngf, training, use_drop):
    bn_shape = tf.shape(bn)
    h, w = bn_shape[1], bn_shape[2]  # bn.get_shape().as_list()[1:3]
    #h *= 2
    #w *= 2
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

def get_deep_generator(generator_inputs, generator_outputs_channels, ngf, use_drop=True, training=True):
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