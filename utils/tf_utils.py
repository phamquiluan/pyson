import tensorflow as tf 
import os

def load_image(path):
    '''
    inputs must be converted to png file
    '''
    assert path[-3:] == 'png'
    raw = tf.read_file(path)
    #raw -> image(range[0,255])
    img_png = tf.image.decode_png(raw)
    #range[0, 255]-> range[0, 1]
    img_float = tf.image.convert_image_dtype(img_png)
    return img_float

def export_generator(fn_generator, ngf, stride, crop_size, checkpoint, output_dir):
    CROP_SIZE = crop_size
    strides = [stride, stride]
    preprocess = lambda x: x*2-1
    deprocess = lambda x: (x+1)/2
        # export the generator to a meta graph that can be imported later for standalone generation
    def extract_patches(image, k_size, strides):
        images = tf.extract_image_patches(tf.expand_dims(
            image, 0), k_size, strides, rates=[1, 1, 1, 1], padding='SAME')[0]
        images_shape = tf.shape(images)
        images_reshape = tf.reshape(
            images, [images_shape[0]*images_shape[1], *k_size[1:3], 3])
        images, n1, n2 = tf.cast(
            images_reshape, tf.uint8), images_shape[0], images_shape[1]
        return images, n1, n2

    def join_patches(images, n1, n2, k_size, strides):
        s1 = k_size[1]//2-strides[1]//2
        s2 = k_size[2]//2-strides[2]//2
        roi = images[:,
                        s1:s1+strides[1],
                        s2:s2+strides[2],
                        :]
        new_shape = [n1, n2, *roi.get_shape().as_list()[1:]]
        print(new_shape)
        reshaped_roi = tf.reshape(roi, new_shape)
        reshaped_roi = tf.transpose(reshaped_roi, perm=[0, 2, 1, 3, 4])
        rs = tf.shape(reshaped_roi)
        rv = tf.reshape(reshaped_roi, [rs[0]*rs[1], rs[2]*rs[3], -1])
        return rv

    def resize(image, new_size=None):
        shape = tf.shape(image)
        h, w = shape[0], shape[1]
        if new_size is None:
            new_h = tf.cast(tf.ceil(h/CROP_SIZE)*CROP_SIZE, tf.int32)
            new_w = tf.cast(tf.ceil(w/CROP_SIZE)*CROP_SIZE, tf.int32)
        else:
            new_h, new_w = new_size
        return tf.image.resize_bilinear(tf.expand_dims(image, 0), (new_h, new_w))[0]
        # inputs = tf.placeholder(tf.float32, [None, *CROP_SIZE, 3], 'inputs')
    inputs = tf.placeholder(tf.float32, [None, None, 3], 'inputs')
    inputs_shape = tf.shape(inputs)
    input_resized = resize(inputs)
    images, n1, n2 = extract_patches(
        input_resized, [1, CROP_SIZE, CROP_SIZE, 1], [1, *strides, 1])
    n1 = tf.identity(n1, 'n1')
    n2 = tf.identity(n2, 'n2')
    batch_input_tensor = tf.identity(images / 255, 'batch_input_tensor')
    batch_input_placeholder = tf.placeholder(
        tf.float32, [None, CROP_SIZE, CROP_SIZE, 3], 'batch_input_placeholder')
    with tf.variable_scope('generator'):
        logits = fn_generator(preprocess(batch_input_placeholder),2, ngf=ngf)
    ft = tf.concat([logits[...,:1], logits[...,:1], logits[...,-1:]], axis=-1)
    batch_output = deprocess(tf.tanh(ft))
    print('batch_output', batch_output)

    h, w = batch_input_placeholder.get_shape().as_list()[1:3]
    batch_output = tf.image.resize_bilinear(batch_output, (h, w))
    batch_output_tensor = tf.identity(
        batch_output, name='batch_output_tensor')
    batch_output_placeholder = tf.placeholder(
        tf.float32, [None, CROP_SIZE, CROP_SIZE, 3], 'batch_output_placeholder')
        
    batch_output = join_patches(batch_output_placeholder, n1, n2, [
                                1, CROP_SIZE, CROP_SIZE, 1], [1, *strides, 1])
    batch_output = resize(batch_output, [inputs_shape[0], inputs_shape[1]])
    outputs = tf.identity(
        tf.cast(batch_output*255, tf.uint8), name='outputs')

    init_op = tf.global_variables_initializer()
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        checkpoint = tf.train.latest_checkpoint(checkpoint)
        print("loading model from checkpoint", checkpoint)
        restore_saver.restore(sess, checkpoint)
        print("exporting model:", checkpoint)
        export_saver.export_meta_graph(
            filename=os.path.join(output_dir, "export.meta"))
        export_saver.save(sess, os.path.join(
            output_dir, "export"), write_meta_graph=False)
    return


class unet_model:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.sess = tf.Session()
        self.build()


    def build(self):
        meta_path = os.path.join(self.checkpoint, 'export.meta')
        tf.train.import_meta_graph(meta_path)
        saver = tf.train.Saver()

        self.inputs = get_tensor_by_name('inputs')
        self.outputs = get_tensor_by_name('outputs')
        self.batch_input_tensor = get_tensor_by_name('batch_input_tensor')
        self.batch_input_placeholder = get_tensor_by_name('batch_input_placeholder')
        self.batch_output_tensor = get_tensor_by_name('batch_output_tensor')
        self.batch_output_placeholder = get_tensor_by_name('batch_output_placeholder')
        self.n1 = get_tensor_by_name('n1')
        self.n2 = get_tensor_by_name('n2')
        saver.restore(self.sess, tf.train.latest_checkpoint(
            self.checkpoint))

    def run(self, path, batch_size=8):
        batch_input, n1_val, n2_val = self.sess.run([self.batch_input_tensor, self.n1, self.n2], {self.inputs:feed_image})
        rv = []
        start = time()
        for i in range(0, len(batch_input), batch_size):
            print('\r {:0.2f} %'.format(i/len(batch_input)), end='')
            rv.append(self.sess.run(self.batch_output_tensor, {self.batch_input_placeholder: batch_input[i:i+batch_size]}))
        output_feed = np.concatenate(rv, axis=0)

        print('Patch speed: ', len(batch_input)/(time()-start))

        return self.sess.run(self.outputs, {self.batch_output_placeholder: output_feed, self.inputs:feed_image})


