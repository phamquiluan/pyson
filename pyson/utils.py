import json
import cv2
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
import time
# Get paths
def get_paths(dir, input_type='png'):
    paths = glob(os.path.join(dir, '*.{}'.format(input_type)))
    assert len(paths) > 0, '\n\tDirectory:\t{}\n\tInput type:\t{} \n num of paths must be > 0'.format(
        dir, input_type)
    print('Found {} files {}'.format(len(paths), input_type))
    return paths

# Split a directory of image in to train and test
def split_train_test(path_to_dir, val_percent=.2):
    train_path = os.path.join(path_to_dir, 'train')
    val_path = os.path.join(path_to_dir, 'val')
    for p in [train_path, val_path]: os.makedirs(p, exist_ok=True)
    files = get_paths(path_to_dir)
    np.random.shuffle(files)
    val_max_idx = int(len(files)*val_percent)
    i = 0
    for i in range(val_max_idx):
        file_name = files[i]
        os.system('mv {} {}'.format(file_name, val_path))
    for i in range(val_max_idx, len(files), 1):
        file_name = files[i]
        os.system('mv {} {}'.format(file_name, train_path))
    print('Train: {}, Val: {}'.format(len(get_paths(train_path)),
                                      get_paths(val_path)))


# Json
def read_json(path):
    '''
        Input: path to json file
        Return: A dictionary of the json file
    '''
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# Image
def write_img(path, rgb):
    "if type float just convert to 255"
    if type(rgb) is float:
        rgb = (255*rgb).astype('uint8')
    if len(rgb.shape) == 2:
        return cv2.imwrite(path, rgb)
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(path, bgr)


def read_img(path, is_gray=False, output_is_float=False):
    '''
        Inputs: path, mode 
        Return: rgb or gray depending on the "mode" argument
    '''
    img = cv2.imread(path)
    if not is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = img/255. if output_is_float else img
    
    return img



# Resize operations
def resize_by_factor(img, f):
    '''
        f: the ratio at which the input will be resize
    '''
    if type(f) == tuple:
        if is_ratio:
            fx, fy = f
    elif type(f) == float:
        fx = fy = f
    else:
        raise "type of f must be tuple or float"

    return cv2.resize(img, (0, 0), fx=fx, fy=fy)

def resize_by_size(img, size):
    return cv2.resize(img, size)


def resize_to_recepfield(image, rcf=256):
    # resize to a factor of the wanted receptive field
    new_h, new_w = np.ceil(np.array(image.shape[:2])/rcf).astype(np.int32)*rcf
    image = cv2.resize(image, (new_w, new_h))
    return image
# ----------------------------PLOT


def plot_images(images, cls_true=None, cls_pred=None, space=0.3, mxn=None, size=5):
    if mxn is None:
        n = int(np.ceil(np.sqrt(len(images))))
        mxn = (n, n)
    fig, axes = plt.subplots(*mxn)
    fig.subplots_adjust(hspace=space, wspace=space)
    fig.figsize=(size, size)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='binary')
            if cls_pred is None and cls_true is not None:
                xlabel = "True: {0}".format(cls_true[i])
            elif cls_pred is None and cls_true is not None:
                xlabel = "True: {0}, Pred: {1}".format(
                    cls_true[i], cls_pred[i])
            else:
                xlabel = None
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()

def show(inp, cmap=None, size=None):
    '''
        Input: either a path or image
    '''
    if type(inp) is str:
        assert os.path.exists(inp)
        inp = read_img(inp)
    size = max(5, inp.shape[1]//65)
    if cmap is None:
        if len(inp.shape)==2:
            cmap='gray'
    img = read_img(inp) if type(inp) == str and os.path.exists(inp) else inp
    plt.figure(figsize=(size, size))
    plt.imshow(inp, cmap=cmap)
    plt.show()

#--------------------------- CONTOUR
def findContours(thresh):
    '''
        Inputs: binary image
        Outputs: contours, hierarchy
    '''
    im2, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy[0]


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return (cnts, boundingBoxes)

def putText(image, pos, text):
    return cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 2)

def random_sample_array(arr, num_of_sample=1):
    l = len(arr)
    return [arr[np.random.choice(l)] for _ in range(num_of_sample)]

def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

def get_min_rect(c):
    "input a contour and return the min box of it"
    center, size, angle = cv2.minAreaRect(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = (box/args.resize_ratio).astype(np.int32)
    return box
# Augmentation

##### Gradient split

def generate_random_gradient(size):
    from random import randint as rint
    img = Image.new("RGB", (size), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    r,g,b = rint(0,255), rint(0,255), rint(0,255)
    dr = (rint(0,255) - r)/500.
    dg = (rint(0,255) - g)/500.
    db = (rint(0,255) - b)/500.
    for i in range(500):
        r,g,b = r+dr, g+dg, b+db
        draw.line((i,0,i,500), fill=(int(r),int(g),int(b)))
    return np.array(img)

def load_db(path_json, max_length=10e12):
    img_dir = os.path.split(path_json)[0]
    d = read_json(path_json)
    labels = list(d.values())
    paths = [os.path.join(img_dir, path) for path in list(d.keys())]
    for i, (p, l) in enumerate(zip(paths, labels)):

        if not type(l) is str \
                or not os.path.exists(p)\
                or len(l) > max_length :

            paths.remove(p)
            labels.remove(l)

        print('\rChecking database: {:0.2f} %'.format(i/len(paths)), end='')
    assert len(paths) == len(labels), '{}-{}'.format(len(paths), len(labels))
    return paths, labels

def load_multiple_db(dbs, max_length=10e12):
    paths, labels = load_db(dbs[0], max_length)
    for db in dbs[1:]:
        rv_ = load_db(db, max_length)
        paths += rv_[0]
        labels += rv_[1]
    return paths, labels

def save_model_keras(model, output_dir):
    from tensorflow.python import keras
    os.makedirs(output_dir, exist_ok=True)
    model_json = model.to_json()
    with open(os.path.join(output_dir, 'model_config.json'), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(output_dir,'weights.h5'))
    print("Saved model to disk")
    
def load_model_keras(output_dir, weight_name='weights.h5'):
    from tensorflow.python import keras
    json_file = open(os.path.join(output_dir, 'model_config.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(output_dir, weight_name))
    print("Loaded model from disk")
    return loaded_model
    

def get_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x1- x2)
    dy = abs(y1- y2)
    return np.argtan(dy/dx)*180 

def mask2cells2(mor1, input_path):
    def is_rect(cnt):
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 0 and w*h >0 and area / w*h > 0.6:
            return True
        else:
            return False
    def get_num_cell(hiers):
        x = hiers[:,-1]
        unique, counts = np.unique(x, return_counts=True)
        return unique[1:]

    # mor2 = cv2.morphologyEx(mor1, cv2.MORPH_CLOSE, np.ones([1,50]))
    cnts, hiers = findContours(mor1)
    mask_debug = np.zeros([*mor1.shape, 3], dtype='uint8')
    i = 0
    j = 1
    k = 2
    out_cnts = {}
    for ci, (cnt, h) in enumerate(zip(cnts, hiers)):
        if is_rect(cnt):
            color1 = int((i%10*255/9))+1
            color2 = int((j%10*255/9))+1
            color3 = int((k%10*255/9))+1
            i+=np.random.choice(10)
            j+=np.random.choice(10)
            k+=np.random.choice(10)
            cv2.drawContours(mask_debug,[cnt], 0, (color1,color2,color3), -1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            cnt = np.int0(box)
            if h[-1] == -1:
                out_cnts['table_{}'.format(ci)] = cnt
            else:
                out_cnts['cell_{}_table_{}'.format(ci,h[-1])] = cnt

    write_img('mask_debug.png', mor1)
    return out_cnts, mask_debug



def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


if __name__ == '__main__':
    path = 'sample_image/1.png'
    image = read_img(path)
    print('Test show from path')
    show(path)
    print('Test show from np image')
    show(image)


