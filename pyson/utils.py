import json
import cv2
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt



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
    # Create figure with nxn sub-plots.
    fig, axes = plt.subplots(*mxn)
    fig.subplots_adjust(hspace=space, wspace=space)
    fig.figsize=(size, size)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        if i < len(images):
            
            ax.imshow(images[i], cmap='binary')
            # Show true and predicted classes.
            if cls_pred is None and cls_true is not None:
                xlabel = "True: {0}".format(cls_true[i])
            elif cls_pred is None and cls_true is not None:
                xlabel = "True: {0}, Pred: {1}".format(
                    cls_true[i], cls_pred[i])
            else:
                xlabel = None
            # Show the classes as the label on the x-axis.
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


def show(inp, cmap=None, size=5):
    '''
        Input: either a path or image
    '''
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

from sklearn.model_selection import train_test_split