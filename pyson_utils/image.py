import cv2
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt


# -------------------READ-WRITE-OPERATE------------------------------
def get_paths(dir, input_type='png'):
    paths = glob(os.path.join(dir, '*.{}'.format(input_type)))
    assert len(paths) > 0, '\n\tDirectory:\t{}\n\tInput type:\t{} \n num of paths must be > 0'.format(
        dir, input_type)
    print('Found {} files {}'.format(len(paths), input_type))
    return paths


def write_img(path, rgb):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(path, bgr)


def read_img(path, is_gray=False):
    '''
        Inputs: path, mode 
        Return: rgb or gray depending on the "mode" argument
    '''
    img = cv2.imread(path)
    if not is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
# ----------------------------PLOT


def plot_images(images, cls_true=None, cls_pred=None, space=0.3):
    n = int(np.ceil(np.sqrt(len(images))))
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(n, n)
    fig.subplots_adjust(hspace=space, wspace=space)
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


def show(inp, size=20):
    '''
        Input: either a path or image
    '''
    img = read_img(inp) if type(inp) == str and os.path.exists(inp) else inp
    plt.figure(figsize=(size, size))
    plt.imshow(inp)
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
        # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def putText(image, pos, text):
    return cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 2)


def split_train_test(path_to_dir, val_percent=.2):
    train_path = os.path.join(path_to_dir, 'train')
    val_path = os.path.join(path_to_dir, 'val')
    files = get_paths(path_to_dir)
    np.random.shuffle(files)
    val_max_idx = int(len(files)*val_percent)
    i = 0
    while i < val_max_idx:
        file_name = files[i]
        os.system('mv {} {}'.format(file_name, val_path))
        i += 1
    while i < len(files):
        file_name = files[i]
        os.system('mv {} {}'.format(file_name, train_path))
    print('Train: {}, Val: {}'.format(len(get_paths(train_path)),
                                      get_paths(val_path)))
