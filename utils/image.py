import cv2 
import numpy as np 
from glob import glob 
import os
import matplotlib.pyplot as plt

def get_paths(dir,input_type='png'):
    paths = glob(os.path.join(dir, '*.{}'.format(input_type)))
    assert len(paths) > 0, '\n\tDirectory:\t{}\n\tInput type:\t{} \n num of paths must be > 0'.format(dir, input_type)
    print('Found {} files {}'.format(len(paths), input_type))
    return paths

def write_img(path, rgb):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(path, bgr)

def read_img(path, mode='rgb'):
    '''
        Inputs: path
        Return: rgb or gray depending on the "mode" argument
    '''
    img = cv2.imread(path)
    if mode == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img 

def resize(img, f, is_ratio=True):
    if type(f) == tuple:
        if is_ratio:
            fx, fy = f
        else:
            assert type(f[0]) == int
            h, w = img.shape[:2]
            h_, w_ = f 
            fx = w_/w
            fy = h_/h
    elif type(f) == float:
        fx = fy = f 
    else:
        raise "type of f must be tuple or float"
    
    return cv2.resize(img, (0, 0), fx=fx, fy = fy)

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
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
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