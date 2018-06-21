import cv2 
import numpy as np 
from glob import glob 
import os



def get_paths(dir, input_type='png'):
    paths = glob(os.path.join(dir, '*.{}'.format(input_type)))
    assert len(paths) > 0, '\n\tDirectory:\t{}\n\tInput type:\t{} \n num of paths must be > 0'.format(dir, input_type)
    print('Found {} files {}'.format(len(paths), input_type))
    return paths


def write_img(path, rgb):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(path, bgr)


def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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