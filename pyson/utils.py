import json
import numpy as np
from glob import glob
import os
import time


def get_paths(directory, input_type='png'):
    """
        Get a list of input_type paths
        params args:
        return: a list of paths
    """
    paths = glob(os.path.join(directory, '*.{}'.format(input_type)))
    assert len(paths) > 0, '\n\tDirectory:\t{}\n\tInput type:\t{} \n num of paths must be > 0'.format(
        dir, input_type)
    print('Found {} files {}'.format(len(paths), input_type))
    return paths

# Json


def read_json(path):
    '''Read a json path.
        Arguments: 
            path: string path to json 
        Returns:
             A dictionary of the json file
    '''
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def load_db(path_json, max_length=10e12):
    img_dir = os.path.split(path_json)[0]
    d = read_json(path_json)
    labels = list(d.values())
    paths = [os.path.join(img_dir, path) for path in list(d.keys())]
    for i, (p, l) in enumerate(zip(paths, labels)):

        if not type(l) is str \
                or not os.path.exists(p)\
                or len(l) > max_length:

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


def get_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return np.argtan(dy/dx)*180


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def noralize_filenames(directory, ext='*'):
    paths = glob('{}.{}'.format(directory, ext))
    for i, path in enumerate(paths):
        base_dir, base_name = os.path.split(path)
        name, base_ext = base_name.split('.')
        new_name = '{:0>4d}.{}'.format(i, base_ext)
        new_path = os.path.join(base_dir, new_name)
        print('Rename: {} --> {}'.format(path, new_path))
        os.rename(path, new_path)


if __name__ == '__main__':
    path = 'sample_image/1.png'
    image = read_img(path)
    print('Test show from path')
    show(path)
    print('Test show from np image')
    show(image)
