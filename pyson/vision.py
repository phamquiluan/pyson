import cv2
import matplotlib.pyplot as plt

def get_skeleton(img):
    """ Get skeleton mask of a binary image 
        Arguments:
            img: input image 2d
        Returns:
            binnary mask skeleton
    """

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while(not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    kernel = np.ones(shape=[args.line_size, args.line_size])
    _ = cv2.dilate(skel, kernel, iterations=1)
    return _

def convert_mask_to_cell(line_mask):
    """ Convert a mask of lines to cells.
        Arguments:
            line_mask: mask of lines
        Returns:
            a list of cells        
    """
    def is_rect(cnt):
        """ Check if a contour is rectangle.
            Arguments:
                cnt: contours.
            Returns:
                Boolean value if contour is a rectangle.
        """
        _,_,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 0 and w*h >0 and area / w*h > 0.6:
            return True
        else:
            return False

    cnts, hiers = findContours(line_mask)
    out_cnts = {}
    for ci, (cnt, h) in enumerate(zip(cnts, hiers)):
        if is_rect(cnt):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            cnt = np.int0(box)
            if h[-1] == -1:
                out_cnts['table_{}'.format(ci)] = cnt
            else:
                out_cnts['cell_{}_table_{}'.format(ci,h[-1])] = cnt
    return out_cnts

def read_img(path, to_gray=False, scale=(0, 255)):
    """ Read image given a path
        Arguments:
            path: path to image
            to_gray: convert image to gray
            scale: if scale (0, 255)
        Return: 
            output image
    """
    img = cv2.imread(path)
    
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if scale[1] != 255:
        min_val, max_val = scale
        # after: 0-1
        img = img / 255
        # scale 0 -> (max_val-min_val)
        img = img * (max_val - min_val) 
        # scale: min_val -> max_val
        img = img + min_val

    return img

def resize_by_factor(image, factor):
    ''' Regize image by a factor
        Arguments:
            image: input image
            factor: the factor by which the image being resized
        Returns:
            resized image
    '''
    if type(factor) == tuple:
        fx, fy = f
    elif type(f) == float:
        fx = fy = f
    else:
        raise "type of f must be tuple or float"

    return cv2.resize(image, (0, 0), fx=fx, fy=fy)

def resize_by_size(image, size):
    """ Resize image by given size
        Arguments:
            image: input image
            size: the size at which the image being reized

        Returns:
            resized image
    """
    return cv2.resize(iamge, size)


def resize_to_recepfield(image, receptivefiled=256):
    """Resize to the factor of the wanted receptive field.
    Example: Image of size 1100-800 -> 1024-768
    Arguments:
        image: input iamge
        receptivefiled:
    Returns:
        resized image
    """
    new_h, new_w = np.ceil(np.array(image.shape[:2])/receptivefiled).astype(np.int32)*rcf
    image = cv2.resize(image, (new_w, new_h))
    return image

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

def show(inp, size=None, cmap='gray', dpi=300):
    '''
        Input: either a path or image
    '''
    if len(inp.shape) == 4:
        inp = inp[0]
    inp = np.squeeze(inp)
    if type(inp) is str:
        assert os.path.exists(inp)
        inp = read_img(inp)
    if size is None:
        size = max(5, inp.shape[1]//65)
    img = read_img(inp) if type(inp) == str and os.path.exists(inp) else inp
    plt.figure(figsize=(size, size), dpi=dpi)
    plt.imshow(inp, cmap=cmap)
    plt.show()

def findContours(thresh):
    ''' Get contour of a binary image
        Arguments:
            thresh: binary image
        Returns:
            Contours: a list of contour
            Hierarchy: 
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

def get_min_rect(c):
    "input a contour and return the min box of it"
    center, size, angle = cv2.minAreaRect(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = (box/args.resize_ratio).astype(np.int32)
    return box