import tensorflow as tf
import glob
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data')



def plot_image(img, title=None):
    """ Simple plot of the image without mask """
    plt.figure(figsize=(15, 20))
    plt.title(title)
    plt.imshow(img)
    plt.show()


def fimg_to_fmask(img_path):
    """

    @param img_path: the path of the image you want to get the mask from
    @return:path of the mask
    """
    # convert an image file path into a corresponding mask file path
    dirname, basename = os.path.split(img_path)
    maskname = basename.replace(".tif", "_mask.tif")
    return os.path.join(dirname, maskname)



def plot_image_with_mask(img, mask, size = 256, pred_mask = None):
  """ returns a copy of the image with edges of the mask added in red """
    img = (np.array(img)*255).astype(np.uint8)
    mask = (np.array(mask)*255).astype(np.uint8)
    img_color = np.dstack((img, img, img))
    mask = (np.array(mask) > 0).reshape((size,size))
    img_color[mask, 0] = 255  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask, 1] = 0
    img_color[mask, 2] = 0
    if pred_mask is not None:
        pred_mask = (pred_mask*255).astype(np.uint8).reshape((size,size,1))
        pred_mask = (np.array(pred_mask) > 0).reshape(size, size)
        print(np.sum(pred_mask))
        img_color[pred_mask, 1] = 255
    plt.imshow(img_color)
    plt.show()
    return img_color



def show_image_with_mask(img_path):
    """
    Show the image in the path given with its mask
    @param img_path: location of the image e.g.: '../data/train/19_81_mask.tif'
    @return: None
    """
    mask_path = fimg_to_fmask(img_path)
    img = plt.imread(img_path)
    mask = plt.imread(mask_path)
    f_combined = img_path + " & " + mask_path
    plot_image_with_mask(img, mask)
    print('plotted:', f_combined)
    return



def Training_and_test_batch(n_images,test_split, new_size=(544,544), show_images=False):
""" Generates training and test batch using a test_split, use this function after get_annotated data """

    n_train=int(n_images*(1-test_split))

    X,Y = get_annotated_data(n_images, show_images, new_size)
    X_train, Y_train = X[:n_train],Y[:n_train]
    X_test,Y_test = X[n_train:],Y[n_train:]
    return (X_train, Y_train, X_test, Y_test)


def get_annotated_data(n_images,
                       show_images = False,
                       new_size = None):
    """
    Read n_images and transform it into arrays

    >>> get_annotated_data(100, \
                            new_size = (520,520))[0].shape
    (100, 520, 520, 1)


    @param n_images: number of images to be fetched
    @param show_images: whether or not to show the images which are loaded
    @param new_size: if you want the image to be resized to a specific size, specify a tuple (img_height, img_width)
    @return: (X, Y): Arrays of shape (n_images, img.shape[0], img.shape[1], 1) \
                            which represents the images and the associated masks
    """
    f_ultrasounds = [img for img in glob.glob(os.path.join(data_dir,"train/*.tif")) if 'mask' not in img][:n_images]
    print(data_dir)
    f_masks = [fimg_to_fmask(fimg) for fimg in f_ultrasounds][:n_images]
    list_X, list_Y = [], []
    for i in range(int(n_images/50)):
        imgs = [Image.open(f_ultrasound) for f_ultrasound in f_ultrasounds[i*50:(i+1)*50]]
        masks = [Image.open(f_mask) for f_mask in f_masks[i*50:(i+1)*50]]


        if new_size is not None:
            imgs = [img.resize(new_size) for img in imgs]
            masks = [mask.resize(new_size) for mask in masks]
        else:
            new_size = imgs[0].size

        if show_images is True:
            for i in range(max([n_images, 10])):
                plot_image_with_mask(imgs[i], masks[i],new_size[0])
                plt.show()
        list_X.append(np.stack(imgs).reshape((-1, new_size[0], new_size[1], 1)) / 255)
        list_Y.append((np.stack(masks).reshape((-1, new_size[0], new_size[1], 1)) / 255).astype('float32'))
        [img.close() for img in imgs]
        [mask.close() for mask in masks]
    return np.concatenate(list_X), np.concatenate(list_Y)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    get_annotated_data(200)