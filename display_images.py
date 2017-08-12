import numpy as np
from PIL import  Image
import time

import definitions
import os

ext = '.png'


def image_for_id(identifier, images_root = definitions.TRAIN_IMAGES):
    prefix = os.path.basename(os.path.dirname(images_root))
    file_base_name = "{0}-{1}-0{2}".format(prefix, identifier, ext)
    for dirname in os.listdir(images_root):
        if os.path.isfile(os.path.join(images_root,dirname, file_base_name)):
            return os.path.join(images_root, dirname, file_base_name)
    return None


def show_image(identifier, images_root = definitions.TRAIN_IMAGES):
    im_path = image_for_id(identifier, images_root, ext)
    img = Image.open(im_path)
    Image._show(img)
    return


def show_images_for_sentence(sentence_id, images_root = definitions.TRAIN_IMAGES):
    list_im = [image_for_id("{0}-{1}".format(sentence_id, i), images_root) for i in range(4)]
    imgs = [ Image.open(im) for im in list_im if im is not None]
    if len(imgs)==0:
        raise FileNotFoundError("no images found for sentence id {}".format(sentence_id))
    images_comb = [np.asarray(img) for img in imgs]
    shape = np.shape(images_comb[0])
    sep = images_comb[0][:10,:,:] *0 + 255
    imgs_comb = np.vstack([images_comb[i//2] if i%2==0 else sep for i in range (len(images_comb) *2 -1)  ])
    imgs_comb = Image.fromarray( imgs_comb)
    Image._show(imgs_comb)
    return
