"""
Utility functions for displaying images from the data set easily.
This should be useful for debugging.
These methods assume that the images are saved in subdirectories under
images_root. So in order to use it, you should first copy the images from 
https://github.com/clic-lab/nlvr to the directory nlvr-data.
"""

import numpy as np
from PIL import  Image
import definitions
import os


ext = '.png'

def __image_for_id(identifier, images_root = definitions.TRAIN_IMAGES):
    prefix = os.path.basename(os.path.dirname(images_root))
    file_base_name = "{0}-{1}-0{2}".format(prefix, identifier, ext)
    for dirname in os.listdir(images_root):
        if os.path.isfile(os.path.join(images_root,dirname, file_base_name)):
            return os.path.join(images_root, dirname, file_base_name)
    return None


def show_image(identifier, images_root = definitions.TRAIN_IMAGES):
    '''
    shows the image associated with the identifier. This is the image that was rendered 
    from the structured representation with that identifier.
    As there are are 6 permutations for each such structured_representation,
    this methods shows the image where the prder of boxes fits their order in the structured representation.
    
    :param identifier: a unique identifier of the form {sentence_id}-{img id}. 
    for example '150-2'
    :param images_root: train/dev/test 
    
    '''
    im_path = __image_for_id(identifier, images_root)
    img = Image.open(im_path)
    Image._show(img)
    return


def show_images_for_sentence(sentence_id, images_root = definitions.TRAIN_IMAGES):
    '''
    
    shows all four images associated with the given sentence id, a=one above the other.
    
    :param sentence_id: a string or int identifying teh sentence. e,g, '150'
    :param images_root: train/dev/test 

    '''
    list_im = [__image_for_id("{0}-{1}".format(sentence_id, i), images_root) for i in range(4)]
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


def show_sample(sample, all_images_for_sentence = False, images_root = definitions.TRAIN_IMAGES):
    '''
    shows the image associates with the given sample, or all images associated with its sentence,
    if all_images_for_sentence=True.
    :param sample: 
    :param all_images_for_sentence: 
    :param images_root: 
    :return: 
    '''
    if not all_images_for_sentence:
        show_image(sample.identifier)
    else:
        sentence_id = str.split(sample.identifier, '-')[0]
        show_images_for_sentence(sentence_id, images_root)




