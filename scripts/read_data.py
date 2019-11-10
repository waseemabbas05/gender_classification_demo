# -*- coding: utf-8 -*-
"""
@author: Bharath Sankaran
@Copyright: Copyright (c) 2019, Scaled Robotics, SL., All rights reserved.
@filename: read_data.py

Redistribution prohibited without prior written consent

"""
#! /usr/bin/python
import pandas as pd
import numpy as np
from glob import glob

import math
import os
import hashlib
from urllib.request import urlretrieve
import zipfile
import shutil


from PIL import Image
from tqdm import tqdm

database_name = 'celeba'
auxilary_name = 'celeba_labels'
img_dwnld_dir = 'img_align_celeba'
aux_dwnld_dir = 'aux_celeba'

""" Initialize the class """
TRAIN_SET_SIZE = 162770
TEST_SET_SIZE = 19962
VALIDATION_SIZE = 19867
    
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
image_mode = 'RGB'
image_channels = 3

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

class ProgressBar(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
        
def _unzip(save_path, _, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """
    print('Extracting {}...'.format(database_name))

    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)

def download_extract(data_path):
    """
    Download and extract database
    :param data_download_path: data_path
    """
    url_images = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
    url_aux = 'https://s3-eu-west-1.amazonaws.com/public.scaledrobotics.com/Auxilary.zip'
    hash_code_images = '00d2c5bc6d35e252742224ab0c1e8fcb'
    hash_code_aux = '75f2daba5a4e0bda0abe557112a1074f'
    extract_path_img = os.path.join(data_path, img_dwnld_dir)
    extract_path_aux = os.path.join(data_path, aux_dwnld_dir)
    save_path_images = os.path.join(data_path, 'celeba.zip')
    save_path_aux = os.path.join(data_path, 'auxilary.zip')
 
    extract_fn = _unzip
  
    if os.path.exists(extract_path_img):
        print('Found {} Data'.format(database_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path_images):
        with ProgressBar(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(
                url_images,
                save_path_images,
                pbar.hook)

    assert hashlib.md5(open(save_path_images, 'rb').read()).hexdigest() == hash_code_images, \
        '{} file is corrupted.  Remove the file and try again.'.format(save_path_images)
        
    if not os.path.exists(save_path_aux):
        with ProgressBar(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(auxilary_name)) as pbar:
            urlretrieve(
                url_aux,
                save_path_aux,
                pbar.hook)
    
    assert hashlib.md5(open(save_path_aux, 'rb').read()).hexdigest() == hash_code_aux, \
        '{} file is corrupted.  Remove the file and try again.'.format(save_path_aux)

    os.makedirs(extract_path_img)
    os.makedirs(extract_path_aux)
    try:
        extract_fn(save_path_images, extract_path_img, database_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path_img)  # Remove extraction folder if there is an error
        raise err
        
    try:
        extract_fn(save_path_aux, extract_path_aux, auxilary_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path_aux)  # Remove extraction folder if there is an error
        raise err

    # Remove compressed data
    os.remove(save_path_images)
    os.remove(save_path_aux)
    
   
class FeatureClass(object):
    
    def __init__(self):        
        self.feature_dimension =  3920
        
    def extract_basic_features(self, images_array):
                
        """ Extract basic HSV and gray scale features to train a linear model """    
        features = np.empty((len(images_array), self.feature_dimension))
        
        for i, image in enumerate(images_array):
            """ Simple gray scale image """
            gray_image = np.array(Image.fromarray(image.astype('uint8')).convert('LA')).flatten()
            """ Simple HSV image """
            hsv_image = np.array(Image.fromarray(image.astype('uint8')).convert('HSV')).flatten()
            """ Normalize feature space """
            gray_feature = gray_image.astype(np.float32)/255.0
            hsv_feature = hsv_image.astype(np.float32)/255.0
            
            """ Accumulate feature """
            features[i, :] = np.concatenate([gray_feature, hsv_feature])
            
        return features
    
        
class Dataset(object):
    """
    Dataset
    """
    def __init__(self, dataset_path, image_width = 28,
                 image_height = 28, image_mode = 'RGB', channels = 3,
                 img_dir = img_dwnld_dir, 
                 aux_dir = aux_dwnld_dir):
        """
        Initalize the class
        :param dataset_name: Database name
        :param data_files: List of files in the database
        """
        self.im_width = image_width
        self.im_height = image_height
        self.im_mode = image_mode
        self.im_channels = channels
        self.dataset_path = dataset_path
                
        self.img_dwnld_dir = img_dir
        self.aux_dwnld_dir = aux_dir

        self.column_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                             'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                             'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
                             'Double_Chin', 'Eyeglasses', 'Goatee','Gray_Hair', 'Heavy_Makeup',
                             'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
                             'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                             'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                             'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 
                             'Wearing_Necktie', 'Young']

        """ Initialize the class """
        self.TRAIN_SET_SIZE = 162770
        self.DATA_SET_SIZE = 202599
        self.VALIDATION_SIZE = 19867
        
        self.X_train = None
        self.Y_train = None
        self.X_validate = None
        self.Y_validate = None
        
        self.prepare_data()
        
    def prepare_data(self):
        
        """ Get images from dataset """
        print('Loading images from disk ...')
        celeb_images, indices = self.get_image_batch(self.DATA_SET_SIZE)
        
        """ get an indexable list using panda"""
        print('Finished images from disk, loading attribute files ...')
        pd_indices = pd.DataFrame(index=indices, data=np.arange(len(indices)))
        pd_indices.sort_index(inplace=True)
        
        """ Get labels from dataset """
        full_label_set, eval_data = self.get_labels()        
        
        val_indices = eval_data.index[eval_data.loc[:,1] == 1]
        train_indices = eval_data.index[eval_data.loc[:,1] == 0]

        """ Getting image indices for validation and train set"""        
        val_img_indices = pd_indices.loc[val_indices].values
        train_img_indices = pd_indices.loc[train_indices].values
        
        """ Clean images in the event of missing files """
        train_img_indices = train_img_indices[~ np.isnan(train_img_indices)]
        val_img_indices = val_img_indices[~ np.isnan(val_img_indices)]
        
        self.Y_train = full_label_set.loc[train_indices].values
        self.Y_validate = full_label_set.loc[val_indices].values
        self.X_train = celeb_images[train_img_indices.astype(int),:,:,:]
        self.X_validate = celeb_images[val_img_indices.astype(int),:,:,:]
        
    @staticmethod
    def get_image(image_path, width, height, mode):
        """
        Read image from image_path
        :param image_path: Path of image
        :param width: Width of image
        :param height: Height of image
        :param mode: Mode of image
        :return: Image data
        """
        image = Image.open(image_path)
    
        if image.size != (width, height):  # HACK - Check if image is from the CELEBA dataset
            # Remove most pixels that aren't part of a face
            face_width = face_height = 108
            j = (image.size[0] - face_width) // 2
            i = (image.size[1] - face_height) // 2
            image = image.crop([j, i, j + face_width, i + face_height])
            image = image.resize([width, height], Image.BILINEAR)
    
        return np.array(image.convert(mode))
        
  
    def get_image_batch(self, number_of_images):
        image_files = sorted(glob(
            os.path.join(self.dataset_path, self.img_dwnld_dir, '*.jpg')
            )[:number_of_images])
        
        indices = [os.path.basename(x) for x in image_files]
        data_batch = np.array(
            [Dataset.get_image(sample_file, 
                               self.im_width, 
                               self.im_height, 
                               self.im_mode) 
            for sample_file in image_files]).astype(np.float32)
    
        # Make sure the images are in 4 dimensions
        if len(data_batch.shape) < 4:
            data_batch = data_batch.reshape(data_batch.shape + (1,))
    
        return data_batch, indices

    def get_batches(self, batch_size):
        """
        Generate batches
        :param batch_size: Batch Size
        :return: Batches of data
        """
        IMAGE_MAX_VALUE = 255
        
        image_files = sorted(glob(
            os.path.join(self.dataset_path, self.img_dwnld_dir, '*.jpg')
            )[:self.DATA_SET_SIZE])
        
        self.data_files = image_files
        self.shape = len(self.data_files), self.IMAGE_WIDTH, \
        self.IMAGE_HEIGHT, self.image_channels

        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            data_batch = self.get_image_batch(
                    self.data_files[current_index:current_index + batch_size],
                    *self.shape[1:3],
                    self.image_mode)

            current_index += batch_size

            yield data_batch / IMAGE_MAX_VALUE - 0.5
            
    
    def get_labels(self):
    
        attr_file = os.path.join(self.dataset_path, self.aux_dwnld_dir, 'list_attr_celeba.txt')
        attr_data = pd.read_csv(attr_file, names=self.column_names,
                          na_values = "?", comment='\t',
                          sep=" ", skipinitialspace=True, header=1)
        
        gender_labels = attr_data[{'Male'}]
        
        eval_file = os.path.join(self.dataset_path, self.aux_dwnld_dir, 'list_eval_partition.txt')
        eval_data = pd.read_csv(eval_file, na_values = "?", comment='\t',
                          sep=" ", index_col=0, skipinitialspace=True, header=None)
        
        return gender_labels, eval_data         
            
    @staticmethod    
    def images_display_grid(images, mode):
        """
        Save images as a square grid
        :param images: Images to be used for the grid
        :param mode: The mode to use for images
        :return: Image of images in a square grid
        """
        # Get maximum size for square grid of images
        save_size = math.floor(np.sqrt(images.shape[0]))
    
        # Scale to 0-255
        images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)
    
        # Put images in a square arrangement
        images_in_square = np.reshape(
                images[:save_size*save_size],
                (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
        if mode == 'L':
            images_in_square = np.squeeze(images_in_square, 4)
    
        # Combine images to grid image
        new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
        for col_i, col_images in enumerate(images_in_square):
            for image_i, image in enumerate(col_images):
                im = Image.fromarray(image, mode)
                new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))
    
        return new_im
            
    

    
    
    

    
    
    
    
    
