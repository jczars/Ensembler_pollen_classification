#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:22:28 2024

@author: jczars
"""
# import
import csv, os
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# funções auxiliares
def create_folders(_save_dir, flag=1):
    """
  -->create folders
  :param: _save_dir: path the folder
  :param: flag: rewrite the folder, (1 for not and display error: 'the folder already exists)
  """
    if os.path.isdir(_save_dir):
        if flag:
            raise FileNotFoundError("folders test already exists: ", _save_dir)
        else:
            print('folders test already exists: ', _save_dir)
    else:
        os.mkdir(_save_dir)
        print('create folders test: ', _save_dir)

def read_print_csv(filename_csv):
    """
    -->read and print the row from csv file
    :param: filename_csv: file name 
    :param: data: data to be inserted into the csv file
    """
    with open(filename_csv, 'r') as file:
        read_csv = csv.reader(file)
        for linha in read_csv:
            print(linha)
def add_row_csv(filename_csv, data):
    """
    -->add new row in csv file
    :param: filename_csv: file name 
    :param: data: data to be inserted into the csv file
    """
    with open(filename_csv, 'a') as file:
        # creating a csv writer object
        csvwriter = csv.writer(file)
        # writing the data rows
        csvwriter.writerows(data)
        
def load_data_train(PATH_BD, K, BATCH, INPUT_SIZE, SPLIT_VALID):
    """
    -->loading train data 
    :param: PATH_BD: file name 
    :param: K: k the kfolders values
    :param: BATCH: batch size
    :param: INPUT_SIZE: input dimensions, height and width, default=(224,224)
    :param: SPLIT_VALID: portion to divide the training base into training and validation
    return: train and valid dataset
    """
    train_dir = PATH_BD + '/Train/k' + str(K)
    print('train_dir ', train_dir)
    
    idg = ImageDataGenerator(rescale=1. / 255, validation_split=SPLIT_VALID)

    train_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='training')

    val_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='validation')

    return train_generator, val_generator

def load_data_train_aug(PATH_BD, K, BATCH, INPUT_SIZE, SPLIT_VALID):
    """
    -->loading train data 
    :param: PATH_BD: file name 
    :param: K: k the kfolders values
    :param: BATCH: batch size
    :param: INPUT_SIZE: input dimensions, height and width, default=(224,224)
    :param: SPLIT_VALID: portion to divide the training base into training and validation
    return: train and valid dataset
    """
    train_dir = PATH_BD + '/Train/k' + str(K)
    print('train_dir ', train_dir)
    
    idg = ImageDataGenerator(
        width_shift_range=0.1,      # Randomly shifts the image horizontally by up to 10% of its width.
        height_shift_range=0.1,     # Randomly shifts the image vertically by up to 10% of its height.
        zoom_range=0.3,             # Applies a random zoom of up to 30%.
        fill_mode='nearest',        # Fills any new pixels created by shifts or zooms with the nearest pixel values.
        horizontal_flip=True,       # Randomly flips the image horizontally.
        rescale=1./255,             # Scales pixel values to a range of 0 to 1 by dividing by 255.
        validation_split=SPLIT_VALID) # Sets aside a portion of data for validation based on the provided split ratio.

    train_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='training')

    val_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='validation')

    return train_generator, val_generator

def load_data_ttv(PATH_BD, K, BATCH, INPUT_SIZE, SPLIT_VALID):
    """
    -->loading train data 
    :param: PATH_BD: file name 
    :param: K: k the kfolders values
    :param: BATCH: batch size
    :param: INPUT_SIZE: input dimensions, height and width, default=(224,224)
    :param: SPLIT_VALID: portion to divide the training base into training and validation
    return: train and valid dataset
    """
    train_dir = PATH_BD + '/Train/k' + str(K)
    val_dir = PATH_BD + '/Val/k' + str(K)
    print('train_dir ', train_dir)
    
    idg = ImageDataGenerator(rescale=1. / 255)

    train_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42)

    val_generator = idg.flow_from_directory(
        directory=val_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42)

    return train_generator, val_generator

def load_data_test(PATH_BD, K, BATCH, INPUT_SIZE):
    """
    -->loading train data 
    :param: PATH_BD: file name 
    :param: K: k the kfolders values
    :param: BATCH: batch size
    :param: INPUT_SIZE: input dimensions, height and width, default=(224,224)
    :return: test dataset
    """
    test_dir = PATH_BD + '/Test/k' + str(K)
    print('test_dir ', test_dir)

    idg = ImageDataGenerator(rescale=1. / 255)
    test_generator = idg.flow_from_directory(
        directory=test_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False,
        seed=42)
    return test_generator

def verifyGPU():
    print("##"*30)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("##"*30)
    print('\n')

def figure_size(CATEGORIES):
    global fig_size
    tam = len(CATEGORIES)
    if tam >20:
        w=100
        h=100
        fig_size = (w,h)
        print('fig_size ', fig_size)
    else:
        fig_size = (10, 8)
        print('fig_size ', fig_size)
    return fig_size
        
"""# Main"""

if __name__=="__main__":
  help(read_print_csv)
  help(add_row_csv)
  help(load_data_train)
  help(load_data_test)
