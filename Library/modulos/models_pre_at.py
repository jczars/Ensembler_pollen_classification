#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 07:32:53 2024

@author: jczars
"""

# Select models
from keras.models import Sequential, Model
from keras import models, layers
from keras import backend as K
from keras.layers import Layer, MaxPooling2D, concatenate, Activation
import keras.layers as kl
from keras.layers import Dropout, Dense, Input
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Dropout
from keras.layers import Conv2D, GlobalAveragePooling2D
import keras, sys

sys.path.append('/home/jczars/anaconda3')
from Library import utils_lib




def print_layer(conv_model, layers_params):
    save_dir=layers_params['save_dir']
    nm_model=layers_params['model']
    if not(save_dir==''):
        _csv_layers=save_dir+nm_model+'_layers.csv'
        utils_lib.add_row_csv(_csv_layers, [['id_test',layers_params['id_test']]])
        utils_lib.add_row_csv(_csv_layers, [['freeze',layers_params['freeze'],
                                             'depth',layers_params['depth'], 
                                             'percentil',layers_params['percentil']]])
        utils_lib.add_row_csv(_csv_layers, [['model',layers_params['model']]])
        
        utils_lib.add_row_csv(_csv_layers, [['trainable','name']])
    layers_arr=[]
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))
        layers_arr.append([layer.trainable, layer.name])
    if not(save_dir==''):
        utils_lib.add_row_csv(_csv_layers, layers_arr)

# Soft Attention
class SoftAttention(Layer):
    def __init__(self,ch,m,concat_with_x=False,aggregate=False,**kwargs):
        self.channels=int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x


        super(SoftAttention,self).__init__(**kwargs)

    def build(self,input_shape):

        self.i_shape = input_shape

        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads) # DHWC

        self.out_attention_maps_shape = input_shape[0:1]+(self.multiheads,)+input_shape[1:-1]

        if self.aggregate_channels==False:

            self.out_features_shape = input_shape[:-1]+(input_shape[-1]+(input_shape[-1]*self.multiheads),)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1]+(input_shape[-1]*2,)
            else:
                self.out_features_shape = input_shape


        self.kernel_conv3d = self.add_weight(shape=kernel_shape_conv3d,
                                        initializer='he_uniform',
                                        name='kernel_conv3d')
        self.bias_conv3d = self.add_weight(shape=(self.multiheads,),
                                      initializer='zeros',
                                      name='bias_conv3d')

        super(SoftAttention, self).build(input_shape)

    def call(self, x):

        exp_x = K.expand_dims(x,axis=-1)

        c3d = K.conv3d(exp_x,
                     kernel=self.kernel_conv3d,
                     strides=(1,1,self.i_shape[-1]), padding='same', data_format='channels_last')
        conv3d = K.bias_add(c3d,
                        self.bias_conv3d)
        conv3d = kl.Activation('relu')(conv3d)

        conv3d = K.permute_dimensions(conv3d,pattern=(0,4,1,2,3))


        conv3d = K.squeeze(conv3d, axis=-1)
        conv3d = K.reshape(conv3d,shape=(-1, self.multiheads ,self.i_shape[1]*self.i_shape[2]))

        softmax_alpha = K.softmax(conv3d, axis=-1)
        softmax_alpha = kl.Reshape(target_shape=(self.multiheads, self.i_shape[1],self.i_shape[2]))(softmax_alpha)


        if self.aggregate_channels==False:
            exp_softmax_alpha = K.expand_dims(softmax_alpha, axis=-1)
            exp_softmax_alpha = K.permute_dimensions(exp_softmax_alpha,pattern=(0,2,3,1,4))

            x_exp = K.expand_dims(x,axis=-2)

            u = kl.Multiply()([exp_softmax_alpha, x_exp])

            u = kl.Reshape(target_shape=(self.i_shape[1],self.i_shape[2],u.shape[-1]*u.shape[-2]))(u)

        else:
            exp_softmax_alpha = K.permute_dimensions(softmax_alpha,pattern=(0,2,3,1))

            exp_softmax_alpha = K.sum(exp_softmax_alpha,axis=-1)

            exp_softmax_alpha = K.expand_dims(exp_softmax_alpha, axis=-1)

            u = kl.Multiply()([exp_softmax_alpha, x])

        if self.concat_input_with_scaled:
            o = kl.Concatenate(axis=-1)([u,x])
        else:
            o = u

        return [o, softmax_alpha]

    def compute_output_shape(self, input_shape):
        return [self.out_features_shape, self.out_attention_maps_shape]


    def get_config(self):
        return super(SoftAttention,self).get_config()

def hyper_model(rows, num_classes):
    model=eval(rows['model'])
    bl=rows['block']
    save_dir=rows['root']
    learning_rate=rows['learning_rate']
    img_size=rows['img_size']
    input_shape=(img_size, img_size, 3)
    # pretreined
    image_input = Input(shape=input_shape)
    base_model = model(input_tensor=image_input,
                        include_top=False,
                        weights='imagenet')
    #base_model.summary()
    fine_model = models.Sequential()
    fine_model.add(base_model)
    
    if bl==0:
        # Fine bl0
        fine_model.add(BatchNormalization())
        fine_model.add(Dense(rows['dense_size'], activation=rows['activation']))
        fine_model.add(GlobalAveragePooling2D())
        
    if bl==1:        
        # Fine bl1
        fine_model.add(GlobalAveragePooling2D())
        fine_model.add(BatchNormalization())
        fine_model.add(Flatten(name="flatten"))
        
        fine_model.add(Dense(rows['dense_size'], activation=rows['activation']))
        fine_model.add(Dropout(rows['dropout']))
        fine_model.add(BatchNormalization())
    if bl==2:        
        # Fine bl2
        model.add(GlobalAveragePooling2D())
        model.add(BatchNormalization())
        model.add(Flatten())
    
        model.add(Dense(rows['dense_size'], activation=rows['activation']))
        model.add(Dropout(rows['dropout']))
        model.add(BatchNormalization())
        
    fine_model.add(Dense(num_classes, name='predictions', activation=rows['last_activation']))
    
    # freeze (Fine)
    freeze = rows['freeze']
    print("\n[INFO] build arcteture freeze...")
    
    for layer in base_model.layers:
        layer.trainable = False
    depth=len(base_model.layers)
    percentil=round(freeze/depth*100, 2)
    print(rows['model'], 'depth ', depth, 'freeze', freeze, 'freeze percentil %', percentil)

    if freeze != depth:
        for layer in base_model.layers[freeze:]:
            layer.trainable = True
    layers_params={'id_test':rows['id_test'],'save_dir':save_dir, 'model':rows['model'], 
                   'freeze':freeze, 'depth':depth, 'percentil':percentil}
    print_layer(base_model, layers_params)
    
    # otimizador
    opt=rows['optimizer']
    if opt == 'Adam':
      opt = keras.optimizers.Adam(learning_rate=learning_rate)
    if opt == 'RMSprop':
      opt=keras.optimizers.RMSprop(learning_rate=learning_rate)
    if opt == 'Adagrad':
      opt = keras.optimizers.Adagrad(learning_rate=learning_rate)
    if opt == 'SGD':
      opt = keras.optimizers.SGD(learning_rate=learning_rate)
    print(opt) 
    
    fine_model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
    #fine_model.summary()
    return fine_model