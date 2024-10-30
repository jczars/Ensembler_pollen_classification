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

from keras.applications import InceptionV3, ResNet152V2, Xception, VGG16, VGG19, DenseNet121, MobileNet, DenseNet201, NASNetLarge, InceptionResNetV2

sys.path.append('/home/jczars/anaconda3')
from Library import utils_lib




def print_layer(conv_model, layers_params):
    save_dir=layers_params['save_dir']
    nm_model=layers_params['model']
    if not(save_dir==''):
        _csv_layers=save_dir+nm_model+'_layers.csv'
        utils_lib.add_row_csv(_csv_layers, [['id_test',layers_params['id_test']]])
        
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
    if rows['model']=='VGG16':
        model=hyper_VGG16PSA(rows, num_classes)
    if rows['model']=='VGG19':
        model=hyper_VGG19PSA(rows, num_classes)
    
    return model

def hyper_VGG16PSA(rows, num_classes):
    model=eval(rows['model'])
    bl=rows['block']
    save_dir=rows['root']
    learning_rate=rows['learning_rate']
    img_size=rows['img_size']
    input_shape=(img_size, img_size, 3)
    # pretreined
    image_input = Input(shape=input_shape)
    base_model = model(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None)
    #Congelar camadas convolucionais
    for layer in base_model.layers:
        layer.trainable = False
    base_model.summary()
    
    # Exclude the last 5 layers of the model.
    conv = base_model.layers[-9].output

    #Soft Attention Layer
    attention_layer,map2 = SoftAttention(aggregate=True,m=16,concat_with_x=False,
                                         ch=int(conv.shape[-1]),
                                         name='soft_attention')(conv)
    attention_layer=(MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer))
    fine_model=(MaxPooling2D(pool_size=(2, 2),padding="same")(conv))
    
    fine_model = concatenate([fine_model,attention_layer])
    fine_model=Activation("relu")(fine_model)
    fine_model= Dropout(rows['dropout'])(fine_model)
        
    fine_model=(Conv2D(filters=512,kernel_size=(3,3), activation="relu",
                       padding="same",kernel_initializer='he_normal')(fine_model))
    fine_model=(BatchNormalization()(fine_model))
    fine_model=(Conv2D(filters=512,kernel_size=(3,3), activation="relu",
                       padding="same",kernel_initializer='he_normal')(fine_model))
    fine_model=(BatchNormalization()(fine_model))
    fine_model=(Conv2D(filters=512,kernel_size=(3,3), activation="relu",
                       padding="same",kernel_initializer='he_normal')(fine_model))
    fine_model=(BatchNormalization()(fine_model))
    
    if rows['model']=='VGG19':
        fine_model=(Conv2D(filters=512, name='conv_end',kernel_size=(3,3), activation="relu",
                       padding="same",kernel_initializer='he_normal')(fine_model))
        fine_model=(BatchNormalization()(fine_model))
        
    
    fine_model=(MaxPooling2D(pool_size=(4, 4),padding="same")(fine_model))
    
    if bl==0:
        # Fine bl0
        fine_model=BatchNormalization()(fine_model)
        fine_model=Dense(rows['dense_size'], activation=rows['activation'])(fine_model)
        fine_model=GlobalAveragePooling2D()(fine_model)

    
    if bl==3:
        # Fine bl3 only VGG16
        fine_model=Flatten()(fine_model)
        fine_model=Dense(rows['dense_size'], name='fc1', activation=rows['activation'])(fine_model)
        fine_model=Dense(rows['dense_size'], name='fc2', activation=rows['activation'])(fine_model)
    
    
    fine_model=Dense(num_classes, activation=rows['last_activation'])(fine_model)
        
    #model.summary()
    #utils_b0.print_layer0(fine_model)

    model = Model(inputs=base_model.input, outputs=fine_model)
    model.summary()
    
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
    
    model.compile(loss="categorical_crossentropy",
                       optimizer=opt,
                       metrics=["accuracy"])
    layers_params={'id_test':rows['id_test'],'save_dir':save_dir, 'model':rows['model']}
    print_layer(base_model, layers_params)
    return model

def hyper_VGG19PSA(rows, num_classes):
    model=eval(rows['model'])
    bl=rows['block']
    save_dir=rows['root']
    learning_rate=rows['learning_rate']
    img_size=rows['img_size']
    input_shape=(img_size, img_size, 3)
    # pretreined
    image_input = Input(shape=input_shape)
    base_model = model(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None)
    #Congelar camadas convolucionais
    for layer in base_model.layers:
        layer.trainable = False
    base_model.summary()
    
    # Exclude the last 5 layers of the model.
    conv = base_model.layers[-9].output

    #Soft Attention Layer
    attention_layer,map2 = SoftAttention(aggregate=True,m=16,concat_with_x=False,
                                         ch=int(conv.shape[-1]),
                                         name='soft_attention')(conv)
    attention_layer=(MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer))
    fine_model=(MaxPooling2D(pool_size=(2, 2),padding="same")(conv))
    
    fine_model = concatenate([fine_model,attention_layer])
    fine_model=Activation("relu")(fine_model)
    fine_model= Dropout(rows['dropout'])(fine_model)
        
    fine_model=(Conv2D(filters=512,kernel_size=(3,3), activation="relu",
                       padding="same",kernel_initializer='he_normal')(fine_model))
    fine_model=(BatchNormalization()(fine_model))
    fine_model=(Conv2D(filters=512,kernel_size=(3,3), activation="relu",
                       padding="same",kernel_initializer='he_normal')(fine_model))
    fine_model=(BatchNormalization()(fine_model))
    fine_model=(Conv2D(filters=512,kernel_size=(3,3), activation="relu",
                       padding="same",kernel_initializer='he_normal')(fine_model))
    fine_model=(BatchNormalization()(fine_model))
    
    if rows['model']=='VGG19':
        fine_model=(Conv2D(filters=512, name='conv_end',kernel_size=(3,3), activation="relu",
                       padding="same",kernel_initializer='he_normal')(fine_model))
        fine_model=(BatchNormalization()(fine_model))
        
    
    fine_model=(MaxPooling2D(pool_size=(4, 4),padding="same")(fine_model))
    
    if bl==0:
        # Fine bl0
        fine_model=BatchNormalization()(fine_model)
        fine_model=Dense(rows['dense_size'], activation=rows['activation'])(fine_model)
        fine_model=GlobalAveragePooling2D()(fine_model)

    
    if bl==3:
        # Fine bl3 only VGG16
        fine_model=Flatten()(fine_model)
        fine_model=Dense(rows['dense_size'], name='fc1', activation=rows['activation'])(fine_model)
        fine_model=Dense(rows['dense_size'], name='fc2', activation=rows['activation'])(fine_model)
    
    
    fine_model=Dense(num_classes, activation=rows['last_activation'])(fine_model)
        
    #model.summary()
    #utils_b0.print_layer0(fine_model)

    model = Model(inputs=base_model.input, outputs=fine_model)
    model.summary()
    
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
    
    model.compile(loss="categorical_crossentropy",
                       optimizer=opt,
                       metrics=["accuracy"])
    layers_params={'id_test':rows['id_test'],'save_dir':save_dir, 'model':rows['model']}
    print_layer(base_model, layers_params)
    return model