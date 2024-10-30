# Select models
from keras.applications import InceptionV3, ResNet152V2, Xception, VGG16, VGG19, DenseNet121, MobileNet, DenseNet201, NASNetLarge, InceptionResNetV2
from keras import models, layers, Model

import keras, sys
import tensorflow as tf
from keras.models import Sequential, Model
from keras import models, layers
from keras.layers import Dropout, Dense, Input
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Dropout
from keras.layers import Conv2D, GlobalAveragePooling2D
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
    for i, layer in enumerate(conv_model.layers):
        print("{0} {1}:\t{2}".format(i, layer.trainable, layer.name))
        layers_arr.append([i,layer.trainable, layer.name])
    if not(save_dir==''):
        utils_lib.add_row_csv(_csv_layers, layers_arr)

def hyper_model(rows, num_classes):
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
    #base_model.summary()
        
    # Exclude the last 9 layers of the model.
    conv = base_model.layers[-2].output
       
    output=Dense(num_classes, name='predictions', activation=rows['last_activation'])(conv)
    fine_model = Model(inputs=base_model.input, outputs=output)
    
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
    fine_model.summary()
    return fine_model