# 03 - fit
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from keras.utils import custom_object_scope

# 03 - fit

def get_model_name(nm_modelo, k):
    return nm_modelo+'_bestLoss_' + str(k) + '.keras'

def callbacks(save_dir, nm_modelo, k, alpha):
    cp_loss = ModelCheckpoint(save_dir + get_model_name(nm_modelo, k),
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min')
    es = EarlyStopping(monitor='val_loss',
               mode='min',
               patience = 10,
               verbose=1)
    lr = ReduceLROnPlateau(monitor='val_loss',
                   factor=0.1,
                   min_delta=alpha,
                   patience=5,
                   verbose=1)
    callbacks_list = [cp_loss, es]
    return callbacks_list

def callbacks_vit():
    cp_loss = EarlyStopping(
        monitor='val_loss',            # Monitor the validation loss.
        patience=10,                   # Stop training after 'patience' epochs with no improvement.
        verbose=1,                     # Print messages when stopping early.
        restore_best_weights=True,     # Restore model weights from the epoch with the best validation loss.
        mode='min'                     # Stop training when 'val_loss' has stopped decreasing.
        )
    callbacks_list = [cp_loss]
    return callbacks_list

def run_train(train, val, input_size, rows,nm_model,
              model_fine, k, config):

    print('batch_size ', rows['batch_size'])
    callbacks_list=callbacks(config['path_test'], nm_model, k, config['alpha'])
    str_time = datetime.datetime.now().replace(microsecond=0)

    with tf.device('/device:GPU:0'):
        print('\n',str_time)
        hist = model_fine.fit(train,
                              batch_size=rows['batch_size'],
                              epochs=config['epochs'],
                              callbacks=callbacks_list,
                              verbose=1,
                              validation_data=val)
    end_time = datetime.datetime.now().replace(microsecond=0)
    delay=end_time-str_time
    print('Training time: %s' % (delay))
    
    # 4-plot Accuracy
    pd_history = pd.DataFrame(hist.history)
    pd_history.plot()
    plt.grid(True)
    plt.savefig(config['path_test'] + 'acc_loss_' +str(rows['id_test'])+'_'+nm_model+'_k'+ str(k) + '.jpg')
    plt.show()

    return hist, str_time, end_time, delay

def run_trainV1(train, val, input_size, rows,nm_model,
              model_fine, k, config):

    print('batch_size ', rows['batch_size'])
    callbacks_list=callbacks_vit()
    str_time = datetime.datetime.now().replace(microsecond=0)

    with tf.device('/device:GPU:0'):
        print('\n',str_time)
        hist = model_fine.fit(train,
                              batch_size=rows['batch_size'],
                              epochs=config['epochs'],
                              callbacks=callbacks_list,
                              verbose=1,
                              validation_data=val)
    end_time = datetime.datetime.now().replace(microsecond=0)
    delay=end_time-str_time
    print('Training time: %s' % (delay))
    
    # saving the model
    nome_model= config['path_test']+nm_model+'_bestLoss_'+str(k)+'.keras'
    model_fine.save(nome_model)
    
    # 4-plot Accuracy
    pd_history = pd.DataFrame(hist.history)
    pd_history.plot()
    plt.grid(True)
    plt.savefig(config['path_test'] + 'acc_loss_' +str(rows['id_test'])+'_'+nm_model+'_k'+ str(k) + '.jpg')
    plt.show()

    return hist, str_time, end_time, delay

def run_train_vit(train, val, input_size, rows,nm_model,
              model_fine, k, save_dir, config):

    print('batch_size ', rows['batch_size'])
    
    
    callbacks_list=callbacks_vit()
    str_time = datetime.datetime.now().replace(microsecond=0)

    with tf.device('/device:GPU:0'):
        print('\n',str_time)
        hist = model_fine.fit(train,
                              batch_size=rows['batch_size'],
                              epochs=config['epochs'],
                              callbacks=callbacks_list,
                              verbose=1,
                              validation_data=val)
    end_time = datetime.datetime.now().replace(microsecond=0)
    delay=end_time-str_time
    print('Training time: %s' % (delay))
    
    """
    Font: https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2023-08-11-tensorflow-i-know-flowers-vit/2023-08-11/
    dia 16/04/24 as 22:55
    Foi alterado:
        * reports_lib.predict_gen_conf
        * saving the model
        * restore the model
        
    """
    # saving the model
    nome_model= save_dir+nm_model+'_bestLoss_'+str(k)
    tf.keras.saving.save_model(model_fine, 
                               nome_model, 
                               overwrite=True, 
                               save_format='tf')
    # 4-plot Accuracy
    pd_history = pd.DataFrame(hist.history)
    pd_history.plot()
    plt.grid(True)
    plt.savefig(config['path_test'] + 'acc_loss_' +str(rows['id_test'])+'_'+nm_model+'_k'+ str(k) + '.jpg')
    plt.show()
    

    return hist, str_time, end_time, delay

def load_model_vit(path_model_vit):
    # Carregar o modelo com o otimizador registrado no escopo de objetos personalizados
    with custom_object_scope({'Addons>RectifiedAdam': tfa.optimizers.RectifiedAdam}):
        model_vit = tf.keras.models.load_model(path_model_vit)
        
    return model_vit

if __name__=="__main__":
    help(get_model_name)
    help(callbacks)
    help(run_train)
    help(run_train_vit)