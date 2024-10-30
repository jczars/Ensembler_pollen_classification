# 03 - fit
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf

# 03 - fit

def get_model_name(nm_modelo, k):
    return nm_modelo+'_bestLoss_' + str(k) + '.keras'

def callbacks(save_dir,nm_modelo, k, alpha):
    cp_loss = ModelCheckpoint(save_dir + get_model_name(nm_modelo, k),
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=True,
                              mode='min')
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       patience = 5,
                       restore_best_weights = True,
                       verbose=1)
    lr = ReduceLROnPlateau(monitor='val_loss',
                           factor=0.1,
                           min_delta=alpha,
                           patience=5,
                           verbose=1)
    callbacks_list = [cp_loss, es, lr]
    return callbacks_list

def fitModels(conf, 
              model_inst,
              train_data_generator, valid_data_generator,
              verbose=2):

  save_dir=conf['save_dir']
  epochs=conf['epochs']
  alpha=conf['alpha']
  k=conf['k']
  #input_shape=conf['input_shape']
  nm_modelo=conf['model']

  print(save_dir+'bestLoss_'+str(k)+'.h5')
  callbacks_list=callbacks(save_dir,nm_modelo, k, alpha)

  str_time = datetime.datetime.now().replace(microsecond=0)

  history = model_inst.fit(train_data_generator,
              epochs=epochs,
              callbacks=callbacks_list,
              validation_data=valid_data_generator)
  end_time = datetime.datetime.now().replace(microsecond=0)
  delay=end_time-str_time
  print('Training time: %s' % (delay))

  #4-plot Accuracy
  pd_history = pd.DataFrame(history.history)
  pd_history.plot()
  plt.grid(True)
  if verbose==2:
    plt.savefig(save_dir+str(k)+'_acc_loss.jpg')
  #plt.show()

  num_eapoch=len(history.history['loss'])

  return model_inst, num_eapoch, str_time, end_time, delay

def fitVitModels(conf, 
              model_inst,
              train_data_generator, valid_data_generator,
              verbose=2):

  save_dir=conf['save_dir']
  epochs=conf['epochs']
  alpha=conf['alpha']
  k=conf['k']
  nm_modelo=conf['model']

  
  callbacks_list=callbacks(save_dir,nm_modelo, k, alpha)

  str_time = datetime.datetime.now().replace(microsecond=0)

  history = model_inst.fit(train_data_generator,
              epochs=epochs,
              callbacks=callbacks_list,
              validation_data=valid_data_generator)
  end_time = datetime.datetime.now().replace(microsecond=0)
  delay=end_time-str_time
  print('Training time: %s' % (delay))

  #4-plot Accuracy
  pd_history = pd.DataFrame(history.history)
  pd_history.plot()
  plt.grid(True)
  if verbose==2:
    plt.savefig(save_dir+str(k)+'_acc_loss.jpg')
  #plt.show()

  num_eapoch=len(history.history['loss'])
  
  """
  Font: https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2023-08-11-tensorflow-i-know-flowers-vit/2023-08-11/
  dia 16/04/24 as 22:55
  Foi alterado:
      * reports_lib.predict_gen_conf
      * saving the model
      * restore the model
      
  """
  # saving the model
  nome_model= save_dir+nm_modelo+'_bestLoss_'+str(k)
  tf.keras.saving.save_model(model_inst, 
                             nome_model, 
                             overwrite=True, 
                             save_format='tf')
  # restore the model
  #restored_model = tf.keras.saving.load_model(nome_model)

  return model_inst, num_eapoch, str_time, end_time, delay

