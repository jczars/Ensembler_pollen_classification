#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:57:13 2024

@author: jczars
"""

import sys,os, gc
sys.path.append('/home/jczars/anaconda3')
print(sys.path)
from Library import models_cnn_lib, reports_lib, utils_lib
from keras import models
import datetime

def sel_base(_base):
    print('\n')
    print("#"*30)
    print('Seleção da base')
    print("#"*30)
    print(_base)
    if _base == 'EQUATORIAL_R':
        _path_data = "./BD/CPD1_Dn_VTcr_220824/EQUATORIAL_R"
        _path_cat = _path_data+"/Train/k1"
        CATEGORIES = sorted(os.listdir(_path_cat))
    if _base == 'POLAR_R':
        _path_data = "./BD/CPD1_Dn_VTcr_220824/POLAR_R"
        _path_cat = _path_data+"/Train/k1"
        CATEGORIES = sorted(os.listdir(_path_cat))

    print(CATEGORIES)
    print(len(CATEGORIES))
    return CATEGORIES, _path_data

"""## train_kfolds"""


def train_kfolds(_k, save_dir_test, PATH_BD, CATEGORIES, model):
    global _p, batch_size, input_size, fig_size
    ME = []
    print('\n')
    print("#"*30)
    print("2.1 - criar metrics_teste_a.csv")
    print("#"*30)
    _csv_me_test_a = save_dir_test + 'metrics_test_'+str(_k)+str(_p)+'.csv'
    columns=[['k', 'val_loss', 'val_acc', 'accuracy', 'precision', 'recall', 'fscore', 'kappa']]
    utils_lib.add_row_csv(_csv_me_test_a, columns)
    print(columns)
    
    for i in range(_p):
        k = _k+i
        str_time = datetime.datetime.now().replace(microsecond=0)
        # load data
        train, val, test = None, None, None
        print("%"*90)
        print('PATH_BD ', PATH_BD, 'k ', k)
        
        print("2.2 - load data train")
        train, val = utils_lib.load_data_train(PATH_BD, K=_K, BATCH=batch_size, INPUT_SIZE=input_size, SPLIT_VALID=0.2)
        
        # instanciando o modelo
        print("2.3 - instanciar o modelo")
        model_inst = None
        model_inst = models_cnn_lib.sel_cnn(model, input_shape, CATEGORIES)

        # Train
        print("2.4 - train instance")
        model_fit = models_cnn_lib.fitModels(save_dir_test, k, model_inst, 
                                             train, val, test, epochs, alpha, 
                                             CATEGORIES, input_shape, verbose=2)
        
        # Relatórios
        
        print("\n2.5 - Relatórios--------")
        test=utils_lib.load_data_test(PATH_BD, _k, batch_size, input_size)
        
        
        path_model = save_dir_test + 'bestLoss_' + models_cnn_lib.get_model_name(k)
        restored_model = models.load_model(path_model)
        y_true, y_pred, df_corre = reports_lib.predict_data_generatorV1(test, 
                                                                   batch_size, 
                                                                   #model_fit, 
                                                                   restored_model,
                                                                   save_dir_test, 
                                                                   k,
                                                                   CATEGORIES)
        
        reports_lib.plot_confusion_matrixV1(y_true, y_pred, CATEGORIES, 
                                            save_dir_test,  k, verbose=0)

        me = reports_lib.metricasV1(y_true, y_pred)
        reports_lib.class_reportsV1(y_true, y_pred, CATEGORIES, 
                                  save_dir_test, k)

        Eval_valid = model_fit.evaluate(val)
        end_time = datetime.datetime.now().replace(microsecond=0)
        delay=end_time-str_time
        
        prec=4
        mes = round(Eval_valid[0], prec), round(Eval_valid[1], prec), me[0], me[1], me[2], me[3], me[4], str_time, end_time, delay
        print(mes)
        
        data=[[str(k), mes[0], mes[1], mes[2], mes[3], mes[4], mes[5], mes[6], mes[7], mes[8], mes[9]]]
        print(data)
        utils_lib.add_row_csv(_csv_me_test_a, data)

        ME.append(mes)
        
        #boxplot 16/04/24 23:55
        reports_lib.boxplot(save_dir_test, k, df_corre)
        
        print("------fim relatórios--------\n")
        
        #Limpar memória
        model_fit = None
        restored_model=None
        y_true, y_pred, df_corre = None, None, None
        #me, Eval_valid, mes, data, ME = None, None, None, None, None
        gc.collect()



"""## lerCnf"""


def lerCnf(PATH_ROOT, MODELS, BASES, _K):
    # criar a pasta raiz
    print('\n')
    print("#"*60)
    print("1.0 - Módulo config")
    print("1.1 - criar pasta raiz")
    print("#"*60)
    utils_lib.create_folders(PATH_ROOT, flag=0)
    nm = 0
    
    for base in BASES:
        print(base)
        # caminho da base
        print("#"*60)
        print("1.2 - selecionar a base")
        print("#"*60)
        CATEGORIES, PATH_BD = sel_base(base)

        for model in MODELS:
            print(nm, model, base)

            # criar a pasta do teste
            print("#"*60)
            print("1.3 - criar pasta do teste")
            print("#"*60)
            nmTeste = str(nm) + '_' + model + '_' + base + '_' + str(_K)
            save_dir_test = PATH_ROOT + '/' + nmTeste + '/'
            utils_lib.create_folders(save_dir_test, flag=0)
            print(nmTeste)
            
            _csv_head = save_dir_test + 'head_'+str(_K)+str(_p)+'.csv'
            #modelo, otimizador, lr, batch_size, img_size, kfolder, Base
            data=[["modelo", "otimizador", "lr", "batch_size", "img_size", "kfolder", "Base"],
                [model, 'Adam', '1e-4', str(batch_size), str(img_size), str(_K)+'_'+str(_p), base]]
            print(data)
            utils_lib.add_row_csv(_csv_head, data)

            train_kfolds(_K, save_dir_test, PATH_BD, CATEGORIES, model)
            nm = nm + 1
            # break
        # break
    return        

"""## Teste"""

if __name__ == "__main__":
    # Sets the working directory
    os.chdir('/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/')
    
    PATH_ROOT = "./1_Fine_tuned/CLASS_TUNED/class_CPD1_k10SEM_2410240732"
    MODELS = ['DenseNet201', 'MobileNet']
    BASES = ['POLAR_R'] #'EQUATORIAL_R'
    
    # Constantes
    alpha = 1e-5
    cont = 1
    img_size = 224
    input_size=(img_size,img_size)
    input_shape = (img_size, img_size, 3)
    batch_size = 16
    epochs = 500
    
    #Variáveis
    _K = 1
    _p = 10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
      
    utils_lib.verifyGPU()
    lerCnf(PATH_ROOT, MODELS, BASES, _K)