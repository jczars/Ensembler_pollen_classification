# -*- coding: utf-8 -*-
"""ens_PL_MbDnIn_Bal_CI_srv_080823.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d0wuGOT_Dd8e51YBf1Jzxu0AWXaQWbck

# import
"""

servidor=True
if not(servidor):
  from google.colab import drive
  drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os, time
from datetime import datetime

#Model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet201
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras import models, layers, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

#Graphs
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

import sys
if not(servidor):
  sys.path.append('/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification')
  print(sys.path)
  
import bib_functions as func
import bib_relatorios as rel

def load_datak(PATH_BD, _k, input_size=(224,224)):

  test_dir=PATH_BD+'/Test/k'+str(_k)
  print('test_dir ', test_dir)

  idg = ImageDataGenerator(rescale=1./255, validation_split=0.2)
  test_generator = idg.flow_from_directory(
    directory=test_dir,
    target_size=input_size,
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    seed=42)

  return test_generator

"""# Load models"""

def ensemble_votoMajor(path_model, k, CATEGORIES, X_test,y_test, save_dir):
  model_1 = load_model(path_model + '/0_DenseNet201_BD_ALL_10/bestLoss_model_' + str(k) + '.h5')
  model_2 = load_model(path_model + '/1_MobileNet_BD_ALL_10/bestLoss_model_' + str(k) + '.h5')
  

  models=[model_1, model_2]
  max, conf=func.gabarito(models, X_test,y_test)
  vencedores=func.contagem_votos(max, conf, y_test)

  y_vot = [CATEGORIES[y] for y in vencedores] # convert
  y_vot=np.array(y_vot)

  return y_vot

def csv_Test(_path_Test, _k):
  path=_path_Test+'/csv_a'
  func.create_folders(path, flag=0)

  for i in range(_k):
    k=i+1
    dataSetTest=_path_Test+'/Test/k'+str(k)
    _csv_data=path+'/'+_base+'_testSet_k'+str(k)+'.csv'
    func.create_dataSet(dataSetTest, _csv_data,  CATEGORIES)
  return path

def header(_csv_header, ALGORITMO, _path_data, _path_model, _altera):
  #Cabeçalho do teste
  header=pd.DataFrame(columns= ['Algoritmo','dataTrain','Modelo', 'altera' ])
  header.loc[0]=ALGORITMO, _path_data, _path_model, _altera
  header.to_csv(_csv_header, index = False, header=True)

"""# Reports"""

def performance_report_pd(arr, classes):
  mat_df=pd.DataFrame(columns= ['Classes','Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support'])
  cr = dict()
  # col=number of class
  col=len(arr)
  support_sum = 0
  for i in range(col):
    vertical_sum= sum([arr[j][i] for j in range(col)])
    horizontal_sum= sum(arr[i])
    #https://stats.stackexchange.com/questions/312780/why-is-accuracy-not-the-best-measure-for-assessing-classification-models
    a = round(arr[i][i] / horizontal_sum, 2)
    p = round(arr[i][i] / vertical_sum, 2)
    r = round(arr[i][i] / horizontal_sum, 2)
    f = round((2 * p * r) / (p + r), 2)
    s = horizontal_sum
    row=[classes[i],a,p,r,f,s]
    support_sum+=s
    cr[i]=row
    mat_df.loc[i]=row
  return mat_df

"""# Matrix"""

def plot_confusion_matrix(y_test, y_pred, CATEGORIES, save_dir='', fold_var=0, normalize=None):
  """
  --> Confusion Matrix, print or save
  :param: y_test: images from Image Date Generator
  :param: y_pred: labels predict
  :param: CATEGORIES: list classes the of images
  :param: normalize: 'None', 'pred'
  :return: vector of the metrics
  """
  global figure_size
  
  print('Confusion Matrix')
  mat =confusion_matrix(y_test, y_pred, normalize=normalize)
  
  #Save matrix 21/01/24
  print('salvando matriz como csv em ', save_dir)
  df_mat = pd.DataFrame(mat)
  df_mat.to_csv(save_dir+'/mat_conf_'+str(fold_var)+'.csv')
  
  plt.figure(figsize = figure_size)
  ax= plt.subplot()
  sns.set(font_scale=0.9)  # Adjust to fit
  sns.heatmap(mat, cmap="Blues",annot=True, fmt="g"); #annot=True to annotate cells

  ax.set_xticklabels(ax.get_xticklabels(),
                      rotation=90, horizontalalignment='right')
  ax.set_yticklabels(ax.get_yticklabels(),
                      rotation=0, horizontalalignment='right')

  # labels, title and ticks
  ax.set_xlabel('Rótulos previstos');
  ax.set_ylabel('Rótulos verdadeiros');
  ax.set_title('Matriz de confusão');
  ax.xaxis.set_ticklabels(CATEGORIES);
  ax.yaxis.set_ticklabels(CATEGORIES);

  if not(save_dir==''):
    plt.savefig(save_dir+'/mat_conf_'+str(fold_var)+'.jpg')
  if not(servidor):
    plt.show()
  return mat

def plt_mats(mats, save_dir='', fold_var=0):
  global figure_size
  
  #Save matrix 21/01/24
  print('salvando matriz como csv em ', save_dir)
  df_mat = pd.DataFrame(mats)
  df_mat.to_csv(save_dir+'/Geral_mat_conf_'+str(fold_var)+'.csv')
  
  plt.figure(figsize = figure_size)
  ax= plt.subplot()
  sns.set(font_scale=0.9)  # Adjust to fit
  sns.heatmap(mats, cmap="Blues",annot=True, fmt=".0f"); #annot=True to annotate cells

  ax.set_xticklabels(ax.get_xticklabels(),
                      rotation=90, horizontalalignment='right')
  ax.set_yticklabels(ax.get_yticklabels(),
                      rotation=0, horizontalalignment='right')

  # labels, title and ticks
  ax.set_xlabel('Rótulos previstos');
  ax.set_ylabel('Rótulos verdadeiros');
  ax.set_title('Matriz de confusão Geral');
  ax.xaxis.set_ticklabels(CATEGORIES);
  ax.yaxis.set_ticklabels(CATEGORIES);

  if not(save_dir==''):
    plt.savefig(save_dir+'/Geral_mat_conf_'+str(fold_var)+'.jpg')
  if not(servidor):
    plt.show()

"""## plus Matrix"""

def somar(m1, m2):
    matriz_soma = []
    # Supondo que as duas matrizes possuem o mesmo tamanho
    quant_linhas = len(m1) # Conta quantas linhas existem
    quant_colunas = len(m1[0]) # Conta quantos elementos têm em uma linha
    for i in range(quant_linhas):
        # Cria uma nova linha na matriz_soma
        matriz_soma.append([])
        for j in range(quant_colunas):
            # Somando os elementos que possuem o mesmo índice
            soma = m1[i][j] + m2[i][j]
            matriz_soma[i].append(soma)
    return matriz_soma

"""# run"""

if __name__=="__main__":
  ALGORITMO = '3_ens_eq_mbdn_BD_ALL_K10B790_150124.py'
  _base = 'BD_ALL0'
  _path_BASE = r"C:\Users\jczarsGamer\anaconda3\QUALIFICA\BD_JOIN\BD_ALL0_R"
  _path_MODEL = r"C:\Users\jczarsGamer\anaconda3\QUALIFICA\BD_JOIN_COMMITE\rel_pro_BD_ALL_K10B200_301223"
  _path_CATEGORIES = r"C:\Users\jczarsGamer\anaconda3\QUALIFICA\BD_JOIN\BD_ALL0_R/Test/k1"
  save_dir = r"C:\Users\jczarsGamer\anaconda3\QUALIFICA\BD_JOIN_COMMITE\BDALL0_k10b200_MbDn_210124"
  _altera = 'aplicando comitê em BD_ALL em 15/01/24'

  _k = 10
  start_time = time.time()
  img_size=224
  figure_size=(30,25)
  input_shape  = (img_size, img_size, 3)
  func.create_folders(save_dir, flag=0) #criar a pasta

  _csv_header = save_dir + '/header.csv'
  header(_csv_header, ALGORITMO, _path_BASE, _path_MODEL, _altera)
  CATEGORIES = sorted(os.listdir(_path_CATEGORIES))

  #criar os csv
  _path_csv=csv_Test(_path_BASE, _k)

  ME=[]
  for i in range(_k):
    k=i+1
    # load data
    _csv_test=_path_csv+'/'+_base+'_testSet_k'+str(_k)+'.csv'
    test_data=pd.read_csv(_csv_test)
    X_test,y_test=func.load_img(test_data, input_shape)

    y_vot=ensemble_votoMajor(_path_MODEL, k, CATEGORIES, X_test,y_test, save_dir)

    #reports
    plot_confusion_matrix(y_test, y_vot, CATEGORIES, save_dir, k, normalize=None)

    #rel.plot_mat_confux(y_test, y_vot, k, CATEGORIES, save_dir, k, fig_size=(15,15))
    #build 02/08/2023
    mat=plot_confusion_matrix(y_test, y_vot, CATEGORIES, save_dir, k, normalize=None)
    if k == 1:
      mats=mat
    else:
      mats=somar(mats, mat)

    me=rel.metricas(y_test, y_vot)
    rel.filter_wrong(test_data, y_test, y_vot, save_dir, k)

    # build 03/08/23
    rep=performance_report_pd(mat, CATEGORIES)
    print(rep)
    rel.repor_class(y_test, y_vot, save_dir, k)
    ME.append(me)


  ### save metricas
  _csv_me_test= save_dir+'/metrics_test.csv'
  metrics_test=pd.DataFrame(columns= ['k', 'accuracy','precision', 'recall', 'fscore', 'kappa' ])
  for i in range(int(_k)):
      metrics_test.loc[i]=i, ME[i][0], ME[i][1], ME[i][2], ME[i][3], ME[i][4]
  metrics_test.loc['Mean']=round(metrics_test.mean(),4)
  metrics_test.loc['Std']=round(metrics_test.std(),4)

  metrics_test.to_csv(_csv_me_test, index = False, header=True)
  print(metrics_test)

  print('\nGeneral matrix')
  plt_mats(mats, save_dir, k,)
  print('\nGeneral reports')
  _csv_reg = save_dir + '/report_general_test.csv'
  reg = performance_report_pd(mats, CATEGORIES)
  reg.to_csv(_csv_reg, index=False, header=True)
  print(reg, '\n')