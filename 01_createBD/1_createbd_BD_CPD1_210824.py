# -*- coding: utf-8 -*-
"""createBD_vistas_130723.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pq39QejtEIEcfAiVaSuMjTFRNKTeXz3S

# import
"""


import os
import numpy as np
import glob
import cv2
import shutil

# Modelos
from keras import models

import sys
sys.path.append('/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/')
from Library import utils_lib, bib_functions


"""Labellling"""

def labelling(CATEGORIES, CATEGORIES_labels, path_data, path_vistas, model):
    verbose = 3
    
    #Create BD
    equatorial_bd = path_vistas+'/EQUATORIAL'
    polar_bd = path_vistas+'/POLAR'
    bib_functions.create_folders(equatorial_bd, flag=0)
    bib_functions.create_folders(polar_bd, flag=0)
    
    #Labelling
    for cat in CATEGORIES:
        path = path_data+'/'+cat+'/*.png'
        images_path = glob.glob(path)
        print(cat, len(images_path))
    
        for img_path in images_path:
            img = cv2.imread(img_path)
            x = bib_functions.normalizatioin(img)
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)
            confianca = np.max(preds)
            pred = np.argmax(preds, axis=-1)
            vista = CATEGORIES_labels[pred[0]]
            vt = vista.split('_')[0]
            nm = img_path.split('/')[-1]
    
            #print(cat, img_path)
            #print(f'nome {nm}, predict {pred}, {vista}, {vt}, {confianca}')
            if vt == 'polar':
                path_cat = polar_bd+'/'+cat
                bib_functions.create_folders(path_cat, flag=0)
                img_dst = path_cat+'/'+nm
                print('polar ', img_dst)
                if verbose == 3:
                    shutil.copy(img_path, img_dst)
            if vt == 'equatorial':
                path_cat = equatorial_bd+'/'+cat
                bib_functions.create_folders(path_cat, flag=0)
                img_dst = path_cat+'/'+nm
                print('equatorial ', img_dst)
                if verbose == 3:
                    shutil.copy(img_path, img_dst)
                    
            #Save labelling in csv 
            _csv_labels = path_vistas + '/labels_.csv'
            data = [[img_path, vt, confianca]]
            print(data)
            utils_lib.add_row_csv(_csv_labels, data)
    
    #Quantizar as vistas
    """
    qt_pl=bib_functions.quantizar_dataSet(polar_bd)
    qt_eq=bib_functions.quantizar_dataSet(equatorial_bd)
    
    _csv_qt_pl = path_vistas + '/qt_polar_.csv'
    data = [[qt_pl]]
    #print(data)
    utils_lib.add_row_csv(_csv_qt_pl, data)
    
    _csv_qt_eq = path_vistas + '/qt_equatorial_.csv'
    data = [[qt_eq]]
    #print(data)
    utils_lib.add_row_csv(_csv_qt_eq, data)
    """

def initial(params):

    CATEGORIES = sorted(os.listdir(params['bd_src']))
    #CATEGORIES = ['1.Thymbra','10.Satureja', '12.Calicotome', '13.Salvia', '14.Sinapis', '15.Ferula', '17.Oxalis', '2.Erica', '20.Olea', '3.Castanea', '4.Eucalyptus', '5.Myrtus', '6.Ceratonia', '7.Urginea', '8.Vitis', '9.Origanum']
    CATEGORIES_labels = sorted(os.listdir(params['path_labels']))
    #CATEGORIES_labels=['equatorial_alongada', 'equatorial_circular', 'equatorial_eliptica', 'polar_circular', 'polar_triangular', 'polar_tricircular']
    print(CATEGORIES_labels)
    
    bib_functions.create_folders(params['bd_dst'], flag=1)
    _csv_head = params['bd_dst'] + '/head_.csv'
    data = [["modelo", "path_labels", "motivo", "data"],
            [params['path_model'], params['path_labels'], params['motivo'], params['date'] ]]
    print(data)
    utils_lib.add_row_csv(_csv_head, data)


    # Load model
    model=models.load_model(params['path_model'])
    model.summary()
    
    return CATEGORIES, CATEGORIES_labels, model

"""# Main"""
# Sets the working directory
os.chdir('/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/')

params={'input_shape':(224,224,3),
        'bd_src': "./BD/CPD1_Is_Rc/",
        'bd_dst':"./BD/CPD1_Dn_VTcr_220824/",
        'path_model': "./0_pseudo_labeling/REPORTS/CLASS_PSEUDO_PRE/8_DenseNet201_sem_BI_5/Train_8_DenseNet201_sem_BI_5/DenseNet201_bestLoss_6.keras",
        'path_labels': "./BD/BI_5/labels/",
        'motivo': "Refazer a BD Vistas utilizando a base isolated",
        'date':"21/08/24"
        }

if __name__=="__main__":
    
    CATEGORIES, CATEGORIES_labels, model=initial(params)
    labelling(CATEGORIES, CATEGORIES_labels, params['bd_src'], params['bd_dst'], model)
    

