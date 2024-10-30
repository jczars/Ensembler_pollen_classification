#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


import sys,os
import cv2
import glob
from tqdm import tqdm

sys.path.append('/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/')
print(sys.path)
from Library import bib_functions as func


# In[2]:


def folders_exists(folder):
  if not(os.path.isdir(folder)):
      os.mkdir(folder)
      print('pasta criada ', folder)


# # resize

# In[3]:


def save_resize(src, dst, tipo='jpg',save=0, verbose=0):
  folders_exists(dst)
  for filename in tqdm(glob.iglob(os.path.join(src, "*."+tipo))):
    
    if verbose==1:
        print('\n [STEP 1 * ]')
        nm = filename.split('/')[-1]
        print('nm: ', nm)
        new_path=dst+'/'+nm
        print('src', filename)
        print('dst', new_path)
    if save==1:
        print('\n [STEP 2 * save==1, salvando as imagens ]')
        print('dst', new_path)
        img = cv2.imread(filename)
        img_re=cv2.resize(img, (224, 224))
        cv2.imwrite(new_path, img_re)
  if verbose>=1:
    print(src)
    images_path = glob.glob(src+"/*."+tipo)
    print('total de imagens ', len(images_path))
    print(dst)
    images_path = glob.glob(dst+"/*."+tipo)
    print('total de imagens ', len(images_path))


# In[4]:


def run(params):
    CATEGORIES = sorted(os.listdir(params['bd_src']))

    folders_exists(params['bd_dst'])
    
    for i in CATEGORIES:
        src=params['bd_src']+'/'+i
        dst=params['bd_dst']+'/'+i
        
        #verbose == 0, só imprime mas, não salva
        if params['verbose']==0:
            print('\n origem', src)
            print('destino', dst)
        save_resize(src, dst, params['tipo'], params['save'], params['verbose'])
    func.graph_img_cat(params['bd_dst'], params['save_dir'])
    


# In[5]:

# Sets the working directory
os.chdir('/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/')
params={'tipo':'png',
        'bd_src': "./BD/Isolated Pollen Grains",
        'bd_dst':"./BD/CPD1_Is_Rc/",
        'save_dir': "./01_createBD/",
        'save': 1, #save 1, salva as imagens redimensionadas
        'verbose': 1 #se verbose == 0, imprime os caminhos src, dst imagens
        }


# In[6]:


run(params)

