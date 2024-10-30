
import pandas as pd
import sys
import gc
sys.path.append(
    '/media/jczars/4C22F02A22F01B22/01_PROP_TESE/02_PSEUDO_LABELS/')
print(sys.path)
from modulos import utils
from modulos import reports_gen
from modulos import models_train
from modulos import models_sel, models_at_sel, models_pre, models_Zero_SA, models_pre_sa
from modulos import get_data
from modulos import get_calssifica


def step_0(conf, _labels, CATEGORIES):
  """step 0
  Tempo=0:
  1-criar a pasta raiz
  2-criar a pasta teste
  3-criar o Train
  4-criar a pasta pseudo_csv
  5-criar o csv
  6-split data

  BI_5_testSet.csv
  BI_5_trainSet.csv
  BI_5_valSet.csv
  """
  _nm = conf['id_test']
  _model = conf['model']
  _aug = conf['aug']
  _base = conf['base']
  _labels = conf['labels']
  # _unlabels=conf['unlabels']
  _root = conf['root']
  _path_BD = conf['path_base']

  print('\n[STEP 0].1-criar a pasta raiz')
  """
  cria pasta raiz dos lotes de testes, exemplo
  '/content/drive/MyDrive/CPD1/CLASS_PSEUDO_Test'
  """
  utils.create_folders(_root, flag=0)

  print('\n[STEP 0].2-criar a pasta teste')
  """
  cria pasta raiz do modelo, exemplo
  .../MobileNet
  """
  nmTeste = str(_nm)+'_'+_model+'_'+_aug+'_'+_base
  path_Teste = _root+'/'+nmTeste
  utils.create_folders(path_Teste, flag=0)

  print('\n[STEP 0].3-criar o Train')
  """
  cria pasta raiz do modelo, exemplo
  save_dir_train= .../0_Mobilenet_sem_BI_5/Train_0_Mobilenet_sem_BI_5/
  """
  save_dir_train, id = utils.criarTestes(path_Teste, nmTeste)
  print(f'save_dir_train, {save_dir_train}, id {id} ')

  print('\n[STEP 0].4-Criar a pasta pseudo_csv')
  _pseudo_csv = save_dir_train+'pseudo_csv'
  utils.create_folders(_pseudo_csv, flag=0)

  print('\n[STEP 0].5-criar o csv')
  _csv_data = _path_BD+'/'+_base+'.csv'  # pastas
  print('_csv_data ', _csv_data)

  # /content/drive/MyDrive/CPD1/BD_BI/BI_5/labels
  data_csv = utils.create_dataSet(_labels, _csv_data, CATEGORIES)

  sizeOflabels = len(data_csv)
  print('Total de dados rotulados ', sizeOflabels)

  print('\n[STEP 0].6-split data')
  _path_train, _path_val, _path_test = get_data.splitData(
      data_csv, _root, _base)
  """Create the struture
  BI_5_testSet.csv
  BI_5_trainSet.csv
  BI_5_valSet.csv
  """

  return_0 = {'path_train': _path_train,
            'path_val': _path_val,
            'path_test': _path_test,
            'save_dir_train': save_dir_train,
            'pseudo_csv': _pseudo_csv,
            'sizeOflabels': sizeOflabels
               }
  return return_0


def step_1(conf, CATEGORIES, return_0, _tempo):
  """
  7-load data train
  8-Treinar o modelo
  9-Evalution
  """

  print('step 1')

  print('\nstep 1.1-load data train')
  training_data = pd.read_csv(return_0['path_train'])
  val_data = pd.read_csv(return_0['path_val'])
  img_size = conf['img_size']
  input_size = (img_size, img_size)
  train, val = get_data.load_data_train(
      training_data, val_data, conf['aug'], input_size)

  print('\nstep 1.2-instanciando o modelo')
  model_inst = None
  # model_inst=models_sel.sel_cnn(conf, CATEGORIES)
  
  if conf['type_model']=='pre':
      model_inst = models_sel.hyper_model(conf, len(CATEGORIES))
  if conf['type_model']=='att':
      model_inst = models_at_sel.hyper_model(conf, len(CATEGORIES))
  if conf['type_model']=='imagenet':
      model_inst = models_pre.hyper_model(conf, len(CATEGORIES))     
  if conf['type_model']=='zero_sa':
      model_inst = models_pre_sa.hyper_model(conf, len(CATEGORIES))
  if conf['type_model']=='pre_sa':
      model_inst = models_pre_sa.hyper_model(conf, len(CATEGORIES)) 
      
  print(f'\n[INFO]--> step 1.3-train tempo {_tempo}')
  model_inst, num_epoch, str_time, end_time, delay = models_train.fitModels(conf, return_0,
                       _tempo,
                       model_inst,
                       train, val
                       )

  print(f'\n[INFO]--> step 1.4-Evalution tempo {_tempo}')
  test_data = pd.read_csv(return_0['path_test'])
  
  img_size = conf['img_size']
  input_size = (img_size, img_size)
  print('\n[INFO]--> ', input_size)
  test = get_data.load_data_test(test_data, input_size)

  me = reports_gen.reports_build(
      conf, test, model_inst, CATEGORIES, _tempo, return_0)
  me = {'test_loss': me['test_loss'],
      'test_accuracy': me['test_accuracy'],
      'precision': me['precision'],
      'recall': me['recall'],
      'fscore': me['fscore'],
      'kappa': me['kappa'],
      'num_epoch': num_epoch,
      'str_time': str_time,
      'end_time': end_time,
      'delay': delay
      }
  return model_inst, me
# step 2


def step_2(conf, _tempo,  _unlabels, model_inst, CATEGORIES, return_0, limiar):
  """
  1-cria o dataset csv unlabels
  2-cria o dataset csv new_unlabels
  3-classificar os dados unlabels
  4-seleção de novos pseudos
  5-regras de parada
  """
  print('return_0 ', return_0)
  print(f"\n[STEP 2]-tempo {_tempo}")
  _pseudo_csv = return_0['pseudo_csv']
  _path_train = return_0['path_train']
  _path_model = return_0['save_dir_train']

  print('[build 1] _path_model ', _path_model)
  print('_pseudo_csv', _pseudo_csv)
  # criar newBD
  
  train_data_csv = pd.read_csv(_path_train)
  print('train_data_csv ', train_data_csv)

  print('\n[STEP 2].3- Classificação')
  data_uns=get_calssifica.classificaImgs(conf,_path_model, _tempo, model_inst,
                  _pseudo_csv, CATEGORIES, conf['tipo'])
  print(f"data_uns {len(data_uns)}")
  
  if len(data_uns)>0:
      print('\n[STEP 2].4- Seleção')
      #_csv_unlabels_t= _pseudo_csv+'/unlabelSet_t'+str(_tempo)+'.csv'
      
      sel_0=get_calssifica.selec(
          conf,
          data_uns,
          _pseudo_csv, 
          CATEGORIES,
          _tempo, 
          train_data_csv,
          limiar)
      
      if not(sel_0==False):          
          returns_2={'_csv_New_data': sel_0['_csv_New_TrainSet'],
                     'path_train': _path_train,
                     'path_test': return_0['path_test'],
                     'save_dir_train': _path_model,
                     'pseudo_csv': _pseudo_csv,
                     'ini':sel_0['ini'],
                     'select':sel_0['select'],
                     'rest':sel_0['rest'],
                     'train':sel_0['train'],
                     'new_train':sel_0['new_train']                 
                     }
          return returns_2
      else: 
        #_unlabels_zero=False
        return 0  
      
  else: 
    #_unlabels_zero=False
    return 0         
   
# step 3

def step_3(conf, CATEGORIES, train, val, _tempo, _model, returns_2):
  """
  1- recarregando o modelo anterior
  2- retreinar o modelo
  3- avaliar o desempenho
  """
  save_dir_test=returns_2['save_dir_train']
  loss_anterior=_tempo-1

  print('\n[STEP 3].1-recarregando o modelo', 'bestLoss_'+str(loss_anterior))

  _load_weights=save_dir_test+conf['model']+'_bestLoss_'+str(loss_anterior)+'.keras'
  _model.load_weights(_load_weights)

  print('\n[STEP 3].2- retreinar o modelo')
  model_inst, num_epoch, str_time, end_time, delay=models_train.fitModels(conf, returns_2,
                       _tempo,
                       _model,
                       train, val
                       )

  print(f'\n[STEP 3].3-Evalution tempo {_tempo}')
  test_data = pd.read_csv(returns_2['path_test'])
  img_size=conf['img_size']
  input_size=(img_size, img_size)
  test=get_data.load_data_test(test_data, input_size)

  me=reports_gen.reports_build(conf,test, model_inst, CATEGORIES,_tempo, returns_2)
  me={'test_loss':me['test_loss'],
      'test_accuracy':me['test_accuracy'],
      'precision':me['precision'],
      'recall':me['recall'],
      'fscore':me['fscore'],
      'kappa':me['kappa'],
      'num_epoch':num_epoch,
      'str_time':str_time,
      'end_time':end_time,
      'delay':delay
      }
  return model_inst, me


if __name__=="__main__":
   help(step_0)
   help(step_1)
   help(step_2)
   help(step_3)