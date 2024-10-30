import sys, gc
sys.path.append('/media/jczars/4C22F02A22F01B22/01_PROP_TESE/02_PSEUDO_LABELS/')
print(sys.path)
from modulos import get_data
from modulos import utils
from modulos import reports

import pandas as pd

def classificaImgs(conf, _path_model, _tempo, model_inst, _pseudo_csv,
                   CATEGORIES, tipo='png', verbose=1):
  """
  1- carrega os pesos do modelo2
  2- instancia o modelo com os pesos
  3- carregando as imagens do não rotulada(unlabels_csv)
  4- faz pseudos rotulações na base não rotulada
  5- salvando o DataSet unlabels, tempo
  """

  print('[classificaImgs].1-carregando os pesos do modelo')
  #nm_modelo+'_bestLoss_' + str(k) + '.keras'
  #Mobilenet_bestLoss_0.keras
  #/media/jczars/4C22F02A22F01B22/01_PROP_TESE/02_PSEUDO_LABELS/REPORTS/CLASS_PSEUDO_test/0_Mobilenet_sem_BI_5/Train_0_Mobilenet_sem_BI_5/
  print('_path_model ', _path_model)
  path=_path_model+conf['model']+'_bestLoss_'+str(_tempo)+'.keras'
  print(path)

  print('[classificaImgs].2- instancia o modelo com os pesos')
  model_inst.load_weights(path)

  print('[classificaImgs].3-carregando as imagens não rotulada---> unlabel DataSet')
  # tempo T0
  if _tempo==0:
      unalbels_generator=get_data.load_unlabels(conf)
  else:      
      _cs_uns_ini=_pseudo_csv+'/unlabelSet_T'+str(_tempo)+'.csv'
      df_uns_ini=pd.read_csv(_cs_uns_ini)
      if len(df_uns_ini)>0:
          print(df_uns_ini.head())
          print(f"\ntempo {_tempo}, read _cs_uns_ini{_cs_uns_ini}")
          unalbels_generator=get_data.load_data_test(df_uns_ini, input_size=(224,224))
      else:
          return df_uns_ini
      

  print('[classificaImgs].4-faz pseudos rotulações na base não rotulada ---> unlabel DataSet')
  data_uns=reports.predict_unlabels_data_gen(conf, unalbels_generator, model_inst, CATEGORIES)
  print(f"data_uns, {len(data_uns)}")
  
  """
  #if _tempo==0:
  _csv_unlabels_t= _pseudo_csv+'/unlabelSet_T'+str(_tempo)+'.csv'
  print(f'[classificaImgs].5- salvando o DataSet unlabels{_csv_unlabels_t} tempo= {_tempo}')
  data.to_csv(_csv_unlabels_t, index = False, header=True)
  """
  return data_uns


def selec(conf,
          data_uns_ini,
          _pseudo_csv, 
          CATEGORIES, 
          _tempo, 
          train_data_csv, limiar):
  """step select
  Tempo=0:
  1- ler csv unlabels
  2- filtrar por confiança
  3- seleção dos pseudos rótulos
      -selecionar no mínimo 100 por classe
  4- excluir os rótulos do dataset unlabels
  5- juntar o traino anterior com os pseudos rótulos selecionadaos
  6- salva o novo TrainSet
  7- regras de parada
  """
  
  utils.renomear_path(conf, data_uns_ini)
  print(data_uns_ini.head())

  print("\n[select 2] Filtrar por confiança")  
  _size_data_uns_ini=len(data_uns_ini)
  print(f'\nSize_unlabels inicial {_size_data_uns_ini}, tempo {_tempo} ')
  
  data_uns_fil=data_uns_ini.loc[data_uns_ini['confianca']>limiar]
  print('Size data_uns_fil', len(data_uns_fil))
  
  if len(data_uns_fil)==0:
    print('pseudos, não passou pelo filtro de 95%')
  """
  menor=utils.get_menor(data_uns_fil, CATEGORIES)
  print('menor', menor)

  print("\n[select 3] seleção dos pseudos rótulos")
  df_sel, df_cat_size=utils.select_pseudos(data_uns_fil, CATEGORIES, menor, _tempo)  
  """

  print(f"\n[select 4] Excluir os rótulos do dataset unlabels, tempo {_tempo}")
  if len(data_uns_fil)>0:
    for i in data_uns_fil['file']:
        #print(i)
        #print(f"data_uns_fil[0] ,{data_uns_fil.loc[0]}")
        data_uns_ini.drop(data_uns_ini[data_uns_ini['file'] == i].index, inplace = True)
  else:
    print('\n[REGRA DE PARA 0 - len(df_sel)>0]')
    return False
  
  _size_uns_select=len(data_uns_fil)
  #print(f'Total data_uns inicial {_size_data_uns_ini}, tempo {_tempo} ')
  print(f'Tamanho dos pseudos selecionados {_size_uns_select}, tempo {_tempo} ')
  
  _size_uns_rest=len(data_uns_ini)
  print(f'unlabels restantes {_size_uns_rest}, tempo {_tempo} ')

  print(data_uns_ini.head())
  
  tempo_px=_tempo+1
  _csv_unlabels_t= _pseudo_csv+'/unlabelSet_T'+str(tempo_px)+'.csv'
  print(f'[BUILD] salvando unlabels restantes, no tempo {tempo_px} ,{_csv_unlabels_t}')
  data_uns_ini.to_csv(_csv_unlabels_t, index = False, header=True)

  if _tempo==0:
      print(f'\nTamanho do train anterior, {len(train_data_csv)}, tempo {_tempo} ')
      
      print("\n[select 5]- juntar o traino anterior com os pseudos rótulos selecionadaos")
      New_train_data = pd.concat([train_data_csv, data_uns_fil], ignore_index=True, sort=False)
      print(f'Tamanho do novo train, {len(New_train_data)} , tempo {_tempo} ')
  else:
      _csv_TrainSet = _pseudo_csv+'/trainSet_T'+str(_tempo)+'.csv'
      print('[BUILD _csv_TrainSet', _csv_TrainSet)
      train_data_csv=pd.read_csv(_csv_TrainSet)
      
      print("\n[select 5]- juntar o traino anterior com os pseudos rótulos selecionadaos")
      New_train_data = pd.concat([train_data_csv, data_uns_fil], ignore_index=True, sort=False)
  
  _size_New_train_data=len(New_train_data)
  _size_train_ant=len(train_data_csv)
  print(f'\nTamanho do train anterior, {_size_train_ant}, tempo {_tempo} ')
  print(f'Tamanho do train proximo, {_size_New_train_data} , tempo prox {tempo_px} ')
  
  #salva o novo TrainSet
  _csv_New_TrainSet = _pseudo_csv+'/trainSet_T'+str(tempo_px)+'.csv'
  print(f"\n[select 4] salva o novo TrainSet, no tempo {tempo_px} ,{_csv_New_TrainSet}")
  New_train_data.to_csv(_csv_New_TrainSet, index = False, header=True)

  tamanho = utils.bytes_to_mb(utils.sys.getsizeof(New_train_data))
  print(f'O tamanho da variável{New_train_data} é aproximadamente {tamanho} MB.')
  
  
  sel_0={'ini':_size_data_uns_ini, 
         'select': _size_uns_select, 
         'rest':_size_uns_rest, 
         'train':_size_train_ant, 
         'new_train': _size_New_train_data,
         '_csv_New_TrainSet':_csv_New_TrainSet}
  
  return sel_0

if __name__=="__main__": 
    help(classificaImgs)
    help(selec)