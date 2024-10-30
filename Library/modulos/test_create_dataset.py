import os
import pandas as pd
import tqdm


def create_dataSet(_path_data, _csv_data, CATEGORIES):
  """
  --> create data set in format csv
  :param: _path_data: path the dataSet
  :param: _csv_data: path with file name '.csv'
  :param: _categories: the classes the data
  :return: DataFrame with file path
  """
  data=pd.DataFrame(columns= ['file', 'labels'])
  print('_path_data ', _path_data)
  print('_csv_data ', _csv_data)
  print('CATEGORIES: ', CATEGORIES)

  c=0
  #cat_names = os.listdir(_path_data)
  for j in CATEGORIES:
      pathfile = _path_data+'/'+j
      print(pathfile)
      filenames = os.listdir(pathfile)
      for i in filenames:
        #print(_path_data+'/'+j+'/'+i)
        data.loc[c] = [str(_path_data+'/'+j+'/'+i), j]
        c=c+1
  #print(c)
  data.to_csv(_csv_data, index = False, header=True)
  data_csv = pd.read_csv(_csv_data)
  print(_csv_data)
  print(data_csv.groupby('labels').count())

  return data

_path_data='/media/jczars/4C22F02A22F01B22/01_PROP_TESE/01_PSEUDO_BD/BI_5/labels'
_csv_data='/media/jczars/4C22F02A22F01B22/01_PROP_TESE/01_PSEUDO_BD/BI_5/BI_5.csv'
CATEGORIES= ['equatorial_alongada', 'equatorial_circular', 'equatorial_eliptica', 'polar_circular', 'polar_triangular', 'polar_tricircular']

create_dataSet(_path_data, _csv_data, CATEGORIES)