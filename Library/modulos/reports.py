from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, CATEGORIES, nm_model,
                          save_dir='', tempo=0, verbose=0):

    """
  --> Confusion Matrix, print or save
  :param: test_data_generator: images from Image Date Generator
  :param: y_pred: labels predict
  :param: CATEGORIES: list classes the of images
  :param: save_dir: path to save confusion Matrix, standar ='', not save
  :param: fold_var: numers order
  :param: verbose: if == 1, print graph the confusion matrix, if == 0, not print
  :return: vector of the metrics
  """
    print('Confusion Matrix')
    mat = confusion_matrix(y_true, y_pred,
                           normalize=None)
    # Save matrix 21/01/24
    print('salvando matriz como csv em ', save_dir)
    df_mat = pd.DataFrame(mat)

    if not (save_dir == ''):
        df_mat.to_csv(save_dir +'mat_conf_test_'+'_'+nm_model+'_tempo_'+
                    str(tempo) + '.csv')

    #plt.figure(figsize=fig_size)
    my_dpi=100
    plt.figure(figsize=(900/my_dpi, 900/my_dpi), dpi=my_dpi)
    ax = plt.subplot()
    sns.heatmap(mat, cmap="Blues", annot=True)  # annot=True to annotate cells

    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=90, horizontalalignment='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(),
                       rotation=0, horizontalalignment='right', fontsize=8)

    # labels, title and ticks
    ax.set_xlabel('Rótulos previstos')
    ax.set_ylabel('Rótulos verdadeiros')
    ax.set_title('Matriz de confusão')


    ax.xaxis.set_ticklabels(CATEGORIES)
    ax.yaxis.set_ticklabels(CATEGORIES)


    if not (save_dir == ''):
        #plt.rcParams['savefig.dpi'] = 300
        #dpi=300
        plt.savefig(save_dir +'mat_conf_test_'+'_'+nm_model+'_tempo_'+
                    str(tempo) + '.jpg')
    if verbose == 1:
        plt.show()

def predict_data_generator(conf, return_0, test_data_generator, model, tempo,
                            CATEGORIES, verbose=2):

    """
    --> evaluation metrics (accuracy_score, precision, recall, fscore, kappa)
    --Version V1 17/04/24 09:59
    :param: test_data_generator: images from Image Date Generator
    :param: batch_size: labels predict
    :param: model: model load
    :param: save_dir: model load
    :param: k: number k in kfolders
    :param: CATEGORIES: list classes the of images
    :param: verbose: enable printing
    :return: print classification reports
    """
    batch_size=conf['batch_size']
    nm_model=conf['model']
    save_dir=return_0['save_dir_train']

    filenames = test_data_generator.filenames
    y_true = test_data_generator.classes
    df = pd.DataFrame(filenames, columns=['filenames'])
    confianca = []
    nb_samples = len(filenames)
    y_preds = model.predict(test_data_generator,
                            steps=nb_samples // batch_size + 1)

    for i in range(len(y_preds)):
        confi = np.max(y_preds[i])
        confianca.append(confi)
        if verbose == 1:
            print('i ', i, ' ', y_preds[i])
            print('confi ', confi)

    y_pred = np.argmax(y_preds, axis=1)
    if verbose == 2:
        print('Size y_true', len(y_true))
        print('Size y_pred', len(y_pred))

    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['confianca'] = confianca
    df.insert(loc=2, column='labels', value='')
    df.insert(loc=4, column='predict', value='')
    df.insert(loc=6, column='sit', value='')

    # veficar acertos
    for idx, row in df.iterrows():
        cat_true = CATEGORIES[row['y_true']]
        cat_pred = CATEGORIES[row['y_pred']]
        if verbose == 1:
            print('cat_true ', cat_true, 'cat_pred ', cat_pred)
        df.at[idx, 'labels'] = cat_true
        df.at[idx, 'predict'] = cat_pred
        if row['y_true'] == row['y_pred']:
            df.at[idx, 'sit'] = 'C'
        else:
            df.at[idx, 'sit'] = 'E'

    df = df.sort_values(by='labels')
    df_Err = df[df['sit'] == 'E']
    df_cor = df[df['sit'] == 'C']


    if not (save_dir == ''):
        df_Err.to_csv(save_dir+'filterWrong_'+'_'+nm_model+'_tempo_'+
                    str(tempo) +'.csv', index=True)

        df_cor.to_csv(save_dir+'filterCorrect_'+'_'+nm_model+'_tempo_'+
                    str(tempo) +'.csv', index=True)

    return y_true, y_pred, df_cor

def predict_unlabels_data_gen(conf, test_data_generator, model, CATEGORIES, verbose=2):

  """
  --> evaluation metrics (accuracy_score, precision, recall, fscore, kappa)
  --Version V1 17/04/24 09:59
  :param: test_data_generator: images from Image Date Generator
  :param: batch_size: labels predict
  :param: model: model load
  :param: save_dir: model load
  :param: k: number k in kfolders
  :param: CATEGORIES: list classes the of images
  :param: verbose: enable printing
  :return: print classification reports
  """
  batch_size=conf['batch_size']

  filenames = test_data_generator.filenames
  y_true = test_data_generator.classes
  df = pd.DataFrame(filenames, columns=['file'])
  confianca = []
  nb_samples = len(filenames)
  y_preds = model.predict(test_data_generator,
                          steps=nb_samples // batch_size + 1)

  for i in range(len(y_preds)):
      confi = np.max(y_preds[i])
      confianca.append(confi)
      if verbose == 1:
          print('i ', i, ' ', y_preds[i])
          print('confi ', confi)

  pred = np.argmax(y_preds, axis=1)
  y_pred = [CATEGORIES[y] for y in pred] # convert

  df['labels'] = y_pred
  df['confianca'] = confianca

  return df

## metrics

def metricas(y_true, y_pred):
      """
      --> evaluation metrics (accuracy_score, precision, recall, fscore, kappa)
    :param: y_true: labels real
    :param: y_pred: labels predict
    :return: vector of the metrics
    """
      #-------metrics
      print('\n3-Metrics')
      print('Acc, precision, recall, fscore, kappa')
      #me=reports_lib.metricasV1(y_true, y_pred)

      precision, recall, fscore, support = metrics.precision_recall_fscore_support(
      y_true, y_pred)
      kappa = metrics.cohen_kappa_score(y_true, y_pred)
      prec=3
      me ={'precision':round(np.mean(precision, axis=0), prec),
       'recall':round(np.mean(recall, axis=0), prec),
       'fscore':round(np.mean(fscore, axis=0), prec),
       'kappa':round(kappa, prec)}

      return me

## class_reports

def class_reports(y_true, y_pred, CATEGORIES, nm_model, save_dir='',
                  tempo=0, verbose=0):
    """
  --> evaluation metrics (accuracy_score, precision, recall, fscore, kappa)
  --Version V1 17/04/24 09:59
  :param: y_true: labels real
  :param: y_pred: labels predict
  :param: CATEGORIES: list classes the of images
  :return: print classification reports
  """

    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=CATEGORIES))
    # Salva o relatório de classificação
    report = classification_report(
        y_true, y_pred, target_names=CATEGORIES, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    if not (save_dir == ''):
        df_report.to_csv(save_dir+'class_report_test_'+'_'+nm_model+
                     '_tempo_'+str(tempo) + '.csv', index=True)

## boxplot

def boxplot(nm_model, df_corre, save_dir='', tempo=0, verbose=0):
    my_dpi=100
    plt.figure(figsize=(900/my_dpi, 900/my_dpi), dpi=my_dpi)

    sns.set_style("whitegrid")

    # Adicionando Título ao gráfico
    sns.boxplot(y=df_corre["labels"], x=df_corre["confianca"])
    plt.title("Classes sem erros de classificação", loc="center", fontsize=18)
    plt.xlabel("acurácia")
    plt.ylabel("classes")

    if not (save_dir == ''):
        plt.savefig(save_dir+'/boxplot_correct_'+nm_model+'_tempo_'+
                    str(tempo) + '.jpg')
    if verbose == 1:
        plt.show()