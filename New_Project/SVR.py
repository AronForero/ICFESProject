import pandas as pd
import numpy as np
from time import time
from sklearn.svm import SVR
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
print('Iniciando... importando datos...')
data_2000 = pd.read_csv('DatosFTPICFES/SABER11/SB11-BASES_DE_DATOS/READY/2000_1.csv', delimiter=';')
y_list = ['PUNT_BIOLOGIA', 'PUNT_MATEMATICAS', 'PUNT_FILOSOFIA', 'PUNT_FISICA', 'PUNT_HISTORIA', 'PUNT_QUIMICA', 
          'PUNT_LENGUAJE', 'PUNT_GEOGRAFIA', 'PUNT_INTERDISCIPLINAR', 'PUNT_IDIOMA']
X_list = data_2000.columns.difference(y_list)
S_data = data_2000.sort_values(by='PUNT_BIOLOGIA')
print('dividiendo datos...')
X = S_data.filter(items = X_list)
Y = S_data.filter(items = y_list)
print('haciendo lo dificil')
svr = SVR(kernel='poly', degree=2)
rs = ShuffleSplit(n=data_2000.shape[0], n_iter=5, test_size=0.2, train_size=0.4)
#cv = KFold(n = X.shape[0], n_folds=5, shuffle=True)
scores = -cross_val_score(svr, X, Y['PUNT_BIOLOGIA'], scoring='mean_absolute_error', cv = rs)
print('The Scores given for the cross_val_score with SVR are:', scores)
print('The mean of the scores above is:', np.mean(scores))
