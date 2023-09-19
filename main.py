import pandas as pd
import numpy as np

'''
read dataset
'''

data = pd.read_csv('dataset/urlset.csv',encoding='ISO-8859-1')[:50000]

'''
data cleaning 
'''
data = data.replace(np.nan,0)
data['label'].replace(["0.7'8049", '0.770083'], 0, inplace=True)
df = data.apply(pd.to_numeric, errors='coerce')
### extract label ###
label = df['label'].values
data = df.drop(['label','domain','Unnamed: 14'],axis=1)
data = data.replace(np.nan,0)


'''
normalization
'''
### Z-score ###
from scipy import stats

z_scores = stats.zscore(data)
z_scores = z_scores.replace(np.nan,0)

'''
Adv-SyN
'''
### Advanced synthetic sampling approach for balancing dataset ###
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(z_scores, label)
X = np.array(X_resampled)


'''
feature extraction
'''
import DSelSa
### Double self sparse autoencoder ###
print("\nExtract Features....\n")
input_shape = X[0].shape
dsae_model = DSelSa.create_dsae(input_shape)
features = dsae_model.predict(X)

'''
feature selection
'''

from OpGoA import *
### Opposition-based Gazelle Optimization Algorithm ###

n_iterations = 1
n_population = 50
opgoa = Feat_selection(n_population,n_iterations,features,y_resampled)
sel_index = opgoa.OPGOA()
print("\nSelected Features:", sel_index)
sel_features = features[:,sel_index]

### split adat for training and testing ###
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(sel_features, y_resampled, test_size=0.2, random_state=42)

'''
attack classification
'''
### Multi Head Depth wise Tern integrated long short term memory  ###
from keras.models import load_model
import MDepthNet

model = load_model('Model')
clf = MDepthNet.MDepthNet(X_train,y_train,X_test,y_test)
pred_vals = clf.predict(model)

'''
performance claculation and comparison 
'''
import performance
    
cnf_matrix, Accuracy,precision,f1_score,recall,mcc,rmse,mae = performance.Performancecalc(y_test,pred_vals)

print('\n_____Proposed Performance_____\n')
print('*** MDepthNet ***\n')
print('Accuracy  :',Accuracy)
print('Precision :',precision)
print('F1_score  :',f1_score)
print('Recall    :',recall)
print('Matthews corrcoef  :',mcc)
print('RMSE :',rmse)
print('MAE  :',mae)

import plot

plot.cnf_metx(cnf_matrix)
plot.ACC_LOSS(model)
plot.ROC_curve(y_test,pred_vals)
plot.plot_comparison(Accuracy,precision,f1_score,recall,mcc,rmse,mae)