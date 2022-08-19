# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:32:44 2022

@author: RajanChokshi
"""

import pandas as pd
import numpy as np
import math

# Scale input array x between [0, 1]
def normalize(x): 
    return (x - x.min())/(x.max()-x.min())

def standardize(x):
    return (x - x.mean())/x.std()

# Massage data -- Normalize or Standardize or do nothing??
def preprocessData(x, mode='NONE'):
    if (mode == 'NORMALIZE'):
        return normalize(x)
    elif (mode == 'STANDARDIZE'):
        return standardize(x)
    else:
        return x

# Import Data
path = r'C:\Users\RajanChokshi\OneDrive - Accutant Solutions LLC\Accutant RC\Business\Training\2022\ALCE Training\SRP Example'  #<--- folder-name for data files
dyna_file = path + '\\DynaCardsv2.xlsx'
dyna = pd.read_excel(dyna_file, header=None, names=['label', 'cardstream'])

# Change label type to category & add a label_code column
dyna.label = pd.Categorical(dyna.label)
dyna['labelCode'] = dyna.label.cat.codes

# Modify cardstream data 
# Remove curly braces
# Convert strings to numpy array and 
# split into two components
dyna['card'] = dyna['cardstream'].replace('[{}]','', regex=True) \
    .apply(lambda x: np.fromstring(x, dtype=float, sep=',')) \
        .apply(lambda x: np.split(x, 2))


# Split each card into load and displacement arrays and normalize values
dyna['load'] = dyna.card.apply(lambda x: x[0])
dyna['disp'] = dyna.card.apply(lambda x: x[1])

# Preprocess data
#MODE = 'STANDARDIZE'
MODE = 'NORMALIZE'
for cName in ['load', 'disp']:
    dyna[cName] = dyna[cName].apply(lambda x: preprocessData(x, mode=MODE))
  
# Featureset 1 - Centroids or means
dyna['mean_load'] = dyna.load.apply(np.mean)
dyna['mean_disp'] = dyna.disp.apply(np.mean)

    
# Featureset 3 - Fourier Descriptors - First five only
from scipy.fft import rfft
LOW = 0
HIGH = 5

# Helper function to create column names
def colNames(prefix, LOW,HIGH):
    colList = []
    for i in range(LOW, HIGH):
        colList.append(prefix+'_'+str(i))
    return colList

# Dataframe of Load FFT-Coeffs
load_fftcoeffs = pd.DataFrame(
    data=dyna.load.apply(lambda x: np.real(rfft(x))[LOW:HIGH]).to_list(),
    columns=colNames('lfftCoeff', LOW, HIGH)    )

# Dataframe of Displacement FFT-Coeffs
disp_fftcoeffs = pd.DataFrame(
    data=dyna.disp.apply(lambda x: np.real(rfft(x))[LOW:HIGH]).to_list(),
    columns=colNames('dfftCoeff', LOW, HIGH)   )

dyna = pd.concat([dyna, load_fftcoeffs, disp_fftcoeffs], axis='columns')
dyna.dropna(axis=0, inplace=True)

# Preprocess calculated features
MODE = 'STANDARDIZE'
for cName in dyna.columns[8:]:
    dyna[cName] = preprocessData(dyna[cName], mode=MODE)

#plot a set of random example cards from each category
import matplotlib.pyplot as plt
  
cardlabels=dyna.label.unique()
fig, axs = plt.subplots(nrows=math.ceil(cardlabels.size/2), ncols=2,
                        sharex=True, sharey=True)
# set labels
plt.setp(axs[-1, :], xlabel='Norm DISP, -')
plt.setp(axs[:, 0], ylabel='Norm LOAD, -')
#plt.setp(axs, xlim=(0.,1.0), ylim=(0.,1.0) )

m = n = 0
nMax = 2
for l in cardlabels:
    idx = dyna[dyna.label==l].sample().index[0]
    axs[m, n].plot(dyna.disp[idx], dyna.load[idx]) 
    axs[m, n].scatter(dyna.mean_disp[idx], dyna.mean_load[idx], c='red') 
    axs[m, n].set_title('ID:'+str(idx)+': '+dyna.label[idx], fontsize = 10)
    n = n + 1
    if (n == nMax):
        n = 0 
        m = m + 1

plt.subplots_adjust(bottom=0.3, top=1.5)
plt.show()
#----------------------------------------------------------------------
# Plot Centroids
from matplotlib.colors import from_levels_and_colors

u, inv = np.unique(dyna.label, return_inverse=True)
cmap, norm = from_levels_and_colors(np.arange(0, len(u)+1)-0.5, 
                                    plt.cm.viridis(np.linspace(0,1,len(u))))
p1 = plt.scatter(dyna.mean_load, dyna.mean_disp, 
                 c=inv, cmap=cmap, norm=norm, alpha=0.4)
plt.legend(p1.legend_elements()[0],u)#, loc='lower right')
plt.show()
#----------------------------------------------------------------------
# Plot first coefficients
p1 = plt.scatter(dyna.lfftCoeff_0, dyna.dfftCoeff_0, 
                 c=inv, cmap=cmap, norm=norm, alpha=0.4)
plt.legend(p1.legend_elements()[0],u)#, loc='lower right')
plt.show()
#----------------------------------------------------------------------

# Corrlation map
f = plt.figure(figsize=(19, 15))
plt.matshow(dyna.corr(), fignum=f.number)
plt.xticks(range(dyna.select_dtypes(['number']).shape[1]), dyna.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(dyna.select_dtypes(['number']).shape[1]), dyna.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
#----------------------------------------------------------------------
# Form dataset X- features & y is target
X = dyna[dyna.columns[8:]].to_numpy()
y = dyna.labelCode

# Sub-divide datatest into training and testing: 70 - 30% split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, random_state=1002)

from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate

NUMFOLDS = 5
# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=1029)
#clf_LR = cross_val_score(log_reg, X_train, y_train, cv = 10, scoring='accuracy')
res_LR = cross_validate(log_reg, X_train, y_train, cv = NUMFOLDS, scoring=['balanced_accuracy','f1_macro'])

# Gaussian Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

# Support Vector Machine
from sklearn.svm import LinearSVC
svm = LinearSVC(dual=False)
#clf_SVM = cross_val_score(svm, X_train, y_train, cv = 10, scoring='accuracy')
res_SVM = cross_validate(svm, X_train, y_train, cv = NUMFOLDS, scoring=['balanced_accuracy','f1_macro'])

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=1029)
#clf_DT = cross_val_score(decision_tree, X_train, y_train, cv = 10, scoring='accuracy')
res_DT = cross_validate(dt, X_train, y_train, cv = NUMFOLDS, scoring=['balanced_accuracy','f1_macro'])

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=1029)
#clf_RF = cross_val_score(random_forest, X_train, y_train, cv = 10, scoring='accuracy')
res_RF = cross_validate(rf, X_train, y_train, cv = NUMFOLDS, scoring=['balanced_accuracy','f1_macro'])


# Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(random_state=1029)
#clf_ET = cross_val_score(extra_tree, X_train, y_train, cv = 10, scoring='accuracy')
res_ET = cross_validate(et, X_train, y_train, cv = NUMFOLDS, scoring=['balanced_accuracy','f1_macro'])

# GradientBoosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=1029)
#clf_GB = cross_val_score(gb, X_train, y_train, cv = 10, scoring='accuracy')
res_gb = cross_validate(gb, X_train, y_train, cv = NUMFOLDS, scoring=['balanced_accuracy','f1_macro'])

# XGBoost Classifier
import xgboost as xgb
#y.astype("category")
xgb_model = xgb.XGBClassifier(use_label_encoder=False)
res_xgb = cross_validate(xgb_model, X_train, y_train, cv = NUMFOLDS, scoring=['balanced_accuracy','f1_macro'])



# Neural network using Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

SPARSE = True
if (SPARSE):
    yy = y
    loss = 'sparse_categorical_crossentropy'
else:
    yy = to_categorical(y)
    loss = 'categorical_crossentropy'
    
X_train, X_test, y_train, y_test = train_test_split(
    X, yy, 
    test_size=0.3, random_state=1002)

nn = Sequential()
nn.add(Dense(units=100, activation='relu', kernel_initializer='he_uniform'))
nn.add(Dense(units=50, activation='relu', kernel_initializer='he_uniform'))
nn.add(Dense(units=4, activation='softmax'))
opt = SGD(learning_rate=0.01, momentum=0.9)
nn.compile(loss=loss,
           optimizer=opt, metrics=['accuracy'])
# fit model
history = nn.fit(X_train, y_train, validation_data=(X_test, y_test),
                 epochs=100, verbose=0)
# evaluate model
_, train_acc = nn.evaluate(X_train, y_train, verbose=0)
_, test_acc = nn.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' %(train_acc, test_acc))
# plot loss during training
plt.subplot(211)
plt.title('Categorical Cross-Entropy Loss', pad=20)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
#
#y_pred = np.argmax(nn.predict(X_test), axis=1)

# Calculate the confusion matrix
#
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
def plot_confusion_matrix(model, model_name):
    model.fit(X_train, y_train)
    if (model_name=='Neural Network'): #and (not SPARSE)):
        y_pred = np.argmax(model.predict(X_test), axis=1)
    else:
        y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    #fig, ax = plt.subplots(figsize=(7.5, 7.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
                                  display_labels=cardlabels)
    disp = disp.plot(xticks_rotation=45)
    
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.title('Confusion Matrix - ' + model_name, fontsize=16)
    plt.show()
    #plt.close(fig)
    

# plot_confusion_matrix(log_reg, 'Logistic Regression')
# plot_confusion_matrix(gnb, 'Gaussian Bayes')
# plot_confusion_matrix(svm, 'Support Vector Machine')
# plot_confusion_matrix(dt, 'Decision Tree')
# plot_confusion_matrix(rf, 'Random Forest Classifier')
# plot_confusion_matrix(et, 'Extra Trees')
plot_confusion_matrix(gb, 'Gradient Boosting Classifier')
plot_confusion_matrix(xgb_model, 'XG Boost')
plot_confusion_matrix(nn, 'Neural Network')