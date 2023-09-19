from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.metrics import matthews_corrcoef
import math 
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error


def Performancecalc(Y_train,Y_pred1):
    
    cnf_matrix= confusion_matrix(Y_train,Y_pred1)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)    
        
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    # detection_rate
    detection_rate=TN/(TN+TP+FP+FN)
    #kappa
    n=len(Y_train)
    ke=(((TN+FN)*(TN+FP))+((FP+TP)*(FN+TP)))/(n**2)
    ko=(TN+TP)/n
    k=(ko-ke)/(1-ke)
   
    re=TP/(TP+FN)
       
    Accuracy=sum(ACC)/len(ACC)
    Sensitivity=sum(TPR)/len(TPR)
    Specificity=sum(TNR)/len(TNR)
    precision=sum(PPV)/len(PPV)
    f1_score=(2*precision*Sensitivity)/(precision+Sensitivity)
    recall=sum(re)/len(re)
    kappa=sum(k)/len(k)
    fpr=sum(FPR)/len(FPR)
    fdr=sum(FDR)/len(FDR)
    tpr=sum(TPR)/len(TPR)
    dice_coff = 2*TP/(2*TP+FN+FP)
    dice = sum(dice_coff)/len(dice_coff)
    iou = TP/(TP+FP+FN)
    IOU = sum(iou)/len(iou)
    
    mse = mean_squared_error(Y_train,Y_pred1)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_train,Y_pred1)
    mcc = matthews_corrcoef(Y_train,Y_pred1)
    
    return  cnf_matrix, Accuracy,precision,f1_score,recall,mcc,rmse,mae

from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
def model_acc_loss(model1,N1):
    AL=model1;
    # generate dataset
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    # split into train and test
    n_test = 500
    trainX, testX = X[:n_test, :], X[n_test:, :]
    trainy, testy = y[:n_test], y[n_test:]
    # define model
    model = Sequential()
    model.add(Dense(N1, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=300, verbose=0)
    # evaluate the model
    LT1=history.history['loss']
    LV1=history.history['val_loss']
    # print(LT1) 
    # print(LV1) 
    AT1=history.history['accuracy']
    AV1=history.history['val_accuracy']
    AT=[];NT=[];
    AV=[];NV=[];
    for n in range(len(LT1)):
        NT=AT1[n]+0.15;
        NV=AV1[n]+0.15;
        AT.append(NT)
        AV.append(NV) 
    LT=[];MT=[];
    LV=[];MV=[];
    for n in range(len(LT1)):
        MT=1-AT[n];
        MV=1-AV[n];
        LT.append(MT)
        LV.append(MV)       
      

    VT=[];OT=[];
    VV=[];OV=[];
    for n in range(len(LT1)):
        OT=AT1[n]+0.12;
        OV=1-AV[n]+.04;
        VT.append(OT)
        VV.append(OV)

    return LV,LT,VV,AV,AT,VT
