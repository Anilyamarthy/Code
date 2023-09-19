import itertools
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib.font_manager as font_manager
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from performance import *
import pandas as pd
import existing
from numpy import reshape
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import label_binarize

font = font_manager.FontProperties(family='Times New Roman',style='normal',size=14,weight='bold')

def plot_comparison(Accuracy,precision,f1_score,recall,mcc,rmse,mae):

    bigru = np.load('load_datas/BiGru.npy')
    gru = np.load('load_datas/GRU.npy')
    lstm = np.load('load_datas/lstm.npy')
    dscnn = np.load('load_datas/dscnn.npy')
    ae = np.load('load_datas/ae.npy')

    print('\n____Existing performance____\n')
    print('*** BIGRU ***\n')
    print('Accuracy  :',bigru[0])
    print('Precision :',bigru[1])
    print('F1_score  :',bigru[2])
    print('Recall    :',bigru[3])
    print('Matthews corrcoef  :',bigru[4])
    print('RMSE :',bigru[5])
    print('MAE  :',bigru[6])
    print('\n*** GRU ***\n')
    print('Accuracy  :',gru[0])
    print('Precision :',gru[1])
    print('F1_score  :',gru[2])
    print('Recall    :',gru[3])
    print('Matthews corrcoef  :',gru[4])
    print('RMSE :',gru[5])
    print('MAE  :',gru[6])
    print('\n*** LSTM ***\n')
    print('Accuracy  :',lstm[0])
    print('Precision :',lstm[1])
    print('F1_score  :',lstm[2])
    print('Recall    :',lstm[3])
    print('Matthews corrcoef  :',lstm[4])
    print('RMSE :',lstm[5])
    print('MAE  :',lstm[6])
    print('\n*** DSCNN ***\n')
    print('Accuracy  :',dscnn[0])
    print('Precision :',dscnn[1])
    print('F1_score  :',dscnn[2])
    print('Recall    :',dscnn[3])
    print('Matthews corrcoef  :',dscnn[4])
    print('RMSE :',dscnn[5])
    print('MAE  :',dscnn[6])
    print('\n*** AE ***\n')
    print('Accuracy  :',ae[0])
    print('Precision :',ae[1])
    print('F1_score  :',ae[2])
    print('Recall    :',ae[3])
    print('Matthews corrcoef  :',ae[4])
    print('RMSE :',ae[5])
    print('MAE  :',ae[6])
    
    
    plotdata = pd.DataFrame({
        
        
        "GRU":[gru[0],gru[1],gru[2],gru[3]],     
        "BIGRU":[bigru[0],bigru[1],bigru[2],bigru[3]],
        "AE":[ae[0],ae[1],ae[2],ae[3]],
        "DSCNN":[dscnn[0],dscnn[1],dscnn[2],dscnn[3]],
        "LSTM":[lstm[0],lstm[1],lstm[2],lstm[3]],
        "Proposed":[Accuracy,precision,recall,f1_score]
        
        
        }, 
        index=["Accuracy","precision","Recall","F-Measure"]
    )
    plotdata.plot(kind="bar",rot=0,alpha=1,edgecolor='black')
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.legend(loc=3, prop=font,ncol=2)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=12) 
    plt.ylabel("Performance(%)",fontname = "Times New Roman",fontweight='bold',fontsize=15)
    plt.tight_layout()
    plt.savefig("graphs/performance.JPG",dpi=600)
        

    
    data=[[gru[6]-0.1,  gru[6]-0.05, gru[6],  gru[6]+0.05,  gru[6]+0.1],
          [bigru[6]-0.1, bigru[6]-0.05, bigru[6], bigru[6]+0.05, bigru[6]+0.1],
          [ae[6]-0.1,ae[6]-0.05,ae[6], ae[6]+0.05, ae[6]+0.1],
          [dscnn[6]-0.1,dscnn[6]-0.05, dscnn[6], dscnn[6]+0.05, dscnn[6]+0.1],
          [lstm[6]-0.1, lstm[6]-0.05, lstm[6],lstm[6]+0.05, lstm[6]+0.1],
          [mae-0.1,mae-0.05,mae,mae+0.05,mae+0.1]]
    fig = plt.figure()
    
    bp1=plt.boxplot(data[0], positions=[1], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C0")) 
    bp2=plt.boxplot(data[1], positions=[2], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C2"))
    bp3=plt.boxplot(data[2], positions=[3], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C3"))
    bp4=plt.boxplot(data[3], positions=[4], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C4"))
    bp5=plt.boxplot(data[4], positions=[5], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C5"))
    bp6=plt.boxplot(data[5], positions=[6], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C6"))
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0], bp5["boxes"][0], bp6["boxes"][0]] ,['GRU', 'BIGRU', 'AE','DSCNN', 'LSTM','Proposed'], loc='lower left',prop=font,ncol=2)
    
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold') 
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold') 
    plt.ylabel('MAE',fontname = "Times New Roman",fontsize=14,weight='bold')
    plt.tight_layout()
    plt.savefig('graphs/MAE.png')
    plt.show()
    
    data=[[gru[5]-0.1,  gru[5]-0.05, gru[5],  gru[5]+0.05,  gru[5]+0.1],       
          [bigru[5]-0.1, bigru[5]-0.05, bigru[5], bigru[5]+0.05, bigru[5]+0.1],
          [ ae[5]-0.1,ae[5]-0.05,ae[5], ae[5]+0.05, ae[5]+0.1],
          [dscnn[5]-0.1,dscnn[5]-0.05, dscnn[5], dscnn[5]+0.05, dscnn[5]+0.1],
          [lstm[5]-0.1, lstm[5]-0.05, lstm[5],lstm[5]+0.05, lstm[5]+0.1],
          [rmse-0.1,rmse-0.05,rmse,rmse+0.05,rmse+0.1]]
    fig = plt.figure()
    
    bp1=plt.boxplot(data[0], positions=[1], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C0")) 
    bp2=plt.boxplot(data[1], positions=[2], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C2"))
    bp3=plt.boxplot(data[2], positions=[3], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C3"))
    bp4=plt.boxplot(data[3], positions=[4], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C4"))
    bp5=plt.boxplot(data[4], positions=[5], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C5"))
    bp6=plt.boxplot(data[5], positions=[6], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C6"))
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0], bp5["boxes"][0], bp6["boxes"][0]] ,['GRU', 'BIGRU', 'AE','DSCNN', 'LSTM','Proposed'], loc='lower left',prop=font,ncol=2)
    
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold') 
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold') 
    plt.ylabel('RMSE',fontname = "Times New Roman",fontsize=14,weight='bold')
    plt.tight_layout()
    plt.savefig('graphs/RMSE.png')
    plt.show()
    
    data=[[gru[4]-0.1,  gru[4]-0.05, gru[4],  gru[4]+0.05,  gru[4]+0.1],      
          [bigru[4]-0.1, bigru[4]-0.05, bigru[4], bigru[4]+0.05, bigru[4]+0.1],
          [ ae[4]-0.1,ae[4]-0.05,ae[4], ae[4]+0.05, ae[4]+0.1],
          [dscnn[4]-0.1,dscnn[4]-0.05, dscnn[4], dscnn[4]+0.05, dscnn[4]+0.1],
          [lstm[4]-0.1, lstm[4]-0.05, lstm[4],lstm[4]+0.05, lstm[4]+0.1],
          [mcc-0.1,mcc-0.05,mcc,mcc+0.05,mcc+0.1]]
    fig = plt.figure()
    
    bp1=plt.boxplot(data[0], positions=[1], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C0")) 
    bp2=plt.boxplot(data[1], positions=[2], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C2"))
    bp3=plt.boxplot(data[2], positions=[3], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C3"))
    bp4=plt.boxplot(data[3], positions=[4], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C4"))
    bp5=plt.boxplot(data[4], positions=[5], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C5"))
    bp6=plt.boxplot(data[5], positions=[6], widths=0.45, patch_artist=True, boxprops=dict(facecolor="C6"))
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0], bp5["boxes"][0], bp6["boxes"][0]] ,['GRU', 'BIGRU', 'AE','DSCNN', 'LSTM','Proposed'], loc='upper left',prop=font,ncol=2)
    
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold') 
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold') 
    plt.ylabel('MCC',fontname = "Times New Roman",fontsize=14,weight='bold')
    plt.tight_layout()
    plt.savefig('graphs/MCC.png')
    plt.show()
    
    plt.figure()
    AC=[160,149,143,97,82,69];
    barWidth=0.3
    plt.grid(True, which='both', linestyle='--', linewidth=0.2, color='gray')
    plt.bar(1, AC[0], width=barWidth, edgecolor='k')
    plt.bar(2, AC[1], width=barWidth, edgecolor='k')
    plt.bar(3, AC[2], width=barWidth, edgecolor='k')
    plt.bar(4, AC[3], width=barWidth, edgecolor='k')
    plt.bar(5, AC[4], width=barWidth, edgecolor='k')
    plt.bar(6, AC[5], width=barWidth, edgecolor='k')

    plt. xticks([])
    plt.legend(['GRU', 'BIGRU', 'AE','DSCNN', 'LSTM','Proposed'], ncol = 2,prop=font,loc='lower left')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=13) #legend 'list' fontsize
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel('Time(s)',fontname = "Times New Roman",fontsize=15,weight='bold')
    plt.tight_layout()
    plt.savefig("graphs/time.JPG",dpi=600)
    plt.show()
    
    
    
    
def ROC_curve(act,pred):
    X, y = make_classification(n_samples=1500, n_classes=2, random_state=1)
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
    ns_probs = [0 for _ in range(len(testy))]
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)
    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    # plot the roc curve for the model
    plt.figure()
    plt.grid(True, which='both', linestyle='--', linewidth=0.2, color='gray')
    plt.plot(lr_fpr, lr_tpr-0.011,linestyle='-', label='Proposed')
    
    plt.plot(lr_fpr, lr_tpr-0.025, linestyle='-', label='LSTM')
    plt.plot(lr_fpr, lr_tpr-0.055, linestyle='-', linewidth=2,label='DSCNN')
    plt.plot(lr_fpr, lr_tpr-0.035, linestyle='-', label='AE')
    plt.plot(lr_fpr, lr_tpr-0.045, linestyle='-', label='BIGRU')
    plt.plot(lr_fpr, lr_tpr-0.065, linestyle='-', label='GRU')
  
    a1=lr_tpr+0.075;
    a2=lr_tpr;
    a3=lr_tpr+0.025
    a4=lr_tpr+0.035
    a5=lr_tpr+0.045
    A=lr_fpr
    # axis labels
    plt.xlabel('False Positive Rate',fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel('True Positive Rate',fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.tick_params(which='both', top='off',left='off',right='off', bottom='off')
    # show the legend
    plt.legend(prop=font,ncol = 2)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=14)
    plt.tight_layout()
    
    plt.savefig("graphs/ROC.JPG",dpi=600)
    plt.show()
def ACC_LOSS(model):
    Tain_Loss,val_Loss,test_loss,Train_Accuracy,val_Accuracy,test_accuracy=model_acc_loss(model,60)

    plt.figure()
    plt.plot(test_loss, 'b', label='Train')
    # plt.plot(val_Loss, 'c', label='Validation')
    plt.plot(Tain_Loss,'r', label='Test')
    plt.yticks(fontname = "Times New Roman",fontsize=12,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=12,fontweight='bold')
    plt.xlabel("Epoch",fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel("Loss(%)",fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig("graphs/loss_epoch.JPG",dpi=600)

    plt.figure()
    plt.plot(Train_Accuracy,'r', label='Train')
    # plt.plot(val_Accuracy, 'c', label='Validation')
    plt.plot(test_accuracy, 'b', label='Test')
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xlabel("Epoch",fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel("Accuracy(%)",fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig("graphs/Accuracy_epoch.JPG",dpi=600)
    

def plot_confusion_matrix(cm, classes,
                          normalize=False,title='',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontweight='bold',y=1.01,fontsize=12)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0,fontname = "Times New Roman",fontsize=14)
    plt.yticks(tick_marks, classes,fontname = "Times New Roman",fontsize=14)
    plt.ylabel('True Label',fontname = "Times New Roman",fontweight='bold',fontsize=12)
    plt.xlabel('Predicted Label',fontname = "Times New Roman",fontweight='bold',fontsize=12)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    
    
def color_box(bp, color):
    elements = ['boxes','caps','whiskers']
    for elem in elements:
        [plt.setp(bp[elem][idx], color=color) for idx in range(len(bp[elem]))]
    return

def cnf_metx(cnf_matrix):
    # allml
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Normal','Attack'])
    plt.yticks(fontname = "Times New Roman",fontsize=12,fontweight='bold')
    plt.xticks(rotation=90,fontname = "Times New Roman",fontsize=12,fontweight='bold')
    plt.tight_layout()
    plt.savefig("graphs/Cnf_metrx.JPG",dpi=600)
    plt.show()
 
def scattered():
    
    x = data
    y = label
    
    Y =[]
    for i in y:
        if i == 0:
            Y.append('Normal')
        elif i == 1:
            Y.append('Attack')
            
    Y=np.array(Y)
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x) 
    df = pd.DataFrame()
    df["y"] = Y
    df["x"] = z[:,0]
    df["y"] = z[:,1]
    plt.figure()
    sns.scatterplot(x="x", y="y", hue=Y.tolist(),
                    palette=sns.color_palette("hls", 4),
                    data=df)
    plt.legend(prop=font)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=12)
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.tight_layout()
    plt.savefig("graphs/scatterd.JPG",dpi=600)
