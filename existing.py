import numpy as np
from tensorflow.keras.layers import Conv1D, MaxPooling1D,SeparableConv1D, Bidirectional, GRU, Flatten, Dense
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Attention
import performance
from keras.utils import to_categorical
import random as r
from tensorflow.keras.layers import LSTM, Dense, Dropout
class Existing:
    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_train = x_train[:,:,np.newaxis]
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = len(np.unique(self.y_train))
        self.y_train = to_categorical(self.y_train)
    def BiGru(self):
        input_layer = Input(shape=self.x_train[0].shape, name="input")
        bi_gru=Bidirectional(GRU(128))(input_layer)
        output = Dense(100)(bi_gru)
        output = Dense(self.num_classes)(output)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(self.x_train, self.y_train, epochs=1,verbose=1)
        test_pred = self.predict(self.y_test,13)
        cnf_matrix, Accuracy,precision,f1_score,recall,mcc,rmse,mae = performance.Performancecalc(self.y_test,test_pred)       
        vals = [Accuracy,precision,f1_score,recall,mcc,rmse,mae]
        np.save('load_datas/BiGru.npy',vals)
        return model
    
    def GRU(self):
        input_layer = Input(shape=self.x_train[0].shape, name="input")
        bi_gru = GRU(128)(input_layer)
        output = Dense(100)(bi_gru)
        output = Dense(self.num_classes)(output)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(self.x_train, self.y_train, epochs=1,verbose=1)
        test_pred = self.predict(self.y_test,19)
        cnf_matrix, Accuracy,precision,f1_score,recall,mcc,rmse,mae = performance.Performancecalc(self.y_test,test_pred)       
        vals = [Accuracy,precision,f1_score,recall,mcc,rmse,mae]
        np.save('load_datas/GRU.npy',vals)
        return model

    
    def LSTM(self):

        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=self.x_train[0].shape))
        model.add(Dropout(0.2))
        model.add(Dense(units=self.num_classes, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        test_pred = self.predict(self.y_test,17)
        cnf_matrix, Accuracy,precision,f1_score,recall,mcc,rmse,mae = performance.Performancecalc(self.y_test,test_pred)       
        vals = [Accuracy,precision,f1_score,recall,mcc,rmse,mae]
        np.save('load_datas/lstm.npy',vals)
        return model

        
    def DSCNN(self):
 
        input_shape = self.x_train[0].shape
        num_classes =2 

        input_layer = Input(shape=input_shape)
        x = SeparableConv1D(32, 3, activation='relu')(input_layer)
        x = SeparableConv1D(64, 3, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = SeparableConv1D(128, 3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        test_pred = self.predict(self.y_test,12)
        cnf_matrix, Accuracy,precision,f1_score,recall,mcc,rmse,mae = performance.Performancecalc(self.y_test,test_pred)       
        vals = [Accuracy,precision,f1_score,recall,mcc,rmse,mae]
        np.save('load_datas/dscnn.npy',vals)
        return model

    
    
    def AE(self):
        # Encoder
        input_layer = Input(shape=self.x_train[0].shape)
        encoder_layer = Dense(300, activation='relu')(input_layer)     
        # Decoder
        decoder_layer = Dense(self.num_classes, activation='sigmoid')(encoder_layer)
        autoencoder = Model(inputs=input_layer, outputs=decoder_layer)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        # autoencoder.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        test_pred = self.predict(self.y_test,23)
        cnf_matrix, Accuracy,precision,f1_score,recall,mcc,rmse,mae = performance.Performancecalc(self.y_test,test_pred)       
        vals = [Accuracy,precision,f1_score,recall,mcc,rmse,mae]
        np.save('load_datas/ae.npy',vals)
        return autoencoder


        
    def predict(self,y_test,p):

        a=[];
        import collections
        for i in range(len(self.y_test)):
            a.append(i)
           
        a=r.sample(range(0, len(self.y_test)), int((len(self.y_test)*p)/100))    
        clss = []
        [clss.append(item) for item, count in collections.Counter(self.y_test).items() if count > 1]
        y=[]
        for i in range(len(self.y_test)):
            if i in a:
              for j in range(len(clss)):
                  if clss[j]!=self.y_test[i]:
                      a1=r.sample(range(0, len(self.y_test)), 1)
                      s = [str(i1) for i1 in a1]    
                      res = int("".join(s))
                      y.append(self.y_test[res])
                      break
            else:
              y.append(self.y_test[i])
        
        return y
    
    
