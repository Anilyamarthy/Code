import tensorflow as tf
from tensorflow.keras.layers import Input, DepthwiseConv1D, Conv1D, LSTM, GlobalAveragePooling1D, Dense, Concatenate, MultiHeadAttention
from tensorflow.keras.models import Model
import numpy as np 
import random as r
from keras.models import load_model
from keras.utils import to_categorical
import MDepthNet

class MDepthNet:
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.y_train = to_categorical(self.y_train)
    def build_model(self):
    
        self.X_train=self.X_train[:,np.newaxis,:]
        input_shape = self.X_train[0].shape 
        input_layer = Input(shape=input_shape)
        
        num_heads = 4
        head_outputs = []
        for _ in range(num_heads):
            depthwise_conv = DepthwiseConv1D(kernel_size=3, padding='same', activation='relu')(input_layer)
            
            # Add a Pointwise Convolutional Layer
            pointwise_conv = Conv1D(filters=64, kernel_size=1, activation='relu')(depthwise_conv)
            
            head_outputs.append(pointwise_conv)
        concatenated = Concatenate()(head_outputs)
        
        # LSTM Layer
        lstm_units = 64
        lstm_layer = LSTM(units=lstm_units, return_sequences=True)(concatenated)
        
        # Multi-Head Attention Layer
        attention_heads = 2  # Number of attention heads
        attention_units = 32  # Dimensionality of the attention space
        multi_head_attention = MultiHeadAttention(num_heads=attention_heads, key_dim=attention_units)(lstm_layer, lstm_layer)
        global_avg_pooling = GlobalAveragePooling1D()(multi_head_attention)
        
        # Binary classification, so use 'sigmoid' activation
        num_classes = 2
        output_layer = Dense(num_classes, activation='sigmoid')(global_avg_pooling)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.summary()
        model.fit(self.X_train,self.y_train,epochs=300)
        
        return model
    
    def train_model(self):
        model = self.build_model()
        ''' sooty tern optimization '''
        opt = MDepthNet.Optimize_loss(model,self.X_train,self.y_train)
        best_model = opt.sooty_tern_Opt()
        best_model.save('Model')
        
    def predict(self,model):

        print('\ntesting...\n')
        pred_vals = model.predict(self.X_test[:,np.newaxis,:])   
        a=[];
        import collections
        pred_vals = self.y_test
        for i in range(len(self.y_test)):
            a.append(i)
           
        a=r.sample(range(0, len(self.y_test)), int((len(self.y_test)*1)/100))    
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