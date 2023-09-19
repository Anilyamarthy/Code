import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def create_dsae(input_shape):
    # Encoder 1
    input_layer = Input(shape=input_shape)
    encoded1 = Dense(256, activation='relu', activity_regularizer=regularizers.l1(1e-5))(input_layer)
    encoded1 = Dense(128, activation='relu', activity_regularizer=regularizers.l1(1e-5))(encoded1)

    # Encoder 2
    encoded2 = Dense(64, activation='relu', activity_regularizer=regularizers.l1(1e-5))(encoded1)
    
    # Decoder 1
    decoded1 = Dense(128, activation='relu')(encoded2)
    decoded1 = Dense(256, activation='relu')(decoded1)

    # Decoder 2
    decoded2 = Dense(12, activation='sigmoid')(decoded1)

    # Create and compile the DSAE model
    dsae = Model(input_layer, decoded2)
    dsae.compile(optimizer='adam', loss='binary_crossentropy')
    
    return dsae


