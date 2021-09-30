import pandas as pd
from keras.models import Sequential
from keras.layers import *

def keras_run(X, Y, in_epochs = 1000, inactivation = 'sigmoid', ininitializer='random_normal') :

    model = Sequential ()
    model.add ( Dense ( 1, input_dim= 8,
                        kernel_initializer = ininitializer,
                        activation = inactivation ) )
    model.compile ( loss='mean_absolute_error', optimizer='adam' )
    # Train the model
    model.fit (
        X,
        Y,
        epochs=in_epochs,
        shuffle=False,
        verbose=2
    )
    return model

def keras_model_predict(model, X_predict) :
    prediction = model.predict ( X_predict )
    prediction = prediction[0][0]
    print ( 'KERAS Got: ', prediction, ' Expect: ', X_predict[0][1] )



