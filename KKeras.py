import pandas as pd
from keras.models import Sequential
from keras.layers import *


def keras_run(X, Y) :

    # Define the model
    model = Sequential ()
    model.add ( Dense ( 1, input_dim=8, kernel_initializer='uniform', activation='sigmoid' ) )

    model.compile ( loss='mean_absolute_error', optimizer='adam' )
    # Train the model
    model.fit (
        X,
        Y,
        epochs=1000,
        shuffle=False,
        verbose=2
    )
    return model



def keras_model_predict(model, X_predict) :
    prediction = model.predict ( X_predict )
    # Grab just the first element of the first prediction (since that's the only have one)
    prediction = prediction[0][0]
    print ( 'KERAS Got: ', prediction, ' Expect: ', X_predict[0][1] )



