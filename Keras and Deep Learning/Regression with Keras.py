# Suppress TensorFlow warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.simplefilter('ignore', FutureWarning)

# Standard imports
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# ==============================
# Load Dataset
# ==============================

filepath = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

# Separate predictors and target
predictors = concrete_data.drop('Strength', axis=1)
target = concrete_data['Strength']

# Normalize predictors
predictors_norm = (predictors - predictors.mean()) / predictors.std()

n_cols = predictors_norm.shape[1]


# ==============================
# Define Regression Model
# ==============================

def regression_model():
    model = Sequential(name="concrete_regression_model")

    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# ==============================
# Train Model
# ==============================

model = regression_model()

model.fit(
    predictors_norm,
    target,
    validation_split=0.3,
    epochs=100,
    verbose=2
)