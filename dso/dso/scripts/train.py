import numpy as np
from dso import DeepSymbolicRegressor
import time

# You can disable all debugging logs.
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

#Disables TensorFlow 2.x behaviorso
#tf.compat.v1.disable_v2_behavior()

import warnings
warnings.filterwarnings('ignore')

# start clock
start_time = time.clock()

# Display tf version
print("tensorflow version:", tf.__version__)

# Generate some data
np.random.seed(0)
X = np.random.random((10, 2))
y = np.sin(X[:,0]) + X[:,1] ** 2

# Create the model
model = DeepSymbolicRegressor() # Alternatively, you can pass in your own config JSON path

# Fit the model
model.fit(X, y) # Should solve in ~10 seconds

# View the best expression
print(model.program_.pretty())

# Make predictions
model.predict(2 * X)

# print walltime
print("wall time:", time.clock() - start_time)