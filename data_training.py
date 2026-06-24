import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

# Initialize variables
is_init = False
size = -1
label = []
dictionary = {}
c = 0

# Get all .npy files, excluding labels.npy
directory = os.getcwd()
npy_files = [f for f in os.listdir(directory) if f.endswith('.npy') and f != 'labels.npy']
if not npy_files:
    raise ValueError("No .npy files found in the current directory!")

# Load data and labels
for i in npy_files:
    data = np.load(i)
    
    if data.ndim != 2:
        raise ValueError(f"Data in {i} must be a 2D array (samples, features)")
    
    if not is_init:
        is_init = True
        X = data
        size = X.shape[0]
        y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
    else:
        if data.shape[1] != X.shape[1]:
            raise ValueError(f"Feature mismatch in {i}: Expected {X.shape[1]}, but got {data.shape[1]}")
        X = np.concatenate((X, data))
        y = np.concatenate((y, np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)))

    label.append(i.split('.')[0])
    dictionary[i.split('.')[0]] = c
    c += 1

# Convert labels to integers
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]

y = np.array(y, dtype="int32")

# One-hot encode labels
y = to_categorical(y)

# Shuffle data
np.random.seed(42)
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

X_new = X[cnt]
y_new = y[cnt]

# Build the model
ip = Input(shape=(X.shape[1],))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)

# Compile and train the model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X_new, y_new, epochs=50, batch_size=32)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))

print("Training complete. Model and labels saved.")

# Label dictionary output
for key, value in dictionary.items():
    print(f"{key} = {value}")