#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import zipfile
import numpy as np
import tensorflow as tf
import nibabel as nib
from tensorflow import keras
from tensorflow.keras import layers
from scipy import ndimage


# In[2]:


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


# In[3]:


scan_paths = [
    os.path.join(os.getcwd(), "../BET_BSE_DATA", x)
    for x in os.listdir("../BET_BSE_DATA")
]

print("Brain scans: " + str(len(scan_paths)))


# In[4]:


import csv
 
# csv file name
filename = "/home/joelkik/DataMining/BET_BSE_DATA/Label_file.csv"
 
# initializing the titles and rows list
fields = []
rfc = set()
nrfc = set()
 
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
     
    # extracting field names through first row
    fields = next(csvreader)
 
    # extracting each data row one by one
    for row in csvreader:
        #print(row)
        if row[1].lower() == 'yes':
            rfc.add(row[0]+'.gz')
        else:
            nrfc.add(row[0]+'.gz')
 
    # get total number of rows
    print("Total no. of rows: %d"%(csvreader.line_num))
 
# printing the field names
print('Field names are:' + ', '.join(field for field in fields))
#print(labels_dict)


# In[5]:


import random

from scipy import ndimage


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


# In[6]:


# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
rfc_scans=[]
nrfc_scans=[]
labels_facial_feature=[]
labels_to_file=[]
for path in os.listdir("/home/joelkik/DataMining/BET_BSE_DATA/files"):
    if path in rfc:
        rfc_scans.append(process_scan("/home/joelkik/DataMining/BET_BSE_DATA/files/"+path))
    else:
        nrfc_scans.append(process_scan("/home/joelkik/DataMining/BET_BSE_DATA/files/"+path))
    #if len(rfc_scans)>20 and len(nrfc_scans)>2:
    #    break
    #labels_facial_feature.append(labels_dict[path])
    #labels_to_file.append(path)


# In[7]:


rfc_scans_np = np.array(rfc_scans)


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(rfc_scans, rfc_scans, test_size=0.20, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=42)


# In[9]:


# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)


# In[13]:


def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""
    print("Encoders")
    #encoding phase
    inputs = keras.Input((width, height, depth, 1))
    print(inputs.shape)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding='same')(inputs)
    print(x.shape)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    print(x.shape)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding='same')(x)
    print(x.shape)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    print(x.shape)
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding='same')(x)
    print(x.shape)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    print(x.shape)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding='same')(x)
    print(x.shape)
    x = layers.MaxPool3D(pool_size=2)(x)
    print(x.shape)
    x = layers.BatchNormalization()(x)


    print("Decoders")
    #decoding phase
    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding='same')(x)
    print(x.shape)
    x = layers.UpSampling3D(size=2)(x)
    x = layers.BatchNormalization()(x)
    print(x.shape)
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding='same')(x)
    print(x.shape)
    x = layers.UpSampling3D(size=2)(x)
    x = layers.BatchNormalization()(x)
    print(x.shape)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding='same')(x)
    print(x.shape)
    x = layers.UpSampling3D(size=2)(x)
    x = layers.BatchNormalization()(x)
    print(x.shape)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding='same')(x)
    print(x.shape)
    x = layers.UpSampling3D(size=2)(x)
    x = layers.BatchNormalization()(x)
    print(x.shape)
    outputs = layers.Conv3D(filters=1, kernel_size=3, activation="relu", padding='same')(x)
    print(outputs.shape)
    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()


# In[ ]:


model.load_weights("/home/joelkik/DataMining/CPSC8650_Project/3d_reconstruct_facial_features.h5")


# In[ ]:


# Compile model.
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_reconstruct_facial_features.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=1,
    callbacks=[checkpoint_cb, early_stopping_cb],
)


# In[ ]:




