import os, sys
import zipfile
import random
import numpy as np
import tensorflow as tf
import nibabel as nib

from tensorflow import keras
from tensorflow.keras import layers

from scipy import ndimage
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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

train_paths = [ os.path.join("/scratch1/rgoli/BET_BSE_DATA/files", x) for x in os.listdir("/scratch1/rgoli/BET_BSE_DATA/files")]

train_names = [ x.split('_')[0] for x in os.listdir("/scratch1/rgoli/BET_BSE_DATA/files")]
orig_scan_paths = [ os.path.join("/scratch1/rgoli/IXI-T1", x)+".nii.gz" for x in train_names]

print("MRI scans with Skull-Stripping: " + str(len(train_paths)))
print("MRI scans with Original: " + str(len(orig_scan_paths)))

with open('mapping.csv', 'w') as f:
    for i in range(len(train_paths)):
        print(train_paths[i],'\t',orig_scan_paths[i], file=f)  

def preprocessing(volumeX,volumeY):
    """Process data by only adding a channel."""
    volumeX = tf.expand_dims(volumeX, axis=3)
    volumeY = tf.expand_dims(volumeY, axis=3)
    return volumeX, volumeY

# trainX = []
# for path in tqdm(train_paths):
#     trainX.append(process_scan(path))
# trainX = np.array(trainX)
    
# with open('trainX.npy', 'wb') as f:
#     np.save(f, trainX)  
    
# trainY = []
# for path in tqdm(orig_scan_paths):
#     trainY.append(process_scan(path))
# trainY = np.array(trainY)
# print(len(trainY))

# with open('trainY.npy', 'wb') as f:
#     np.save(f, trainY)
    
trainX=np.load('brain_scans_np.npy')
trainY=np.load('trainY.npy')

print(trainX.shape)
print(trainY.shape)

x_train, x_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.10, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 1
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(preprocessing)
    .batch(batch_size)
    .prefetch(1)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(preprocessing)
    .batch(batch_size)
    .prefetch(1)
)

Input_img = keras.Input(shape=(128, 128, 64,1))
                        
x1 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(Input_img)
x2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x1)
x2 = layers.MaxPool3D( (2, 2, 2))(x2)
encoded = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x2)

# decoding architecture
x3 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(encoded)
x3 = layers.UpSampling3D((2, 2, 2))(x3)
x2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x3)
x1 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x2)
decoded = layers.Conv3D(3, (3, 3, 3), padding='same')(x1)

autoencoder = keras.Model(Input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()
# print(Input_img.shape)
# sample = train_dataset.take(1)
# print(type(sample))
# print(list(sample)[0][0].shape)
# print(list(sample)[0][1].shape)


# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "MRI_reconstruct_v2.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

autoencoder.fit(
    train_dataset,
    validation_data=validation_dataset,
    steps_per_epoch=30,
    validation_steps=5,
    epochs=30,
    shuffle=True,
    verbose=1,
    use_multiprocessing=True,
    workers=40,
    callbacks=[checkpoint_cb, early_stopping_cb ],
)

#     steps_per_epoch=(len(x_train) // batch_size),