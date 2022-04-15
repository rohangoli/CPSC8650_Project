import os
import zipfile
import random
import numpy as np
import tensorflow as tf
import nibabel as nib

from tensorflow import keras
from tensorflow.keras import layers

from scipy import ndimage
from tqdm import tqdm

print("Is cuda available?", tf.test.is_built_with_cuda())

print("Is GPU available?", tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None))
# print("cuDNN enabled? ", torch.backends.cudnn.enabled)
      
tf.config.list_physical_devices('GPU')
    
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

scan_paths = [ os.path.join("data/BET_BSE_DATA/files", x) for x in os.listdir("data/BET_BSE_DATA/files")]
val_scan_paths = [ os.path.join("/scratch1/rgoli/IXI-T1", x) for x in os.listdir("/scratch1/rgoli/IXI-T1")]

print("MRI scans with Skull-Stripping: " + str(len(scan_paths)))

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


# train_data = np.array([process_scan(path) for path in scan_paths])
train_data = []
for path in tqdm(scan_paths[:]):
    train_data.append(process_scan(path))
    
val_data = []
for path in tqdm(val_scan_paths[:]):
    val_data.append(process_scan(path))

train_data = np.array(train_data)
print(train_data.shape)
train_data = np.expand_dims(train_data, axis=-1)
print(train_data.shape)

val_data = np.array(val_data)
print(val_data.shape)
val_data = np.expand_dims(val_data, axis=-1)
print(val_data.shape)
    
# # Define data loaders.
# train_loader = tf.data.Dataset.from_tensor_slices((train_data))
# val_loader = tf.data.Dataset.from_tensor_slices((val_data))

## Auto-Encoder

Input_img = keras.Input(shape=(128, 128, 64,1))  
    
#encoding architecture
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

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "MRI_reconstruction.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

autoencoder.fit(train_data, train_data,
                epochs=5,batch_size=128,shuffle=True,
                validation_data=(val_data,val_data),
                callbacks=[checkpoint_cb, early_stopping_cb],verbose=1
               )