
import numpy as np
from math import floor
import matplotlib.pyplot as plt
from PIL import Image
from urllib.request import urlretrieve

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D


def get_data(download=True, unzip=True, load=True, loadupto=6000):
    """'Cocodataset stuff' is downloaded, unziped, and loaded to numpy array."""

    from os import path, makedirs, getcwd, listdir
    from zipfile import ZipFile

    filename = 'cocodataset'
    zdirname = path.join(getcwd(), filename + '.zip')
    fdirname = path.join(getcwd(), filename)

    # Download data
    if download is True:
        if path.exists(zdirname):
            print('Ignored download: Cannot download file that already exists!')
        else:
            url = 'http://images.cocodataset.org/zips/val2017.zip'
            filename = 'cocodataset'
            urlretrieve(url, zdirname)
            print('* File downloaded')

    # Unzip data
    if unzip is True:
        if path.exists(fdirname):
            print('Ignored unzip: Cannot unzip file that already exists!')
        else:
            dirname = path.join(getcwd(), filename)
            makedirs(dirname, exist_ok=True)
            with ZipFile(filename + '.zip', 'r') as zipfile:
                zipfile.extractall(fdirname)
            print('* File unzipped')

    # Load data
    if load is True:
        images = []
        shapes = []
        i = 1
        for fdir in listdir(fdirname):
            for file in listdir(path.join(fdirname, fdir)):
                fpath = path.join(*[fdirname, fdir, file])
                image = Image.open(fpath)
                image = np.array(image)     # Convert to numpy array
                shape = np.array(image.shape)
                if len(shape) < 3:  # Skip black and white images
                    continue
                shapes.append(shape)
                images.append(image)
                if i >= loadupto:
                    break
                i += 1

        # Trim all images to a the smallest size
        shapes = np.array(shapes)
        min_shape = np.amin(shapes, axis=0)
        min_shape = (124, 124, 3)
        data_imgs = np.array([image[: min_shape[0], : min_shape[1], : min_shape[2]] for image in images])
        print('* Loaded {0} images to numpy array'.format(i))

        return data_imgs


# -------- Setup --------#
split = (0.6, 0.2, 0.2)     # (% for training , % for validating, % for testing)    Must sum to 1.

# Parameters
n_batch = 32        # Number of images trained at a time
n_epochs = 40       # Number of training sessions

# Download, unzip, and load data
data = get_data(download=False, unzip=False, load=True)

# Scale data
data = data / 255   # From RGB scale 0 -> 255 to normalized scale 0 -> 1

# Create noisy input data
x_data = data + np.random.normal(loc=0, scale=0.5, size=data.shape)     # Normally distributed around 0 w/ sd of 0.5

# Create normal output data
y_data = data

# Split data into sets
x_train = x_data[0: int(split[0]*x_data.shape[0]/n_batch)*n_batch, ...]
y_train = y_data[0: int(split[0]*y_data.shape[0]/n_batch)*n_batch, ...]

x_valid = x_data[int(split[0]*x_data.shape[0]/n_batch)*n_batch:
                 int((split[0] + split[1])*x_data.shape[0]/n_batch)*n_batch, ...]
y_valid = y_data[int(split[0]*x_data.shape[0]/n_batch)*n_batch:
                 int((split[0] + split[1])*x_data.shape[0]/n_batch)*n_batch, ...]

x_test = x_data[int((split[0] + split[1])*x_data.shape[0]/n_batch)*n_batch:
                floor(x_data.shape[0]/n_batch)*n_batch, ...]
y_test = y_data[int((split[0] + split[1])*x_data.shape[0]/n_batch)*n_batch:
                floor(x_data.shape[0]/n_batch)*n_batch, ...]


# -------- Create model -------- #
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', padding='same',
                 input_shape=(data.shape[1], data.shape[2], data.shape[3]), data_format='channels_last'))
model.add(AveragePooling2D((2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(AveragePooling2D((2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(3, (1, 1), activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adadelta')

print(model.summary())


# -------- Train model -------- #
history = model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid),
                    epochs=n_epochs, batch_size=n_batch, shuffle=True)
model.save_weights('model_weights.h5')

# Plot training process
plt.plot(history.history['loss'], linewidth=2, label='Training')
plt.plot(history.history['val_loss'], linewidth=2, label='Validation')
plt.legend(loc='upper right')
plt.title('Denoising Filter Training: Loss vs Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


# # -------- Load model weights -------- #
# model.load_weights('model_weights.h5')


# -------- Test model -------- #
y_test_act = model.predict(x=x_test, batch_size=64)
for i in range(y_test_act.shape[0]):
    plt.subplot(221)        # Enable subplot of (2 columns, 2 rows, 1st item)
    plt.title('Input (distorted image)')
    plt.xticks([])          # Remove axis ticks
    plt.yticks([])          # Remove axis ticks
    x = 255 * x_test[i]     # Scale
    x = x.astype(int)       # Remove decimals
    plt.imshow(x)           # Plot image
    plt.subplot(223)
    plt.title('Output (regenerated image)')
    plt.xticks([])
    plt.yticks([])
    yy = 255 * y_test_act[i]
    yy = yy.astype(int)
    plt.imshow(yy)
    plt.subplot(224)
    plt.title('Truth (original image)')
    plt.xticks([])
    plt.yticks([])
    y = 255 * y_test[i]
    y = y.astype(int)
    plt.imshow(y)
    plt.tight_layout()
    plt.show()


