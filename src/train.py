# Import all the necessary libraries
import datetime
import os
import sys

import keras
import numpy as np
import skimage.io  # Used for imshow function
import skimage.transform  # Used for resize function
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Dropout, Lambda
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT, IMG_WIDTH = (256, 256)
IMG_CHANNELS = 3
BATCH_SIZE = 16



def get_training_data(path, output_shape=(None, None)):
    """
    Loading images from train path into a numpy array
    :param path:
    :param output_shape:
    :return:
    """

    # img_paths = ['{0}/{1}/images/{1}.png'.format(path, id) for id in os.listdir(path)]

    img_paths = [f"{path}/{id}/images/{id}.png" for id in os.listdir(path)]
    X_data = np.array([skimage.transform.resize(skimage.io.imread(path)[:, :, :3],
                                                output_shape=output_shape, mode='constant',
                                                preserve_range=True) for path in img_paths],
                      dtype=np.uint8)  # take only 3 channels/bands

    return X_data


def get_train_data_labels(path, output_shape=(None, None)):
    """
    Get training data labels
    Loading and concatenating masks into a numpy array
    :param path:
    :param output_shape:
    :return:
    """

    # img_paths = [glob.glob('{0}/{1}/masks/*.png'.format(path, id)) for id in os.listdir(path)]
    img_paths = [f"{path}/{id}/masks/*.png" for id in os.listdir(path)]

    Y_data = []
    for i, img_masks in enumerate(
            img_paths):  # loop through each individual nuclei for an image and combine them together
        masks = skimage.io.imread_collection(
            img_masks).concatenate()  # masks.shape = (num_masks, img_height, img_width)
        mask = np.max(masks, axis=0)  # mask.shape = (img_height, img_width)
        mask = skimage.transform.resize(mask, output_shape=output_shape + (1,), mode='constant',
                                        preserve_range=True)  # need to add an extra dimension so mask.shape = (img_height, img_width, 1)
        Y_data.append(mask)
    Y_data = np.array(Y_data, dtype=np.bool)

    return Y_data


def dice_coef(y_true, y_pred):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    Dice's coefficient measures how similar a set and another set are
    :param y_true:
    :param y_pred:
    :return: Dice
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    return 1 - dice_coef(y_true, y_pred)


def contracting_path(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """

    :param x:
    :param filters:
    :param kernel_size:
    :param padding:
    :param strides:
    :return:
    """
    c = Conv2D(filters, kernel_size, activation='elu', kernel_initializer='he_normal', padding=padding)(x)
    c = Dropout(0.1)(c)
    c = Conv2D(filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding=padding)(c)
    p = MaxPooling2D((2, 2))(c)
    return c, p


def expansive_path(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    """

    :param x:
    :param skip:
    :param filters:
    :param kernel_size:
    :param padding:
    :param strides:
    :return:
    """

    u = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
    u = concatenate([u, skip])
    c = Conv2D(filters, kernel_size, activation='elu', kernel_initializer='he_normal', padding=padding)(u)
    c = Dropout(0.2)(c)
    c = Conv2D(filters, kernel_size, activation='elu', kernel_initializer='he_normal', padding=padding)(c)
    return c


def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """
    
    :param x: 
    :param filters: 
    :param kernel_size: 
    :param padding: 
    :param strides: 
    :return: 
    """
    c = Conv2D(filters, kernel_size, activation='elu', kernel_initializer='he_normal', padding=padding)(x)
    c = Dropout(0.3)(c)
    c = Conv2D(filters, kernel_size, activation='elu', kernel_initializer='he_normal', padding=padding)(c)
    return c


def UNet(inputs):
    """

    :param inputs:
    :return:
    """
    f = [16, 32, 64, 128, 256]

    p0 = Lambda(lambda x: x / 255)(inputs)

    c1, p1 = contracting_path(p0, f[0])
    c2, p2 = contracting_path(p1, f[1])
    c3, p3 = contracting_path(p2, f[2])
    c4, p4 = contracting_path(p3, f[3])

    bn = bottleneck(p4, f[4])

    u1 = expansive_path(bn, c4, f[3])
    u2 = expansive_path(u1, c3, f[2])
    u3 = expansive_path(u2, c2, f[1])
    u4 = expansive_path(u3, c1, f[0])

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)

    model = keras.models.Model(inputs, outputs)
    print(model.summary())

    return model, outputs


def data_augmentation(input, augment=False):
    # data augmentation
    if augment:
        data_gen_args = dict(shear_range=0.5, rotation_range=50, zoom_range=0.2,
                             width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
    else:
        data_gen_args = dict()
    datagen = ImageDataGenerator(**data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods
    datagen.fit(input, augment=True, seed=42)

    return datagen.flow(input, batch_size=BATCH_SIZE, shuffle=True, seed=42)


def train_model(X_train, Y_train, ):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    model, outputs = UNet(inputs)
    model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
    print(model.summary())

    earlystopper = EarlyStopping(patience=5, verbose=1)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    model_name = f"{timestamp}.h5"
    checkpointer = ModelCheckpoint(f'{model_name}', verbose=1, save_best_only=True)

    size_train = len(X_train)
    val_cutoff = int(size_train * 0.9)
    X_train2 = X_train[:val_cutoff]
    Y_train2 = Y_train[:val_cutoff]
    X_val = X_train[val_cutoff:]
    Y_val = Y_train[val_cutoff:]

    # combine generators into one which yields image and masks
    train_generator = zip(data_augmentation(X_train2, augment=True), data_augmentation(Y_train2, augment=True))
    test_generator = zip(data_augmentation(X_val), data_augmentation(Y_val))

    model.fit_generator(train_generator,
                                  validation_data=test_generator,
                                  epochs=5,
                                  steps_per_epoch=len(X_train) / (BATCH_SIZE * 2),
                                  validation_steps=BATCH_SIZE / 2,
                                  callbacks=[earlystopper, checkpointer])


def process_cmd_args():
    """
    Reads CMD args.
    source_folder_name :name of the folder that would be used as a source folder
    destination_folder_name: name of the folder that would be used as a destination folder

    :return: tuple(logo_file_name, destination_folder_name, local_temp_folder, source_folder_name)
    """
    if len(sys.argv) < 2:
        print("Not enough cmd arguments.")

    train_path = sys.argv[1]

    return train_path


def main():
    train_path = process_cmd_args()

    print("Loading images from train path")
    X_train = get_training_data(train_path, output_shape=(IMG_HEIGHT, IMG_WIDTH))

    print("Loading and concatenating masks")
    Y_train = get_train_data_labels(train_path, output_shape=(IMG_HEIGHT, IMG_WIDTH))

    print("Train model")
    train_model(X_train, Y_train, )
    return None


if __name__ == '__main__':
    main()
