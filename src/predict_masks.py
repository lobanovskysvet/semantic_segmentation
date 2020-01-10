import datetime


import numpy as np
import pandas as pd
import skimage.io  # Used for imshow function
import skimage.transform  # Used for resize function
from keras.models import load_model
from skimage.morphology import label  # Used for Run-Length-Encoding RLE to create final submission
from train import IMG_HEIGHT, IMG_WIDTH, bce_dice_loss, dice_coef


def rle_encoding(x):
    """
    numpy array of shape (height, width), 1 - mask, 0 - background
    :param x:
    :return: run length as list
    """

    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def upsample(source_folder_name, Y_hat):
    # Upsample Y_hat back to the original X_test size (height and width)
    Y_hat_upsampled = []

    img = skimage.io.imread(source_folder_name)  # read original test image directly from path
    img_upscaled = skimage.transform.resize(Y_hat[0],
                                            (img.shape[0], img.shape[1]),
                                            mode='constant',
                                            preserve_range=True)  # upscale Y_hat image according to original test image
    Y_hat_upsampled.append(img_upscaled)  # append upscaled image to Y_hat_upsampled
    return Y_hat_upsampled


def save_predicted_value(Y_hat, destination_folder_name):
    image = np.array(Y_hat[0][:, :, 0])
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    skimage.io.imsave(fname=f"{destination_folder_name}/{timestamp}.png", arr=image)


def encoding(Y_hat_upsampled):
    # Apply Run-Length Encoding on our Y_hat_upscaled
    new_test_ids = []
    rles = []

    rle = list(prob_to_rles(Y_hat_upsampled[0]))
    rles.extend(rle)
    new_test_ids.extend([1] * len(rle))
    len(new_test_ids)  # note that for each test_image, we can have multiple entries of encoded pixels
    return new_test_ids, rles


def create_submission_df(new_test_ids, rles, destination_folder_name):
    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    sub.to_csv(f"{destination_folder_name}/{timestamp}.csv", index=False)


def predict_value(source_folder_name, destination_folder_name, segmentation_model):
    model_loaded = load_model(f"{segmentation_model}", custom_objects={'dice_coef': dice_coef,
                                                                       'bce_dice_loss': bce_dice_loss})

    # Get test data
    X_predict = np.array([skimage.transform.resize(skimage.io.imread(source_folder_name)[:, :, :3],
                                                   output_shape=(IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                                   preserve_range=True)], dtype=np.uint8)  # take only 3 channels/bands

    # Use model to predict test labels
    Y_hat = model_loaded.predict(X_predict, verbose=1)

    # Save predicted value
    save_predicted_value(Y_hat, destination_folder_name)

    Y_hat_upsampled = upsample(source_folder_name, Y_hat)

    new_test_ids, rles = encoding(Y_hat_upsampled)
    create_submission_df(new_test_ids, rles, destination_folder_name)
