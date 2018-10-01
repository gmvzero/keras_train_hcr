from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam
import numpy
import cv2
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--data', type=str, required=True,
                help="Path of input dataset")
ap.add_argument('--model', type=str, required=True,
                help="Path of model to fine-tuning")
ap.add_argument('--save_model', type=str, default='./new_model.model',
                help="Path of model to save after fine-tuning")
args = vars(ap.parse_args())

if __name__ == '__main__':
    # Load model
    model = load_model(args['model'])

    # Change the last dense layer softmax with a #of class
    model.layers.pop()
    for layer in model.layers:
        layer.trainable = False

    epochs = 50
    learning_rate = 1e-5
    batch_size = 32
    img_rows, img_cols = 64, 64
    img_channels = 3

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        zca_whitening=True,
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    train_datagen.flow_from_directory(
        directory=args['data_dir'],
        target_size=(img_rows, img_cols),
        shuffle=False
    )

    model.fit_generator(
        generator=train_datagen,
        use_multiprocessing=True,
        steps_per_epoch=len(train_datagen)
    )

    model.save(args['save_model'])
