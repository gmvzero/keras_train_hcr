from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

if __name__ == '__main__':
    data_dir = Path('../Hi_Ka3/')
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        featurewise_center=False,
        featurewise_std_normalization=False
    )

    total = 0
    for image in train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=(64, 64),
        shuffle=False,
        save_to_dir='../Augment/',
        class_mode='binary',
        save_prefix='N',
        save_format='jpeg'
    ):
        print(total)
        total += 1

