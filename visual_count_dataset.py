from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--data', type=str, required=True,
                help="Path of input dataset")
args = vars(ap.parse_args())

NB_MAX_IMAGE = 2766


def barplot(x_data, y_data, error_data, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    # Draw bars, position them in the center of the tick mark on the x-axis
    ax.bar(x_data, y_data, color='#539caf', align='center')
    # Draw error bars to show standard deviation, set ls to 'none'
    # to remove line between points
    ax.errorbar(x_data, y_data, yerr=error_data, color='#297083', ls='none', lw=2, capthick=2)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    # plt.show()


if __name__ == '__main__':
    # Get the #of class from the directory dataset
    data_dir = Path(args['data'])

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

    for child in data_dir.iterdir():
        nb_label = len(list(child.glob('*')))
        nb_increase_each_img = int((NB_MAX_IMAGE - nb_label)/ nb_label)

        for img_path in child.glob('*'):
            total = 0
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            for batch in train_datagen.flow(x,
                                            batch_size=1,
                                            save_to_dir=child,
                                            save_prefix='J',
                                            save_format='jpeg'):
                total += 1
                if total > nb_increase_each_img:
                    break
