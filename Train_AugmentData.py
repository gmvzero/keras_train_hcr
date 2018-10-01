from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense
import json
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--data', type=str, required=True,
                help="Path of input dataset")
ap.add_argument('--model', type=str, required=True,
                help="Path of model to fine-tuning")
ap.add_argument('--save_model', type=str, default='./new_model.model',
                help="Path of model to save after fine-tuning")
ap.add_argument('--nb_class', type=int, required=True,
                help='The number of class in data dir')
args = vars(ap.parse_args())

if __name__ == '__main__':
    # Load model
    model = load_model(args['model'])

    # Get the #of class from the directory dataset
    data_dir = Path(args['data'])

    # Change the last dense layer softmax with a #of class
    model.layers.pop()
    for layer in model.layers:
        layer.trainable = False
    prediction = Dense(units=args['nb_class'], kernel_initializer="he_normal",
                       activation="softmax")(model.layers[-1].output)
    new_model = Model(inputs=model.input, outputs=prediction)

    epochs = 30
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
        featurewise_center=False,
        featurewise_std_normalization=False
    )

    datagen = train_datagen.flow_from_directory(
        directory=args['data'],
        target_size=(img_rows, img_cols),
        shuffle=False
    )

    # Save the label of index
    label_file = open('label.txt', 'w')
    label_file.write(json.dumps(datagen.class_indices))

    opt = Adam(lr=learning_rate, decay=learning_rate / epochs)
    new_model.compile(loss="binary_crossentropy", optimizer=opt,
                      metrics=["accuracy"])

    new_model.fit_generator(
        generator=datagen,
        epochs=epochs,
        use_multiprocessing=True,
        steps_per_epoch=len(datagen)
    )

    new_model.save(args['save_model'])
