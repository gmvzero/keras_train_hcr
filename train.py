import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cnn
matplotlib.use("Agg")


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-tm", "--typemodel", default="number",
                help="type model")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-o", "--output", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

epochs = 50
learning_rate = 1e-5
batch_size = 10
img_rows, img_cols = 64, 64
img_channels = 3
data = []
labels = []
subdirs = {}
num_classes = 10
data_dir = copy_file(args["dataset"], './balanced_data')
model = cnn.implement_cnn(img_cols, img_rows, img_channels, num_classes)
data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                    height_shift_range=0.1, shear_range=0.2, zoom_range=0.1, validation_split=0.2)

# initialize the model
print("[INFO] compiling model...")

opt = Adam(lr=learning_rate, decay=learning_rate / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

train_generator = data_generator.flow_from_directory(
    directory=data_dir,
    target_size=(img_rows, img_cols),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_generator = data_generator.flow_from_directory(
    directory=data_dir,
    target_size=(img_rows, img_cols),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())

with open(os.path.join('../labels.txt'), 'w', encoding="utf-8") as outfile:
    outfile.write(args["typemodel"] + '_labels\n')
    outfile.write(str(labels) + '\n')

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
H = model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=epochs
                        )

model.evaluate_generator(generator=valid_generator
                         )

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Gender Classification")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["output"])
