import matplotlib

matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import glob
import os
import resnet

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-o", "--output", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size
epochs = 50
learning_rate = 1e-5
batch_size = 32
img_rows, img_cols = 64, 64
img_channels = 3

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
# grab the image paths and randomly shuffle them
name_classes = os.listdir(args["dataset"])
# random.seed(42)
# random.shuffle(imagePaths)
# loop over the input images
imagePaths = glob.glob(os.path.join(args["dataset"], '-a', '*g'))
subdirs = {}
length = len(imagePaths)
for name in name_classes:
    imagePaths = glob.glob(os.path.join(args["dataset"], name, '*g'))
    subdirs[name] = imagePaths
    current_length = len(imagePaths)
    if current_length < length:
        length = current_length
for name in name_classes:
    imagePaths = subdirs.get(name)
    for idx, imagePath in enumerate(imagePaths):
        if idx < length:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (64, 64), cv2.INTER_LINEAR)
            label = name.replace('-', '')
            labels.append(label)
            image = img_to_array(image)
            data.append(image)

labels = np.array(labels)
data = np.array(data, dtype="float") / 255.0
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=52)
testY = to_categorical(testY, num_classes=52)
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2)
# initialize the model
print("[INFO] compiling model...")
model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), 52)
opt = Adam(lr=learning_rate, decay=learning_rate / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)

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
