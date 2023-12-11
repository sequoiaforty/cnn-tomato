import numpy as np
from PIL import Image
import glob

import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt

CLASS_NAMES = ["Bacterial_spot",
               "Early_blight",
               "Healthy",
               "Late_blight",
               "Leaf_mold",
               "Septoria_leaf_spot",
               "Spider_mites",
               "Target_spot",
               "Tomato_mosaic_virus",
               "Tomato_yellow_leaf_curl_virus"]


# Takes as input path to image file and returns
# resized 3 channel RGB image of as numpy array of size (256, 256, 3)
def getPic(img_path):
    return np.array(Image.open(img_path).convert('RGB'))


# Return the images and corresponding labels as numpy arrays
def get_ds(data_path):
    img_paths = list()
    labels_list = list()
    dir_names = CLASS_NAMES

    for dir_name in dir_names:

        # Recursively find all the image files from the path
        for img_path in glob.glob(data_path + "/" + dir_name + "/*"):
            img_paths.append(img_path)
            labels_list.append(dir_names.index(dir_name))

    images = np.zeros((len(img_paths), 256, 256, 3))
    labels = np.array(labels_list)

    # Read and resize the images
    for i, img_path in enumerate(img_paths):
        images[i] = getPic(img_path)

    return images, labels


# Load the train, validation, and test data
train_images, train_labels = get_ds("dataset_split/train")
val_images, val_labels = get_ds("dataset_split/val")
test_images, test_labels = get_ds("dataset_split/test")

print("done loading.")


def show_samples():
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i+1500])
        #plt.xlabel(CLASS_NAMES[train_labels[i]])
    plt.show()


# Model Architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#model.summary()


#Train model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=15,
                    validation_data=(val_images, val_labels))

#Evaluate model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

plt.show()