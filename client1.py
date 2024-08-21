import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from imutils import paths
import random
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import flwr as fl
# Load and compile Keras model
from tensorflow.keras.applications import EfficientNetB0

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

BATCH_SIZE = 8
train_data_dir = 'D:/TQ/Federated/1/train/'

test_data_dir = 'D:/TQ/Federated/1/val/'

print("Loading Images..")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Set the validation split percentage
)
test_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)


print("Class Counts in Training Samples:")
total_classes = sorted(os.listdir(train_data_dir))
print(total_classes)
for category in total_classes:
    category_path = os.path.join(train_data_dir, category)
    sample_count = len(os.listdir(category_path))
    print(f"{category}: {sample_count} samples")



train_classes = sorted(os.listdir(train_data_dir))
test_classes = sorted(os.listdir(test_data_dir))

base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the base model layers
base_model.trainable = False

# Add custom classification layers on top of the base model
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Add dropout layer with a dropout rate of 0.5
outputs = Dense(7, activation='softmax')(x)  # Assuming binary classification

# Create the full model
model = Model(inputs, outputs)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
traingen = train_datagen.flow_from_directory(train_data_dir,
                                                   target_size=(225, 225),
                                                   class_mode='categorical',
                                                   classes=total_classes,
                                                   subset='training',
                                                   batch_size=BATCH_SIZE, 
                                                   shuffle=True,
                                                   seed=42)

validgen = train_datagen.flow_from_directory(train_data_dir,
                                               target_size=(225, 225),
                                               class_mode='categorical',
                                               classes=total_classes,
                                               subset='validation',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=42)

testgen = test_datagen.flow_from_directory(test_data_dir,
                                             target_size=(225, 225),
                                             class_mode='categorical',
                                             classes=total_classes,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=42)

results_list = []


# Define Flower client
class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, traingen, validgen, testgen):
        self.model = model
        self.traingen = traingen
        self.validgen = validgen
        self.testgen = testgen
        
        
    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        return model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        callbacks = [early_stopping]

        print("Starting fit..")

        history = model.fit(
            traingen,
            epochs=5,
            steps_per_epoch=len(traingen),
            validation_data=validgen,
            validation_steps=len(validgen),
            callbacks=callbacks,
            verbose = 1
        )
        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.traingen)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        print("Local Training Metrics on client 1: {}".format(results))

        
        y_true = testgen.classes

        # Calculate predictions on the test data
        y_pred = model.predict(testgen)
        y_pred_labels = np.argmax(y_pred, axis=1)  # Get the predicted class labels
        #print(y_pred_labels, y_true)

        # Model Evaluation
        print("Accuracy Before Aggregation")
        test_loss, test_accuracy = self.model.evaluate(testgen, steps=len(testgen), verbose=0)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        # Calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred_labels)
        print("Confusion Matrix Before Aggregation:")
        print(cm)

        results_list.append(results)
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        y_true = testgen.classes
        # Model Evaluation
        test_loss, test_accuracy = self.model.evaluate(testgen, steps=len(testgen), verbose=0)
        print("Accuracy after Aggregation")

        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        num_examples_test = len(self.testgen)
        # Calculate predictions on the test data
        y_pred = model.predict(testgen)
        y_pred_labels = np.argmax(y_pred, axis=1)  # Get the predicted class labels
        #print(y_pred_labels, y_true)

        # Calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred_labels)
        print("Confusion Matrix After Aggregation:")
        print(cm)

        return test_loss, num_examples_test, {"accuracy": test_accuracy}


# Start Flower client
client = FlwrClient(model, traingen, validgen, testgen)
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=client
)
model.save("client1_50_fedavg.h5")
