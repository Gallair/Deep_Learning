import numpy as np
import itertools
import random
import zipfile
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

def plot_decision_boundaries(model, X,y):
  """
  Plots the decision boundaries of a classification model
  """
  # Define the axis boundaries of the plot and create a meshgrid
  x_min, x_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
  y_min, y_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1

  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  # Create predicted values
  x_in = np.c_[xx.ravel(), yy.ravel()] # stacks 2D array together

  # Makes predictions
  y_pred = model.predict(x_in)

  # Check for multyclass

  if len(y_pred[0]) > 1:
    print("Doing multiclass classification")
    # change predictions to multiclass
    y_pred = np.argmax(y_pred, axis = 1).reshape(xx.shape)
  else:
    print("Doing binary Classification")
    y_pred = np.round(y_pred).reshape(xx.shape)

  # Plot the boundary
  plt.contourf(xx,yy,y_pred, cmap=plt.cm.RdYlBu, alpha = 0.7)
  plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())

def make_confusion_matrix(y_true, y_pred, classes = None, cm_title = "Confusion Matrix", figsize = (10,10), text_size = 15):
  cm = confusion_matrix(y_true, tf.round(y_pred))
  cm_norm = cm.astype("float")/cm.sum(axis=1)[:, np.newaxis]
  n_classes = cm.shape[0]

  # let's draw it
  fig, ax = plt.subplots(figsize = figsize)

  # Create matrix plot
  cax = ax.matshow(cm, cmap = plt.cm.Blues)
  fig.colorbar(cax)

  # Creat classes
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label axes
  ax.set(title = cm_title,
        xlabel = "Predicted label",
        ylabel ="True label",
        xticks =np.arange(n_classes),
        yticks =np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)

  # Set x-axis labels to bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.set_ticks_position("bottom")

  # Adjust label size
  ax.yaxis.label.set_size(text_size)
  ax.xaxis.label.set_size(text_size)
  ax.title.set_size(text_size)

  # Set treshhold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)",
            horizontalalignment = "center",
            color ="white" if cm[i,j] > threshold else "black",
            size = text_size)

def plot_random_image(model, images, true_labels, classes):
  """
  picks a random image, plots and labels it with a prediction and truth label
  """

  # set the random integer
  i = random.randint(0, len(images))

  # Create predictions and targets
  target_image = images[i]
  pred_probs = model.predict(target_image.reshape(1,28,28))
  pred_label = classes[pred_probs.argmax()]
  true_label = classes[true_labels[i]]

  # Plot image
  plt.imshow(target_image, cmap=plt.cm.binary)

  # Change the color of the titles depending if prediction is right or wrong
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"

  # add xlabel information (prediction/true label)
  plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                   100*tf.reduce_max(pred_probs),
                                                   true_label),
                                                   color = color)

def unzip_file(filename):
  # unzip the downloaded file
  zip_ref = zipfile.ZipFile(filename)
  zip_ref.extractall()
  zip_ref.close()

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """
  epochs = range(len(history.history["loss"]))

  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  # Plot loss
  plt.plot(epochs, loss, label = "training_loss")
  plt.plot(epochs, val_loss, label = "val_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  # plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label = "training_accuracys")
  plt.plot(epochs, val_accuracy, label = "val_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()

# create the function to import an image and resize it to be able to be used with por model
def load_and_prep_image(filename, img_shape = 224):
  """
  Reads an image from filename, turns it in tensor and reshape it to (img_shape,img_shape, color_channel)
  """
  # Read in image
  img = tf.io.read_file(filename)
  # Decode the read file into tensor
  img = tf.image.decode_image(img)
  # Resize the image
  img = tf.image.resize(img, size = [img_shape, img_shape])
  # Rescale the image
  img = img/255.

  return img

def pred_and_plot(model, file_name,class_names):
  """
  Imports image located at file name and make prediction with model and plots the image with predicted class as the title
  """
  # import the target  image and preprocess it
  img = load_and_prep_image(file_name)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis = 0))

  # Get the predicted class
  pred_class = class_names[int(tf.round(pred))]

  # plot the image and predicted class as the title
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);


# Create TensorBoard callback (functionalized because we need to create a new one for each model)
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
  print(f"Saving TensorBoard log for files to: {log_dir}")
  return tensorboard_callback


def create_model(model_url, num_classes = 10):
  """
  Takes a TensorFlow Hub URL and creates a Keras sequential model

  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in the output layer, should be equal to a number of target classes, default 10.

  Returns:
    An uncompiled Keras Sequential model with model_url as a feature extractor layer and Dense output layer with num_classes output neurons.
  """
  # Download the pretrained model and save it as keras layer

  feature_extractor_layer = hub.keras_layer.KerasLayer(model_url,
                                                       trainable = False,
                                                       name = "feature_extractor_layer",
                                                       input_shape=IMAGE_SHAPE+(3,)) # freeze already learned patterns

# Create our own sequential model
  model = tf.keras.Sequential([
      feature_extractor_layer,
      layers.Dense(num_classes, activation = "softmax", name = "output_layer")
  ])
  return model
  
import os  
def walk_through_dir(directory_name):
  for dirpath, dirnames, filenames in os.walk(directory_name):
    print(f"There are {len(dirnames)} directories and {len(filenames)} in the {dirpath}")
