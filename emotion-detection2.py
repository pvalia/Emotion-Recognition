#!/usr/bin/env python
# coding: utf-8

# ## Import Modules

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


# ## Load the Dataset

# In[ ]:


TRAIN_DIR = './archive/train/train/'
TEST_DIR = './archive/test/test/'


# In[ ]:


def load_dataset(directory):
    image_paths = []
    labels = []
    
    for label in os.listdir(directory):
        for filename in os.listdir(directory+label):
            image_path = os.path.join(directory, label, filename)
            image_paths.append(image_path)
            labels.append(label)
            
        print(label, "Completed")
        
    return image_paths, labels


# In[ ]:


## convert into dataframe
train = pd.DataFrame()
train['image'], train['label'] = load_dataset(TRAIN_DIR)
# shuffle the dataset
train = train.sample(frac=1).reset_index(drop=True)
train.head()


# In[ ]:


test = pd.DataFrame()
test['image'], test['label'] = load_dataset(TEST_DIR)
test.head()


# ## Exploratory Data Analysis

# In[ ]:


sns.countplot(train['label'])


# In[ ]:


from PIL import Image
img = Image.open(train['image'][0])
plt.imshow(img, cmap='gray');


# In[ ]:


# to display grid of images
plt.figure(figsize=(20,20))
files = train.iloc[0:25]

for index, file, label in files.itertuples():
    plt.subplot(5, 5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')


# ## Feature Extraction

# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.3,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory(TRAIN_DIR,
                                                batch_size=64,
                                                target_size=(48,48),
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(TEST_DIR,
                                                batch_size=64,
                                                target_size=(48,48),
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical')


# In[ ]:


# config
training_set.class_indices
input_shape = (48, 48, 1)
output_class = 7


# ## Model Creation

# In[ ]:


model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(output_class, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

model.summary()

# In[ ]:

# train the model
history = model.fit(x=training_set, batch_size=128, epochs=2, validation_data=test_set)


# ## Plot the Results

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()

plt.show()


# ## Test with Image Data

# In[ ]:


image_index = random.randint(0, len(test))
print("Original Output:", test['label'][image_index])
pred = model.predict(training_set[image_index].reshape(1, 48, 48, 1))
prediction_label = le.inverse_transform([pred.argmax()])[0]
print("Predicted Output:", prediction_label)
plt.imshow(training_set[image_index].reshape(48, 48), cmap='gray');


# In[ ]:


image_index = random.randint(0, len(test))
print("Original Output:", test['label'][image_index])
pred = model.predict(training_set[image_index].reshape(1, 48, 48, 1))
prediction_label = le.inverse_transform([pred.argmax()])[0]
print("Predicted Output:", prediction_label)
plt.imshow(training_set[image_index].reshape(48, 48), cmap='gray');


# In[ ]:


image_index = random.randint(0, len(test))
print("Original Output:", test['label'][image_index])
pred = model.predict(training_set[image_index].reshape(1, 48, 48, 1))
prediction_label = le.inverse_transform([pred.argmax()])[0]
print("Predicted Output:", prediction_label)
plt.imshow(training_set[image_index].reshape(48, 48), cmap='gray');


# In[]:

train_loss, train_acc = model.evaluate(training_set)
testn_loss, test_acc = model.evaluate(test_set)
print("final train accuracy = {:.2f}, validation accuracy = {:.2f}".format(train_acc*100, test_acc*100))


# %%
