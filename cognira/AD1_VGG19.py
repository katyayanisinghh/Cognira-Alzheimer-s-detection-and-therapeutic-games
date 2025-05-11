#!/usr/bin/env python
# coding: utf-8

# # 1. Install Dependencies and Setup

# In[1]:




# # 2. Load Data

# In[2]:


import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import warnings
warnings.filterwarnings(action="ignore")
# import os

# if os.path.exists('/Users/katyayanisingh/Desktop/cognira/model.h5'):
#     print("Model saved successfully.")
# else:
#     print("Model not saved.")
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

# Create output directory
output_dir = '/Users/katyayanisingh/Desktop/cognira'
os.makedirs(output_dir, exist_ok=True)

# Define the model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
# if os.path.exists('/Users/katyayanisingh/Desktop/cognira/model.h5'):
#     print("Model saved successfully.")
# else:
#     print("Model not saved.")

# Save the model AFTER it is defined
model.save(os.path.join(output_dir, 'model.h5'))





# In[3]:


data = tf.keras.utils.image_dataset_from_directory(
    '/Users/katyayanisingh/Desktop/Dataset',
    image_size=(45, 45),
    batch_size=32
)






# In[4]:


data_iterator = data.as_numpy_iterator()


# In[5]:


batch = data_iterator.next()


# In[6]:


#images array
batch[0].shape


# In[7]:


#labels array
batch[1]


# In[8]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# # 3. Scale Data 

# In[9]:


data = data.map(lambda x,y: (x/255, tf.keras.utils.to_categorical(y, 4)))


# In[10]:


scaled_iterator = data.as_numpy_iterator()


# In[11]:


batch = scaled_iterator.next()


# In[12]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx].all())


# ![2.jpg](attachment:2.jpg)

# # 4. Split Data

# In[13]:


len(data)


# In[14]:


train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)


# In[15]:


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# In[16]:


print(len(train))
print(len(val))
print(len(test))


# # 5. Build Deep Learning Model

# In[17]:


model_cnn = Sequential()

model_cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(45, 45, 3)))
model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))

model_cnn.add(Flatten())

model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(4, activation='softmax'))

model_cnn.compile(optimizer=tf.optimizers.Adadelta(), loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])

model_cnn.summary()


# # 6. Train Model

# In[18]:


logdir='logs'


# In[19]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[20]:


hist=model_cnn.fit(train, epochs=100, validation_data=val, callbacks=[tensorboard_callback])


# # 7. Plot Performance

# In[21]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show(block=False)()


# In[22]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show(block=False)()


# # 8. Evaluate

# In[23]:


from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()


# In[24]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model_cnn.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[25]:


print(pre.result(), re.result(), acc.result())


# # 9. Test

# In[26]:


def pred_alza(img):
    resize = tf.image.resize(img, (45,45))
    yhat=model_cnn.predict(np.expand_dims(resize/255, 0))
    id_label = []
    for i in yhat[0]:
        if i < yhat[0].max():
            id_label.append(0)
        else:
            id_label.append(1)

    id_label = id_label
    name_label = ['MildDemented','ModerateDemented','NonDemented', 'VeryMildDemented']
    temp = list(zip(id_label, name_label))
    for i in range(len(temp)):
        if temp[i][0]==1:
            label = temp[i][1]

    return(label) 


# In[27]:


import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/katyayanisingh/Desktop/Dataset/ModerateDemented/26(19).jpg')
label = pred_alza(img)
plt.imshow(img)
plt.title(label)
plt.show(block=False)()


# # 10. Transfer learning

# In[28]:


import tensorflow as tf
from keras.applications import DenseNet201
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

base_model = tf.keras.applications.DenseNet201(
    weights='imagenet',  # Correct path to weights
    include_top=False, 
    input_shape=(45, 45, 3)
)



# In[29]:


x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)  # Add your custom dense layers here

predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False


# In[30]:


# Compile the model
model.compile('adam', loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()


# In[31]:


logdir='logs_resnet'


# In[32]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[33]:


hist_DenseNet=model.fit(train, epochs=100, validation_data=val, callbacks=[tensorboard_callback])


# # 11. Plot Performance 

# In[34]:


fig = plt.figure()
plt.plot(hist_DenseNet.history['loss'], color='teal', label='loss')
plt.plot(hist_DenseNet.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show(block=False)()


# In[35]:


fig = plt.figure()
plt.plot(hist_DenseNet.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist_DenseNet.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show(block=False)()


# # 12. Evaluate 

# In[36]:


from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()


# In[37]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[38]:


print(pre.result(), re.result(), acc.result())


# # 13. Test 

# In[39]:


def pred_alza(img):
    resize = tf.image.resize(img, (45,45))
    yhat=model.predict(np.expand_dims(resize/255, 0))
    id_label = []
    for i in yhat[0]:
        if i < yhat[0].max():
            id_label.append(0)
        else:
            id_label.append(1)

    id_label = id_label
    name_label = ['MildDemented','ModerateDemented','NonDemented', 'VeryMildDemented']
    temp = list(zip(id_label, name_label))
    for i in range(len(temp)):
        if temp[i][0]==1:
            label = temp[i][1]

    return(label) 


# In[40]:


import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/katyayanisingh/Desktop/Dataset/ModerateDemented/26 (19).jpg')
label = pred_alza(img)
plt.imshow(img)
plt.title(label)
plt.show(block=False)()


# # 14. Save the Model

# In[41]:


from tensorflow.keras.models import load_model
import os

model.save('/Users/katyayanisingh/Desktop/cognira/model.h5')




# # 15. Load Model

# In[42]:


from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

#new_model = load_model('image_alz_classifier.h5')


# In[43]:


def pred_alza(img):
    resize = tf.image.resize(img, (45,45))
    yhat=model.predict(np.expand_dims(resize/255, 0))
    id_label = []
    for i in yhat[0]:
        if i < yhat[0].max():
            id_label.append(0)
        else:
            id_label.append(1)

    id_label = id_label
    name_label = ['MildDemented','ModerateDemented','NonDemented', 'VeryMildDemented']
    temp = list(zip(id_label, name_label))
    for i in range(len(temp)):
        if temp[i][0]==1:
            label = temp[i][1]

    return(label) 


# In[44]:


import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/katyayanisingh/Desktop/Dataset/ModerateDemented/26 (2).jpg')
label = pred_alza(img)
plt.imshow(img)
plt.title(label)
plt.show(block=False)()


# In[45]:


import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/katyayanisingh/Desktop/Dataset/ModerateDemented/27 (2).jpg')
label = pred_alza(img)
plt.imshow(img)
plt.title(label)
plt.show(block=False)()


# In[46]:


import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/katyayanisingh/Desktop/Dataset/ModerateDemented/26 (62).jpg')
label = pred_alza(img)
plt.imshow(img)
plt.title(label)
plt.show(block=False)()


# In[47]:


import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/katyayanisingh/Desktop/Dataset/ModerateDemented/26 (44).jpg')
label = pred_alza(img)
plt.imshow(img)
plt.title(label)
plt.show(block=False)()

