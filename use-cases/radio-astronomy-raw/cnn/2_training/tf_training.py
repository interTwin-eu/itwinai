import sys

import os

import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

from config import PATH2FILES, PATH4IMAGES, PATH4CHECKPOINTS, FILENAME, LABELS 
from models import models_htable


def get_callback(resol):
    checkPointPath = os.path.join(
	f'{PATH4CHECKPOINTS}',
	f'ch_point_{modelname}', 
	'prot-{epoch:03d}-{accuracy:.3f}-{val_accuracy:.3f}.keras')

    valAccuracyCheckPointCallBack = ModelCheckpoint(
        checkPointPath,
        mode='max', 
        monitor='accuracy',
        save_freq='epoch',
        save_weights_only=False,
        save_best_only=True,
        verbose=1
    )

    valVal_AccuracyCheckPointCallBack = ModelCheckpoint(
        checkPointPath,
        mode='max',
        monitor='val_accuracy',
        save_freq='epoch',
        save_weights_only=False,
        save_best_only=True,
        verbose=1
    )


    return [valAccuracyCheckPointCallBack, valVal_AccuracyCheckPointCallBack]


def label_encoding(labels):
    map_dict = {
        'None': 0,
        'Pulse': 1,
        'BBRFI': 2,
        'NBRFI': 3,
        'Pulse+BBRFI': 4,
        'Pulse+NBRFI': 5,
        'NBRFI+BBRFI': 6,
        'Pulse+NBRFI+BBRFI': 7
    }
    
    return np.array([map_dict[i] for i in labels])


resol = int(sys.argv[1])
modelname = sys.argv[2]


data = np.load(f'{PATH2FILES}{FILENAME}')
labels = np.load(f'{PATH2FILES}{LABELS}')
print("loaded")

data = data[..., tf.newaxis]
labels = label_encoding(labels)
print("encoded")

x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
print("split")

model = models_htable[modelname](resol)
callback = get_callback(resol)
opt = Adam(learning_rate=1E-4)

num_epoch = 100
model.compile(optimizer=opt,
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("compiled")

history = model.fit(
    x_train,
    y_train,
    epochs=num_epoch, 
    validation_data=(x_val, y_val),
    callbacks=callback
)
print("fitted")
plt.clf()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'Model Loss: {resol}x{resol}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, ls='--')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plotting accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(f'Model Accuracy: {resol}x{resol}')
plt.ylabel('Accuracy')
plt.grid(True, ls='--')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.savefig(
    f'{PATH4IMAGES}accuracy_accros_epoch_for_{modelname}_{resol}x{resol}.jpg',
    format='jpg',
    dpi=300
)

test_loss, test_acc = model.evaluate(x_val, y_val, verbose=2)

