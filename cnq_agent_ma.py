# -*- coding: utf-8 -*-
"""
Агент с мультидействиями
"""

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
import random
import numpy as np

# Conv Neural Q-learning Agent
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


class CNQAgent:
    def __init__(self, action_size, model_name):
        self.model_name = model_name
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # input image dimensions
        img_rows, img_cols = 224, 320
        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        nb_pool = 2
        # convolution kernel size
        nb_conv = 3
        nb_classes = 12

        model = Sequential()
        model.add(Conv2D(32, nb_conv, nb_conv,
                         border_mode='valid',
                         input_shape=(img_rows, img_cols, 3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
        model.add(Conv2D(64, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
        model.add(Conv2D(128, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.2))
        model.add(Conv2D(256, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(2, size=self.action_size)
        act_values = self.model.predict(state)
        return act_values[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            
            for i in range(12):
                if action[i]>0:
                    target_f[0][i] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        self.model.save_weights(self.model_name + '.hdf5')

    def load_model(self):
        self.model.load_weights(self.model_name + '.hdf5')
