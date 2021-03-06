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
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Merge
from keras.engine.topology import get_source_inputs


class CNQAgent:
    def __init__(self, action_size, model_name):
        self.model_name = model_name
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
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
        nb_classes = 4

        # model = Sequential()
        img_input = Input(shape=(224,320,3))
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        x = Flatten(name='flatten')(x)
        # x = Dense(64, activation='relu', name='fc1')(x)
        x = Dense(64, activation='relu', name='fc2')(x)
        # x = Dense(nb_classes, activation='softmax', name='predictions')(x)
        left_branch = Model(img_input, x, name='vgg19')

        info_input = Input(shape=(5,))
        x2 = Dense(12, activation='relu', name='ifc1')(info_input)
        x2 = Dense(12, activation='relu', name='ifc2')(x2)
        right_branch = Model(info_input, x2, name='info_net')

        merged = Merge([left_branch, right_branch], mode='concat')

        final_model = Sequential()
        final_model.add(merged)
        final_model.add(Dense(24, activation='relu'))
        final_model.add(Dense(nb_classes, activation='softmax'))

        final_model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])
        # print(final_model.inputs)
        # exit()
        return final_model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def q2buttons(self, q_vector):
        m = np.argmax(q_vector)
        buttons = np.zeros(self.action_size)
        buttons[m + 4] = 1
        return buttons

    def act(self, state, prev_action, reward):
        if np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_size)
            m = random.randrange(4)
            q_vector = np.zeros(4)
            q_vector[m] = 1
        else:
            info = self.info_vector(prev_action, reward)
            # print('info vector: {}'.format(info))
            q_vector = self.model.predict([state, info])[0]
            # print(q_vector)
        return self.q2buttons(q_vector)

    def info_vector(self, action, reward):
        info_vector = np.zeros(5)
        info_vector[0:4] = action[4:8]
        info_vector[4] = reward
        return np.reshape(info_vector, (1,5))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            info = self.info_vector(action, reward)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict([next_state, info])[0])
            target_f = self.model.predict([state, info])
            
            # print('KEEEK: {}'.format(action))
            m = np.argmax(action) - 4

            target_f[0][m] = target

            self.model.fit([state, info], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        self.model.save_weights(self.model_name + '.hdf5')

    def load_model(self):
        self.model.load_weights(self.model_name + '.hdf5')
