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
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
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
        self.memory = deque(maxlen=9000)
        self.m_state = [] 
        self.m_action = []
        self.m_reward = []
        self.m_info = []

        self.omega = 0.3
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.q_history = 20
        self.q_step = 2
        self.q_lenght = 7
        self.info_lenght = 40
        self.actions = deque(maxlen=self.info_lenght)
        for i in range(self.info_lenght):
            self.actions.append(np.zeros(self.q_lenght))
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
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
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

        actions_input = Input(shape=(self.info_lenght, self.q_lenght, 1))
        x2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='m2block1_conv1')(actions_input)
        x2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='m2block1_conv2')(x2)
        x2 = MaxPooling2D((2, 2), strides=(2, 2), name='m2block1_pool')(x2)

        # Block 2
        x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='m2block2_conv1')(x2)
        x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='m2block2_conv2')(x2)
        x2 = MaxPooling2D((2, 2), strides=(2, 2), name='m2block2_pool')(x2)

        x2 = Flatten(name='m2flatten')(x2)
        x2 = Dense(64, activation='relu', name='m2fc2')(x2)
        right_branch = Model(actions_input, x2, name='m2_vgg19')

        merged = Merge([left_branch, right_branch], mode='concat')

        final_model = Sequential()
        final_model.add(merged)
        final_model.add(Dense(24, activation='relu', name='mfc1'))
        final_model.add(Dense(self.q_lenght, activation='softmax', name='mfc2'))

        final_model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])
        # print(final_model.inputs)
        # exit()
        return final_model

    def remember(self, state, action, reward, next_state, done, info):
        self.m_state.append(state) 
        self.m_action.append(action) 
        self.m_reward.append(reward) 
        self.m_info.append(info) 

    def q2buttons(self, q_vector):
        m = np.argmax(q_vector)
        buttons = np.zeros(self.action_size)
        buttons[m + 4] = 1
        return buttons

    def act(self, state, prev_q_vector, reward, pos_x):
        if np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_size)
            m = random.randrange(self.q_lenght)
            q_vector = np.zeros(self.q_lenght)
            q_vector[m] = 1
        else:
            # info = self.info_vector(prev_q_vector, reward, pos_x)
            # print('info vector: {}'.format(info))

            state = np.array([state])
            actions = np.array(self.actions)
            actions = actions.reshape((1, self.info_lenght, self.q_lenght, 1))
            q_vector = self.model.predict([state, actions])[0]
            self.actions.append(q_vector)
            # print(q_vector)
        return q_vector

    def info_vector(self, q_vector, reward, pos_x):
        vector = np.zeros(self.info_lenght)
        vector[0] = reward
        vector[1] = pos_x
        vector[2:] = q_vector[:]

        return np.reshape(vector, (1, self.info_lenght))

    def replay(self):
        x_1 = []
        x_2 = []
        y = []
        leng = len(self.m_state) - self.info_lenght
        self.q = np.zeros((leng, self.q_lenght))
        
        # place X
        for i in range(leng):
            x_1.append(self.m_state[i])
            actions = np.array(self.m_action[i:i + self.info_lenght])
            actions = actions.reshape((self.info_lenght, self.q_lenght, 1))
            x_2.append(actions)
        x_1 = np.array(x_1)
        x_2 = np.array(x_2)
        # print('x_1 shape: {} '.format(x_1.shape))
        # print('x_2 shape: {} '.format(x_2.shape))

        # recalculate Q for all memory
        local_summ = 0
        for i in range(leng, 0):
            state = np.array([self.m_state[i]])
            actions = np.array(x_2[i])
            local_q = self.model.predict([state, actions])[0]  
            m = np.argmax(self.m_action[i])
            local_q[m] += self.omega * self.m_reward[i]

            start_history = i + 1
            end_history = start_history + self.q_history
            if start_history >= leng:
                break
            if end_history >= leng:
                end_history = leng-1

            for j in range(start_history, end_history):
                state = np.array([self.m_state[j]])
                actions = np.array(x_2[i])
                local_q[m] += j*0.1 * self.gamma * np.amax(self.model.predict([state, actions])[0])

            self.q[i] = local_q

        self.model.fit([x_1, x_2], self.q, epochs=2, verbose=0, batch_size=16)
        self.m_state = [] 
        self.m_action = []
        self.m_reward = []
        self.m_info = []
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        self.model.save_weights(self.model_name + '.hdf5')

    def load_model(self):
        self.model.load_weights(self.model_name + '.hdf5')
        self.model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])
