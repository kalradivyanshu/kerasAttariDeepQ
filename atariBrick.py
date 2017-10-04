from collections import deque
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import convolutional, MaxPooling2D
from keras.optimizers import Adam
import numpy as np
import random
import gym
import cv2
import sys
import pickle

Conv2D = convolutional.Conv2D

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 2000)
        self.gamma = 0.95
        #self.epsilon = 1.0
        #self.epsilon_min = 0.01
        #self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4, 4), input_shape=(160, 210, 1)))
        model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(28, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(15, activation='tanh'))
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        try:
            minibatch = random.sample(self.memory, batch_size)
        except ValueError:
            minibatch = self.memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            #print(target_f)
            self.model.fit(state, target_f, epochs = 1, verbose = 0)

if __name__ == "__main__":
    try:
        env = gym.make('Breakout-v0')

        agent = DQNAgent((1, 160, 210), env.action_space.n)
        episodes = 20000
        rewardL = []
        for e in range(episodes):
            state = env.reset()
            gray_image = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            combinedStates = deque(maxlen = 5)
            combinedStates.append(gray_image)
            state = np.reshape(gray_image, [1, 160, 210, 1])
            #print(state.shape)
            for time_t in range(3000):
                #env.render()
                state = sum(combinedStates)
                #print(state.shape)
                state = np.reshape(state, [1, 160, 210, 1])
                action = agent.act(state)
                if random.random() < 0.05:
                    action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
                combinedStates.append(next_state)
                next_state = sum(combinedStates)
                #print("::", next_state.shape)
                next_state = np.reshape(next_state, [1, 160, 210, 1])
                agent.remember(state, action, reward, next_state, done)
                #state = combinedStates
                if done:
                    break
            print("episode: ", e, "score: ", time_t)
            rewardL.append(time_t)
            agent.replay(32)
    except Exception as ex:
        s = str(ex)
        e = sys.exc_info()[0]
        print(e, s)
        print(str(9))
        pass
    finally:
        print("Saving...")
        agent.model.save_weights('atari.h5')
        pickle.dump(rewardL, open("timeAtari.pickle", "wb"))
        print("Saved.")
