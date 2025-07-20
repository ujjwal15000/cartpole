import random
from collections import deque

import gym
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'

opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
random.seed(0)
tf.random.set_seed(0)


class Agent:
    def __init__(self, model=None, weights=None):
        self.epsilon = 1.0  # exploration percentage
        self.epsilon_decay = 0.9975  # exploration decay
        self.epsilon_min = 0.1  # exploration min
        self.memory = deque(maxlen=2000)  # previous memory
        self.batch_size = 32  # training batch
        self.gamma = 0.9  # future reward discount (adding future prediction's reward to curr reward with some grain of salt)
        self.treward = []  # list of rewards
        self.max_treward = 0
        self.env = gym.make("CartPole-v1", render_mode="human")
        # self.env = gym.make("CartPole-v1")
        if model:
            self.model = model
        elif weights:
            self.model = load_model(weights)
        else:
            self.model = self._create_model()

    def _create_model(self, weights = None):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, activation="relu", input_dim=4))
        model.add(tf.keras.layers.Dense(24, activation="relu"))
        model.add(tf.keras.layers.Dense(2, activation="linear"))
        model.compile(loss="mse", optimizer=opt)
        if weights:
            model.set_weights(weights)
        return model

    def crossover(self, other_agent):
        child = self._create_model()

        # 0 -> this, 1 -> other
        for i in range(len(child.layers)):
            w = random.choice([0, 1])
            if w == 0:
                child.layers[i].set_weights(self.model.layers[i].get_weights())
            else:
                child.layers[i].set_weights(other_agent.model.layers[i].get_weights())

        return Agent(child)

    # def mutate(self, mutation_rate):
    #     weights = self.model.get_weights()
    #     new_weights = []
    #     for w in weights:
    #         mutation = np.random.randn(*w.shape) * mutation_rate
    #         w_new = w + mutation
    #         new_weights.append(w_new)
    #
    #     self.model.set_weights(new_weights)

    def mutate(self, mutation_rate):
        weights = self.model.get_weights()
        for i in range(len(weights)):
            if random.random() < mutation_rate:
                weights[i] += random.random()
        return weights

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state.reshape(1, 4), verbose=0)[0])

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, done in batch:
            if not done:
                reward += self.gamma * np.amax(self.model.predict(next_state.reshape(1, 4), verbose=0)[0])
            target = self.model.predict(state.reshape(1, 4), verbose=0)
            target[0, action] = reward
            self.model.fit(state.reshape(1, 4), target, epochs=2, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes):
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            for f in range(1, 5000):
                action = self.act(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                self.memory.append([state, action, next_state, reward, done])
                state = next_state
                if done or trunc:
                    self.treward.append(f)
                    self.max_treward = max(self.max_treward, f)
                    # print(f"max reward: {self.max_treward}, current frame reached: {f}")
                    break
            if len(self.memory) > self.batch_size:
                self.replay()

    def test(self, episodes):
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            for f in range(1, 5000):
                action = np.argmax(self.model.predict(state.reshape(1, 4))[0])
                next_state, reward, done, trunc, _ = self.env.step(action)
                state = next_state
                if done or trunc:
                    self.treward.append(f)
                    self.max_treward = max(self.max_treward, f)
                    print(f"max reward: {self.max_treward}")
                    break


agent = Agent()
agent.train(1500)

