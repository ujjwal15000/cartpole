import random
import time

import gym
import gym_snake
import matplotlib as plt
import os
import numpy as np
from sequential_model import SequentialModel
from gym.wrappers import TimeLimit
from gym.envs.classic_control import CartPoleEnv

os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)


class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.max_steps = 5000  # Set the max steps to 5000
        self.render_mode = render_mode
        self.viewer = None

    def step(self, action):
        # Perform the usual step
        state, reward, done, info, _ = super().step(action)

        # Modify the angle limits to allow the pole to dip below the horizon
        x, x_dot, theta, theta_dot = state
        if abs(theta) > np.pi:  # 180 degrees
            done = False  # Continue the episode even if the pole dips below the horizon

        # Custom logic to end episode (if any) can be added here
        # if self.current_step >= self.max_steps:
        #     done = True

        return state, reward, done, info, _


class Agent:
    def __init__(self, model=None, weights=None, render=False):
        self.k_crossover = 3  # num crossover points
        self.max_treward = 0
        self.env = gym.make("snake-v0", render_mode="human") if render else gym.make("snake-v0")
        # self.env = TimeLimit(self.env.env, max_episode_steps=5000)
        # self.env = CustomCartPoleEnv(render_mode="human") if render else CustomCartPoleEnv()
        if model:
            self.model = model
        elif weights:
            self.model = SequentialModel.load(weights)
        else:
            self.model = self._create_model()

    @staticmethod
    def _create_model(weights=None):
        model = SequentialModel()
        model.add_layer(32, 4, "ReLU")
        model.add_layer(64, activation_function="ReLU")
        model.add_layer(64, activation_function="ReLU")
        model.add_layer(2, activation_function="Softmax")

        if weights:
            SequentialModel.load(weights)
        return model

    # k point crossover
    def crossover(self, other_agent):
        child1 = self._create_model()
        child2 = self._create_model()

        # 0 -> this, 1 -> other
        for i in range(len(self.model.layers)):
            weights_curr, biases_curr = self.model.layers[i].get_weights()
            weights_other, biases_other = other_agent.model.layers[i].get_weights()
            n_units = self.model.layers[i].units
            if n_units - 2 == 0:
                child1.layers[i].weights[:, 0] = weights_curr[:, 0]
                child1.layers[i].weights[:, 1] = weights_other[:, 1]
                child2.layers[i].weights[:, 0] = weights_other[:, 0]
                child2.layers[i].weights[:, 1] = weights_curr[:, 1]
                child1.layers[i].biases[:, 0] = biases_curr[:, 0]
                child1.layers[i].biases[:, 1] = biases_other[:, 1]
                child2.layers[i].biases[:, 0] = biases_other[:, 0]
                child2.layers[i].biases[:, 1] = biases_curr[:, 1]
                continue

            random_unit_array = np.arange(1, n_units - 1)  # not picking from both extremes
            picked_units = np.random.choice(random_unit_array, size=self.k_crossover, replace=False)
            prev_k = 0
            curr_pick = 0
            for curr_k in picked_units:
                child1.layers[i].weights[:, prev_k:curr_k] = weights_curr[:,
                                                             prev_k:curr_k] if curr_pick == 0 else weights_other[:,
                                                                                                   prev_k:curr_k]

                child2.layers[i].weights[:, prev_k:curr_k] = weights_curr[:,
                                                             prev_k:curr_k] if curr_pick == 1 else weights_other[:,
                                                                                                   prev_k:curr_k]

                child1.layers[i].biases[:, prev_k:curr_k] = biases_curr[:,
                                                            prev_k:curr_k] if curr_pick == 0 else biases_other[:,
                                                                                                  prev_k:curr_k]

                child2.layers[i].biases[:, prev_k:curr_k] = biases_curr[:,
                                                            prev_k:curr_k] if curr_pick == 1 else biases_other[:,
                                                                                                  prev_k:curr_k]
                prev_k = curr_k
                curr_pick = not curr_pick

        return Agent(child1), Agent(child2)

    def random_layer_crossover(self, other_agent):
        child = self._create_model()

        # 0 -> this, 1 -> other
        for i in range(len(child.layers)):
            w = random.choice([0, 1])
            if w == 0:
                weights = self.model.layers[i].get_weights()
                child.layers[i].set_weights(*weights)
            else:
                weights = other_agent.model.layers[i].get_weights()
                child.layers[i].set_weights(*weights)

        return Agent(child)

    def average_weight_crossover(self, other_agent):
        child = self._create_model()

        for i in range(len(child.layers)):
            weights_curr = self.model.layers[i].get_weights()
            weights_other = other_agent.model.layers[i].get_weights()
            child.layers[i].set_weights((weights_curr[0] + weights_other[0]) / 2,
                                        (weights_curr[1] + weights_other[1]) / 2)

        return Agent(child)

    def mutate(self, mutation_rate, mutation_strength):
        if random.random() > mutation_rate:
            return
        weights = self.model.get_weights()
        for layer in weights:
            # if random.random() < mutation_rate:
            layer.weights = layer.weights + (mutation_strength * np.random.randn(*layer.weights.shape))
            layer.biases = layer.biases + mutation_strength * np.random.randn(*layer.biases.shape)

    def train(self, episodes):
        rewards = []
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            for f in range(1, 10000):
                # action = self.act(state)
                action = np.argmax(self.model.predict(state.reshape(1, 4))[0])
                next_state, reward, done, trunc, _ = self.env.step(action)
                state = next_state
                if done or trunc:
                    rewards.append(f)
                    # self.max_treward = max(self.max_treward, f)
                    # print(f"max reward: {self.max_treward}, current frame reached: {f}")
                    break
        self.max_treward = np.mean(rewards)

    def test(self, episodes):
        rewards = []
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            for f in range(1, 10000):
                action = np.argmax(self.model.predict(state.reshape(1, 4))[0])
                next_state, reward, done, trunc, _ = self.env.step(action)
                state = next_state
                if done or trunc:
                    self.max_treward = max(self.max_treward, f)
                    print(f"max reward: {self.max_treward}")
                    rewards.append(self.max_treward)
                    break
        return rewards


class GeneticAlgo:
    def __init__(self, pop_size=10, render=False):
        self.pop_size = pop_size
        self.mutation_rate = 0.01
        self.mutation_strength = 0.5
        self.population = []
        self.render = render
        self._init_pop()

    def _init_pop(self):
        for _ in range(self.pop_size):
            self.population.append(Agent(render=self.render))

    def train_population(self, episodes):
        for p in self.population:
            p.train(episodes)

    def crossover(self, gen):
        new_pop = []
        sorted_pop = sorted(self.population, key=lambda x: x.max_treward, reverse=True)
        sorted_pop = sorted_pop[0: int(self.pop_size / 4)]

        print(f"best rewards in generation {gen}: {[x.max_treward for x in sorted_pop]}")
        if sorted_pop[0].max_treward > 4500:
            rewards = Agent(sorted_pop[0].model, render=False).test(5)

            if np.mean(rewards) > 4500:
                Agent(sorted_pop[0].model, render=False).test(1)
                time.sleep(2)


        total_fitness = sum(p.max_treward for p in sorted_pop)

        for i in range(len(sorted_pop)):
            r1 = random.uniform(0, total_fitness)
            r2 = random.uniform(0, total_fitness)

            j = 0
            p1 = 0
            p2 = 0

            cumulative_fitness = 0
            while j < len(sorted_pop):
                cumulative_fitness += sorted_pop[j].max_treward
                if r1 <= cumulative_fitness:
                    p1 = j
                    break
                j += 1

            j = 0
            cumulative_fitness = 0
            while j < len(sorted_pop):
                cumulative_fitness += sorted_pop[j].max_treward
                if r2 <= cumulative_fitness:
                    p2 = j
                    break
                j += 1

            child1, child2 = sorted_pop[p1].crossover(sorted_pop[p2])
            child1.mutate(self.mutation_rate, self.mutation_strength)
            child2.mutate(self.mutation_rate, self.mutation_strength)
            new_pop.append(child1)
            new_pop.append(child2)

            mutated_parent = Agent(sorted_pop[i].model.copy())
            mutated_parent.mutate(self.mutation_rate, self.mutation_strength)

            new_pop.append(mutated_parent)

        for p in sorted_pop:
            p.max_treward = 0
            new_pop.append(p)

        self.population = new_pop

    def train_and_populate(self, episodes):
        gen = 0
        while True:
            self.train_population(episodes)
            self.crossover(gen)
            gen += 1


# algo = GeneticAlgo(50)
# algo.train_and_populate(episodes=5)




env = gym.make("snake-v0")
env.reset()

while True:
    action = env.action_space.sample()
    env.render()
    next_state, reward, done, trunc =  env.step(action)
    print(reward)
    if done or trunc:
        print("end: " + str(reward))
        env.reset()