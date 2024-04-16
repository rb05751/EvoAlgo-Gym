from base_model_and_evolver import Model, Evolver
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import time

class BipedalWalkerModel(Model):
    def take_action(self, observation):
        action_logits = np.dot(self.weights.T, observation)

        alpha = 0.5 # for smoothing out tanh
        action = np.tanh(alpha * action_logits)

        return action, action_logits

class PendulumV1Model(Model):
    def take_action(self, observation):
        action_probs = self.softmax(np.dot(self.weights.T, observation))

        action = [np.dot(np.array([-2, -1, 0, 1, 2]), action_probs)]

        return action, action_probs


class CliffWalkingEvolver(Evolver):

    def evaluate_episode(self, model, env_name):
        import gymnasium as gym  # Import gym inside the function

        # This function will be executed in a separate thread for each episode.
        env = gym.make(env_name, render_mode=None)
        # print(f"Evaluating episode for env: {env}")
        observation, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        num_timesteps = 0

        #### Only for Cliff Walking ###
        coord_map = {}
        for i in range(self.config_params['OBS_DIM']):
            for j in range(self.config_params['ACTION_DIM']):
                obs = i * self.config_params['ACTION_DIM'] + j
                coord_map[obs] = [i, j]
        #### Only for Cliff Walking ###

        while not terminated and not truncated and num_timesteps < 100:
            #### Only for Cliff Walking ####
            one_hot_array = [0] * (self.config_params['OBS_DIM']*self.config_params['ACTION_DIM'])
            one_hot_array[observation] = 1
            observation = np.array(one_hot_array)
            #### Only for Cliff Walking ####

            action, probs = model.take_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += float(reward)

            num_timesteps += 1

        env.close()

        ### Only for Cliff Walking ##
        end = coord_map.get(observation)
        dist_to_goal = np.sqrt((end[0] - (self.config_params['OBS_DIM']-1)) ** 2 + (end[1] - (self.config_params['ACTION_DIM']-1)) ** 2)
        max_dist = np.sqrt((0 - (self.config_params['OBS_DIM']-1)) ** 2 + (0 - (self.config_params['ACTION_DIM']-1)) ** 2)
        episode_reward += max(0, max_dist - dist_to_goal) * 100
        # print(end, dist_to_goal, max_dist, observation)
        ### Only for Cliff Walking ###

        return episode_reward

    def evolve(self):
        pop = [self.model(
            weights=np.random.uniform(-1, 1, (self.config_params['OBS_DIM'], self.config_params['ACTION_DIM']))
            ) for _ in range(self.config_params['N'])]
        best_models = []

        for i in range(self.config_params['G']):
            start = time.time()

            # 1. Evaluate each model on task, calculating average reward over episodes
            with Pool() as pool:
                models_with_args = [(model, self.config_params['NUM_EPS_PER_EVAL'], self.config_params['ENV_NAME'], i)
                                    for
                                    i, model in enumerate(pop)]
                results = pool.map(self.evaluate_model, models_with_args)

                for model_index, avg_reward, std_reward in results:
                    pop[model_index].avg_reward = avg_reward
                    pop[model_index].std_reward = std_reward

            # 2. Pick the top K models to reproduce
            most_fit_models = list(sorted(pop, key=lambda x: x.avg_reward - x.std_reward, reverse=True))[
                              :self.config_params['K']]

            # 3. Create next generation of models
            next_gen = self.create_next_generation(most_fit_models, self.config_params['N'],
                                                   self.config_params['OBS_DIM'],
                                                   self.config_params['ACTION_DIM'])

            # 4. Set pop equal to new generation and start over
            best_model = Model(weights=most_fit_models[0].weights)
            best_model.avg_reward = most_fit_models[0].avg_reward
            best_model.std_reward = most_fit_models[0].std_reward
            best_models.append(best_model)

            pop = next_gen

            print(
                f"Best model at end of generation {i + 1}: ({best_models[-1].avg_reward},{best_models[-1].std_reward}): "
                f"Wall-Clock(s) = {time.time() - start}")

        return best_models

class FrozenLakeEvolver(Evolver):

    def evaluate_episode(self, model, env_name):
        import gymnasium as gym  # Import gym inside the function

        # This function will be executed in a separate thread for each episode.
        env = gym.make(env_name, desc=None, map_name=f"{self.config_params['OBS_DIM']}x{self.config_params['ACTION_DIM']}",
                       is_slippery=False, render_mode=None)
        # print(f"Evaluating episode for env: {env}")
        observation, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        num_timesteps = 0

        coord_map = {}
        for i in range(self.config_params['OBS_DIM']):
            for j in range(self.config_params['ACTION_DIM']):
                obs = i * self.config_params['ACTION_DIM'] + j
                coord_map[obs] = [i, j]

        while not terminated and not truncated:
            one_hot_array = [0] * (self.config_params['ACTION_DIM']*self.config_params['OBS_DIM'])
            one_hot_array[observation] = 1
            observation = np.array(one_hot_array)

            action, probs = model.take_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += float(reward)

            num_timesteps += 1

        env.close()

        end = coord_map.get(observation)
        dist_to_goal = np.sqrt((end[0] - (self.config_params['ACTION_DIM']-1)) ** 2 + (end[1] - (self.config_params['ACTION_DIM']-1)) ** 2)
        max_dist = np.sqrt((0 - (self.config_params['ACTION_DIM']-1)) ** 2 + (0 - (self.config_params['ACTION_DIM']-1)) ** 2)
        episode_reward += max(0, max_dist - dist_to_goal) * 100
        # print(end, dist_to_goal, max_dist, observation)

        return episode_reward

    def evolve(self):
        pop = [self.model(
            weights=np.random.uniform(-1, 1, (64, 4))
            ) for _ in range(self.config_params['N'])]
        best_models = []

        for i in range(self.config_params['G']):
            start = time.time()

            # 1. Evaluate each model on task, calculating average reward over episodes
            with Pool() as pool:
                models_with_args = [(model, self.config_params['NUM_EPS_PER_EVAL'], self.config_params['ENV_NAME'], i)
                                    for
                                    i, model in enumerate(pop)]
                results = pool.map(self.evaluate_model, models_with_args)

                for model_index, avg_reward, std_reward in results:
                    pop[model_index].avg_reward = avg_reward
                    pop[model_index].std_reward = std_reward

            # 2. Pick the top K models to reproduce
            most_fit_models = list(sorted(pop, key=lambda x: x.avg_reward - x.std_reward, reverse=True))[
                              :self.config_params['K']]

            # 3. Create next generation of models
            next_gen = self.create_next_generation(most_fit_models, self.config_params['N'],
                                                   self.config_params['OBS_DIM'],
                                                   self.config_params['ACTION_DIM'])

            # 4. Set pop equal to new generation and start over
            best_model = Model(weights=most_fit_models[0].weights)
            best_model.avg_reward = most_fit_models[0].avg_reward
            best_model.std_reward = most_fit_models[0].std_reward
            best_models.append(best_model)

            pop = next_gen

            print(
                f"Best model at end of generation {i + 1}: ({best_models[-1].avg_reward},{best_models[-1].std_reward}): "
                f"Wall-Clock(s) = {time.time() - start}")

        return best_models