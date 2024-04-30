import numpy as np
import random
import json
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import time

class Model:
    def __init__(self, weights):
        self.weights = weights
        self.avg_reward = None
        self.std_reward = None

    @staticmethod
    def softmax(x, temperature=1.0):
        e_x = np.exp((x - np.max(x)) / temperature)  # Subtract np.max(x) for numerical stability
        return e_x / e_x.sum()

    def take_action(self, observation):
        action_probs = self.softmax(np.dot(self.weights.T, observation))
        action = np.argmax(action_probs)

        return action, action_probs


class Evolver:
    def __init__(self, model, config_parameters):
        self.model = model
        self.config_params = config_parameters

    @staticmethod
    def softmax(x, temperature=1.0):
        e_x = np.exp((x - np.max(x)) / temperature)  # Subtract np.max(x) for numerical stability
        return e_x / e_x.sum()

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

        while not terminated and not truncated:
            try:
                action, probs = model.take_action(observation)

                observation, reward, terminated, truncated, info = env.step(action)

                episode_reward += float(reward)

                num_timesteps += 1
            except Exception as e:
                print(f"Failed on this action: {action} because {e}")
                break

        env.close()

        return episode_reward

    def evaluate_model(self, args):
        model, num_episodes, env_name, model_index = args

        print("Evaluating Model")

        with ThreadPoolExecutor() as executor:
            # print(f"Evaluating Model: {model_index}")
            future_to_episode = {executor.submit(self.evaluate_episode, model, env_name): i for i in
                                 range(num_episodes)}
            results = [future.result() for future in future_to_episode]

        avg_reward = np.mean(results)
        std_reward = np.std(results)
        return model_index, avg_reward, std_reward

    def create_next_generation(self, most_fit_models, N, obs_dim, action_dim):
        next_gen = []
        for i in range(N // 2):
            # Get model weights for weighted selection
            rewards = [x.avg_reward - x.std_reward for x in most_fit_models]
            model_weights = self.softmax(x=np.array(rewards))
            randomly_selected_model = random.choices(most_fit_models, model_weights, k=1)[0]

            # apply mutation
            mutation_weights = np.random.normal(0, 1, (obs_dim, action_dim))
            weights = randomly_selected_model.weights + mutation_weights

            # append new offspring
            offspring = self.model(weights)
            next_gen.append(offspring)

        next_gen.extend(most_fit_models)  # letting most fit models survive to next generation

        return next_gen

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