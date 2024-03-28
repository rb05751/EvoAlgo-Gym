from env_configs import env_details
from base_model_and_evolver import *
import json
import time
import numpy as np
import gymnasium as gym


def evaluate(model, num_episodes, env):
    all_rewards = []

    for i in range(num_episodes):
        episode_reward = 0
        observation, info = env.reset()
        terminated = False
        truncated = False
        episode_rewards = []
        start = time.time()
        while not terminated and not truncated:

            action, probs = model.take_action(observation)

            # print(action, ": " , probs)

            observation, reward, terminated, truncated, info = env.step(action)  # observation is (8,) array

            episode_reward += reward

            episode_rewards.append(reward)

        all_rewards.append({
            'reward': episode_reward,
            'num_time_steps': len(episode_rewards),
            'time(s)': round(time.time()-start, 3)
        })

        print(f"Episode {i+1} Reward: {episode_reward} | Num Time Steps: {len(episode_rewards)}")

    avg_reward = np.mean(list(map(lambda x: x['reward'], all_rewards)))
    std_reward = np.std(list(map(lambda x: x['reward'], all_rewards)))
    return all_rewards, avg_reward, std_reward

if __name__ == '__main__':

    selected_model = Model(weights=np.array(
        [
            [
                5.030276670848435,
                -3.1333584174206006,
                4.557055269131426
            ],
            [
                0.8692526521797317,
                -5.67902280159379,
                -1.242702098628922
            ],
            [
                1.8035919772587587,
                -9.800504386726592,
                -1.1120875407776425
            ],
            [
                1.6673695201566077,
                -11.867717943286863,
                -2.016174290881707
            ],
            [
                6.530255106763017,
                -3.311071243106321,
                -3.6486023440554267
            ],
            [
                -24.329094058509806,
                0.7828715106467402,
                6.816717671204786
            ]
        ]))
    weights = np.array(np.array(selected_model.weights))

    # evaluate
    env = gym.make('Acrobot-v1', render_mode='human')

    results, avg_reward, std_reward = evaluate(model=selected_model, num_episodes=100, env=env)

    print(f"Mean Sample Score: {avg_reward}")
    print(f"Stdev Sample Score: {std_reward}")

    # Store best models in json file for later loading
    file_path = 'acrobot_v1_evaluation.json'

    # Writing to a json file
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)
