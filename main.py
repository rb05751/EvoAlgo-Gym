from env_configs import env_details
import json


if __name__ == '__main__':
    for ENV_NAME in list(env_details.keys())[-1:]:
        print(f"Testing for {ENV_NAME}")
        config_params = env_details[ENV_NAME]
        config_params['env_config_params']['G'] = 10

        evolver = config_params['env_evolver'](model=config_params['env_model'],
                                               config_parameters=config_params['env_config_params'])

        best_models = evolver.evolve()

        # Store best models in json file for later loading
        file_path = f'best_models_{ENV_NAME}.json'

        best_models = [{'weights': mod.weights.tolist(),
                        'avg_reward': mod.avg_reward,
                        'std_reward': mod.std_reward} for mod in best_models]

        # Writing to a json file
        with open(file_path, 'w') as file:
            json.dump(best_models, file, indent=4)
