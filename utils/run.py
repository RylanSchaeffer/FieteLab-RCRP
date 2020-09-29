import numpy as np
import os.path


def load_env_data(env_str):

    env = np.load(os.path.join(os.getcwd(),
                               'offline_clustering',
                               'envs',
                               f'env_{env_str}.pkl'),
                  allow_pickle=True)
    positions = env['X']
    num_positions = positions.shape[0]
    num_clusters = num_positions

    obs = np.load(os.path.join(os.getcwd(),
                               'offline_clustering',
                               'data',
                               f'obs_{env_str}.pkl'),
                  allow_pickle=True)

    obs_vectors = obs['Obs'].astype(np.int).reshape(num_positions, -1)
    num_parameters = obs_vectors.shape[1]

    load_env_data_results = dict(
        env=env,
        positions=positions,
        num_positions=num_positions,
        obs=obs,
        obs_vectors=obs_vectors,
        num_parameters=num_parameters,
        num_clusters=num_clusters,
    )

    return load_env_data_results