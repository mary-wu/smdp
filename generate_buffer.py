import os
import yaml
import copy
import torch
import pickle
import argparse
import wandb

from environment import *
from utils.utils import ExperienceReplay
from ai import AI

floatX = 'float32'


def interact_with_environment(env, replay_buffer, params, ai):

    init_states_dict = {}
    state, done = env.reset(), False
    init_states_dict[state["tabular"]] = init_states_dict.get(state["tabular"], 0) + 1
    state = randomize_init_state(env)
    state = state["image"]

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    num_transitions = 0

    rand_actions = []
    second_actions = []
    max_actions = []

    # Interact with the environment for max_timesteps
    while num_transitions < int(params['replay_max_size']):

        while not done and episode_timesteps < params['max_episode_steps']:

            episode_timesteps += 1

            rand_prob = np.random.uniform(0, 1)
            if rand_prob < params['rand_action_p']:
                action = env.action_space.sample()
                rand_actions.append(action)
            elif rand_prob < (params['2nd_action_p'] + params['rand_action_p']):
                action = ai.get_2ndmax_action(state)
                second_actions.append(action)
            else:
                action = ai.get_max_action(state)[0]
                max_actions.append(action)

            # Perform action and log results
            next_state, reward, done, info = env.step(action)
            next_state = next_state["image"]

            # Log discounted returns (for printing)
            k = len(reward)
            disc_factors = np.power(params['gamma'], range(len(reward)))
            r = np.dot(disc_factors, reward)
            episode_reward += r

            # Only consider "done" if episode terminates due to failure condition
            done_float = float(done) if episode_timesteps < params['max_episode_steps'] else 0

            # Store data in replay buffer
            replay_buffer.add(s=state, a=action, r=r, s2=next_state, term=done_float, k=k)
            state = copy.copy(next_state)

        # Reset environment
        state, done = env.reset(), False
        rand_prob = np.random.uniform(0, 1)
        if rand_prob < params['rand_init']:
            state = randomize_init_state(env)
        init_states_dict[state["tabular"]] = init_states_dict.get(state["tabular"], 0) + 1
        state = state["image"]

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        done = False

        num_transitions += 1

    # Save final buffer and performance
    buffer_folder = f"./buffers/{params['model_name']}_{params['rand_action_p']}_{params['2nd_action_p']}_{params['suffix']}_{params['replay_max_size']}_{params['random_seed']}"
    replay_buffer.save(buffer_folder)
    with open(buffer_folder + "/init_states_dict", 'wb') as handle:
        pickle.dump(init_states_dict, handle)

    print(f"DONE generating buffer. Encountered {len(init_states_dict.keys())} unique initial states.")


def randomize_init_state(env):
    for _ in range(10):
        rand_prob = np.random.uniform(0, 1)
        if rand_prob < 0.33:
            action = env.actions.left
            state, _, _, _ = env.step(action, k_fwd=0)
        elif rand_prob < 0.67:
            action = env.actions.right
            state, _, _, _ = env.step(action, k_fwd=0)
        else:
            action = env.actions.forward
            state, _, _, _ = env.step(action, k_fwd=1)
    return state


def worker(params):

    folder_path = (os.getcwd() + "/models/" + params['model_name']).replace("/", "\\")

    cfg_file = os.path.join(folder_path, 'config.yaml')
    model_params = yaml.safe_load(open(cfg_file, 'r'))

    buffer_folder = f"./buffers/{params['model_name']}_{params['rand_action_p']}_{params['2nd_action_p']}_{params['suffix']}_{params['replay_max_size']}_{params['random_seed']}"
    params['wandb_key'] = None if model_params['wandb_key'] is None or model_params['wandb_key'] == 'None' else model_params['wandb_key']

    if params['wandb_key'] is not None:
        _ = os.system('wandb login {}'.format(params['wandb_key']))
        os.environ['WANDB_API_KEY'] = params['wandb_key']
        params['unique_run_id'] = wandb.util.generate_id()
        wandb.init(id=params['unique_run_id'], resume='allow', project="smdp_toy_example", group=buffer_folder,
                   name=f"seed{params['random_seed']}")
        wandb.config.update(params)

    np.random.seed(model_params['random_seed'])
    random.seed(model_params['random_seed'])
    torch.manual_seed(model_params['random_seed'])
    random_state = np.random.RandomState(model_params['random_seed'])
    device = torch.device(model_params["device"])

    ai = AI(nb_actions=3,
            gamma=model_params['gamma'],
            learning_rate=model_params['learning_rate'], epsilon=model_params['epsilon'], final_epsilon=model_params['final_epsilon'],
            test_epsilon=model_params['test_epsilon'], annealing_steps=model_params['annealing_steps'],
            minibatch_size=model_params['minibatch_size'],
            replay_max_size=model_params['replay_max_size'], update_freq=model_params['update_freq'],
            learning_frequency=model_params['learning_frequency'], ddqn=model_params['ddqn'],
            network_size=model_params['network_size'], normalize=model_params['normalize'], rng=random_state, device=device,
            bcq_threshold=model_params['bcq_threshold'])

    network_weights_file = f"./models/{params['model_name']}/ai/q_network_weights.pt"
    ai.load_weights(weights_file_path=network_weights_file)

    np.random.seed(seed=params['random_seed'])
    random.seed(params['random_seed'])

    params['gamma'] = model_params['gamma']

    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = TabularWrapper(env)
    env = BinaryRewardWrapper(env)
    env = FlatObsWrapper(env)

    env = WithOptions(env)
    env.nb_actions = len(env.options)

    _ = env.reset()

    replay_buffer = ExperienceReplay(capacity=params['replay_max_size'])

    interact_with_environment(env, replay_buffer, params, ai)


def run():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg_file = os.path.join(dir_path, 'buffer_config.yaml')
    params = yaml.safe_load(open(cfg_file, 'r'))

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--replay_max_size", default=None, type=int)
    parser.add_argument("--rand_action_p", default=None, type=float)
    parser.add_argument("--2nd_action_p", default=None, type=float)
    parser.add_argument("--random_seed", default=0, type=int)

    args = parser.parse_args()

    # add list attributes from args to the corresponding ydict values
    for k, v in params.items():
        av = getattr(args, k, None)
        if av:
            params[k] = av

    print('\n')
    print('Parameters ')
    for key in params:
        print(key, params[key])
    print('\n')
    worker(params)


if __name__ == '__main__':
    run()
