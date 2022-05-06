import os
import pickle
import yaml
from ai import AI
from experiment import DQNExperiment
from environment import *
import torch
import wandb
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = ROOT_DIR

np.set_printoptions(suppress=True, linewidth=200, precision=2)


def run():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg_file = os.path.join(dir_path, 'config_minigrid.yaml')
    params = yaml.safe_load(open(cfg_file, 'r'))

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--ddqn', default=None, type=int) # Using integer as boolean here
    parser.add_argument('--sq_learning', default=None, type=int)
    parser.add_argument('--network_size', default=None, type=str)
    parser.add_argument('--replay_max_size', default=None, type=int)
    parser.add_argument('--rand_action_p', default=None, type=float)
    parser.add_argument('--2nd_action_p', default=None, type=float)
    parser.add_argument('--random_seed', default=0, type=int)

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

    for param in ['test_data_folder', 'bcq_threshold']:
        if param not in params:
            params[param] = None

    np.random.seed(params['random_seed'])
    random.seed(params['random_seed'])
    torch.manual_seed(params['random_seed'])
    random_state = np.random.RandomState(params['random_seed'])
    device = torch.device(params['device'])

    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = TabularWrapper(env)
    env = BinaryRewardWrapper(env)
    env = FlatObsWrapper(env)
    env.state_shape = [108]  # Used to initialize empty last state

    if params['option_env']:
        env = WithOptions(env)
        env.nb_actions = len(env.options)
    else:
        env.nb_actions = len(env.actions)
    _ = env.reset()

    params['ddqn'] = True if (params['ddqn'] or (params['ddqn'] == 1)) else False
    params['sq_learning'] = True if (params['sq_learning'] or (params['sq_learning'] == 1)) else False

    if not params['online_ql']:
        params['folder_name'] = f"buf{params['replay_max_size']}_rand{params['rand_action_p']}_second{params['2nd_action_p']}_seed{params['random_seed']}" #+ "1\\ai"
    else:
        params['folder_name'] = f'online'

    params['hyperparams'] = f"buf{params['replay_max_size']}_rand{params['rand_action_p']}_second{params['2nd_action_p']}"
    params['model_type'] = ('BCQ' if params['network_size'] == 'bcq_fc' else ('DDQN' if params['ddqn'] else 'DQN')) # + f'_seed{params['random_seed']}'
    params['model_type'] = params['hyperparams'] + '_' + params['model_type']
    params['wandb_key'] = None if params['wandb_key'] is None or params['wandb_key'] == 'None' else params['wandb_key']

    if params['wandb_key'] is not None:
        _ = os.system('wandb login {}'.format(params['wandb_key']))
        os.environ['WANDB_API_KEY'] = params['wandb_key']
        params['unique_run_id'] = wandb.util.generate_id()
        wandb.init(id=params['unique_run_id'], resume='allow', project='smdp_toy_example', group=params['model_type'],
                   name=f"seed{params['random_seed']}")
        wandb.config.update(params)

    for ex in range(params['num_experiments']):
        print('\n')
        print('>>>>> Experiment ', ex, ' >>>>>')
        print('\n')
        network_size = params['network_size']
        ai = AI(nb_actions=env.nb_actions, gamma=params['gamma'],
                learning_rate=params['learning_rate'], epsilon=params['epsilon'], final_epsilon=params['final_epsilon'],
                test_epsilon=params['test_epsilon'], annealing_steps=params['annealing_steps'],
                minibatch_size=params['minibatch_size'],
                replay_max_size=params['replay_max_size'], update_freq=params['update_freq'],
                learning_frequency=params['learning_frequency'], ddqn=params['ddqn'],
                network_size=network_size, normalize=params['normalize'], rng=random_state, device=device,
                bcq_threshold=params['bcq_threshold'])

        if params['test']:  # note to pass correct folder name
            network_weights_file = os.path.join(ROOT_DIR, 'models', params['folder_name'], 'q_network_weights_501.pt')
            ai.load_weights(weights_file_path=network_weights_file)

        # Load replay buffer for offline learning
        if not params['online_ql']:
            print(f'Loading replay buffer for offline learning')
            buffer_folder = f"./buffers/{params['buffer_name']}_{params['rand_action_p']}_{params['2nd_action_p']}_train_{params['replay_max_size']}_{params['random_seed']}"
            ai.transitions.load(buffer_folder, size=params['replay_max_size'])
            val_buffer_rand, val_buffer_second = params['rand_action_p'], params['2nd_action_p']
            # val_buffer_rand, val_buffer_second = 0.25, 0.25

            val_buffer_folder = f"./buffers/{params['buffer_name']}_{val_buffer_rand}_{val_buffer_second}_val"
            ai.val_transitions.load(val_buffer_folder, size=params['val_buffer_size'])
            val_buffer_rand, val_buffer_second = 0.25, 0.25
            val_buffer_folder = f"./buffers/{params['buffer_name']}_{val_buffer_rand}_{val_buffer_second}_val"
            ai.same_val_transitions.load(val_buffer_folder, size=params['val_buffer_size'])

        expt = DQNExperiment(env=env, ai=ai, episode_max_len=params['episode_max_len'], annealing=params['annealing'],
                             replay_min_size=params['replay_min_size'], test_epsilon=params['test_epsilon'],
                             folder_location=os.path.join(OUTPUT_DIR, params['folder_location']),
                             folder_name=params['folder_name'], score_window_size=100, rng=random_state,
                             sq_learning=params['sq_learning'], online_ql=params['online_ql'],
                             saving_period=params['saving_period'],
                             test_data_folder=os.path.join(OUTPUT_DIR, params['test_data_folder']),
                             episodes_per_test=params['episodes_per_test'],
                             use_wandb=False if params['wandb_key'] is None else True)
        env.reset()
        if not params['test']:
            with open(expt.folder_name + '/config.yaml', 'w') as y:
                yaml.safe_dump(params, y)  # saving params for reference
            expt.do_epochs(number=params['num_epochs'], is_learning=params['is_learning'],
                           episodes_per_epoch=params['episodes_per_epoch'], is_testing=params['is_testing'],
                           episodes_per_test=params['episodes_per_test'])

            # if offline rl, evaluate best ai on the test set
            if not params['online_ql']:
                test_scores = []
                expt.ai = expt.best_ai
                expt.ai.device = params['device']
                for i in range(params['episodes_per_test']):
                    test_scores.append(expt.evaluate(number=1, test_case=i))
                print(f'Test score using best model is: {sum(test_scores) / len(test_scores)}')
                with open(expt.folder_name + '/test_scores_min_val_loss', 'wb') as handle:
                    pickle.dump(test_scores, handle)

                test_scores = []
                expt.ai = expt.best_ai_same_val
                expt.ai.device = params['device']
                for i in range(params['episodes_per_test']):
                    test_scores.append(expt.evaluate(number=1, test_case=i))
                print(f'Test score using best model (evaluated on same val test) is: {sum(test_scores) / len(test_scores)}')
                with open(expt.folder_name + '/test_scores_static_val', 'wb') as handle:
                    pickle.dump(test_scores, handle)

    else:
        test_scores = []
        for i in range(params['episodes_per_test']):
            test_scores.append(expt.evaluate(number=1, test_case=i))
        avg_test_score = sum(test_scores) / len(test_scores)
        print(f'Average test score across test cases: {avg_test_score}')
        return avg_test_score


if __name__ == '__main__':
    run()
