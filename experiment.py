import os
import numpy as np
import pickle
from utils.utils import write_to_csv, plot
from copy import deepcopy
import wandb


class DQNExperiment(object):
    def __init__(self, env, ai, episode_max_len, annealing=False, test_epsilon=0.0,
                 replay_min_size=0, score_window_size=100, folder_location='/experiments/', folder_name='expt',
                 saving_period=10, rng=None, sq_learning=True, online_ql=True, test_data_folder=None,
                 episodes_per_test=20, use_wandb=False):
        self.rng = rng
        self.fps = 0
        self.episode_num = 0
        self.last_episode_steps = 0
        self.total_training_steps = 0
        self.score_computer = 0
        self.score_agent = 0
        self.eval_scores = []
        self.eval_scores_dirs = []
        self.eval_scores_locs = []
        self.eval_steps = []
        self.epsilon = []
        self.env = env
        self.ai = ai
        self.annealing = annealing
        self.anneal_count = 0
        self.test_epsilon = test_epsilon
        self.saving_period = saving_period
        self.episode_max_len = episode_max_len
        self.score_agent_window = np.zeros(score_window_size)
        self.steps_agent_window = np.zeros(score_window_size)
        self.replay_min_size = max(self.ai.minibatch_size, replay_min_size)
        self.last_state = np.empty(self.env.state_shape, dtype=np.uint8)
        self.folder_name = self._create_folder(folder_location, folder_name)
        self.curr_epoch = 0
        self.sq_learning = sq_learning
        self.unique_states = dict()
        self.online_ql = online_ql
        self.min_val_loss = float('inf')
        self.same_min_val_loss = float('inf')
        self.train_loss = []
        self.use_wandb = use_wandb

        try:
            with open(test_data_folder, 'rb') as handle:
                self.test_states = pickle.load(handle)
            print(f"Successfully loaded {len(self.test_states)} test cases")
        except Exception:
            self.test_states = []
            for i in range(episodes_per_test):
                self.test_states.append(self.randomize_init_state(return_actions=True))
            with open(test_data_folder, 'wb') as handle:
                pickle.dump(self.test_states, handle)
            print(f"Generated and saved {len(self.test_states)} test cases at: {test_data_folder}")

    def randomize_init_state(self, return_actions=False):
        # randomly move around to start on a new place on the grid
        # these transitions are not recorded
        actions = []
        for _ in range(10):
            rand_prob = np.random.uniform(0, 1)
            if rand_prob < 0.33:
                action = self.env.actions.left
                state, _, _, _ = self.env.step(action, k_fwd=0)
            elif rand_prob < 0.67:
                action = self.env.actions.right
                state, _, _, _ = self.env.step(action, k_fwd=0)
            else:
                action = self.env.actions.forward
                state, _, _, _ = self.env.step(action, k_fwd=1)
            actions.append(action)
        if return_actions:
            return actions
        return state

    def do_epochs(self, number=1, episodes_per_epoch=10000, is_learning=True, is_testing=True, episodes_per_test=10000):
        for epoch in range(self.curr_epoch, number):
            episodes = 0
            loss_window = []
            while episodes < episodes_per_epoch:
                losses = self.do_episodes(number=1, is_learning=is_learning)
                episodes += 1
                loss_window.append(sum(losses) / len(losses))
            self.train_loss.append((sum(loss_window) / len(loss_window)))

            if is_testing:

                if self.online_ql:

                    eval_steps = 0
                    eval_episodes = 0
                    eval_scores = 0
                    print(f'EVALUATING: {episodes_per_test} episodes in environment using current network')

                    # Metric 1: final score
                    while eval_episodes < 1:
                        eval_scores += self.evaluate(number=1)
                        eval_steps += self.last_episode_steps
                        eval_episodes += 1
                    self.eval_scores.append(eval_scores / eval_episodes)
                    self.eval_steps.append(eval_steps / eval_episodes)
                    self.epsilon.append(self.ai.epsilon)
                    self._plot_and_write(plot_dict={'scores': self.eval_scores}, loc=self.folder_name + "/scores",
                                         x_label="Epochs", y_label="Mean Score", title="", kind='line', legend=True,
                                         moving_average=True)
                    self._plot_and_write(plot_dict={'steps': self.eval_steps}, loc=self.folder_name + "/steps",
                                         x_label="Epochs", y_label="Mean Steps", title="", kind='line', legend=True)
                    self._plot_and_write(plot_dict={'epsilon': self.epsilon}, loc=self.folder_name + "/epsilon",
                                         x_label="Epochs", y_label="Mean Steps", title="", kind='line', legend=True)
                    self._plot_and_write(plot_dict={'train_loss': self.train_loss},
                                         loc=self.folder_name + "/train_loss",
                                         x_label="Epochs", y_label="Mean Score", title="", kind='line', legend=True,
                                         moving_average=True)

                    # Metric 2 (for online data): evaluate on test set the score across diff init states
                    eval_episodes = 0
                    eval_scores = 0
                    while eval_episodes < episodes_per_test:
                        eval_scores += self.evaluate(number=1, test_case=eval_episodes)
                        eval_episodes += 1
                    self.eval_scores_locs.append(eval_scores / eval_episodes)
                    self._plot_and_write(plot_dict={'scores': self.eval_scores_locs},
                                         loc=self.folder_name + "/scores_diff_locations",
                                         x_label="Epochs", y_label="Mean Score", title="", kind='line', legend=True,
                                         moving_average=True)

                else:
                    # Offline learning: record training and validation loss
                    eval_scores = 0
                    eval_scores += self.evaluate(number=1)
                    self.eval_scores_locs.append(eval_scores)
                    self._plot_and_write(plot_dict={'scores': self.eval_scores_locs},
                                         loc=self.folder_name + "/val_loss",
                                         x_label="Epochs", y_label="Mean Score", title="", kind='line', legend=True,
                                         moving_average=True)
                    self._plot_and_write(plot_dict={'scores': self.train_loss},
                                         loc=self.folder_name + "/train_loss",
                                         x_label="Epochs", y_label="Mean Score", title="", kind='line', legend=True,
                                         moving_average=True)

                    if self.use_wandb:
                        wandb.log({
                            "train_loss": self.train_loss[-1],
                            "val_loss": self.eval_scores_locs[-1]
                        })

                if epoch % self.saving_period == 0:
                    print(f"Saving to {self.folder_name + '/ai/q_network_weights_' + str(epoch + 1) + '.pt'}")
                    self.ai.dump_network(
                        weights_file_path=self.folder_name + '/ai/q_network_weights_' + str(epoch + 1) + '.pt')
                    try:
                        print(f"Saving best network to {self.folder_name + '/ai/q_network_weights_min_val_loss' + '.pt'}")
                        self.best_ai.dump_network(
                            weights_file_path=self.folder_name + '/ai/q_network_weights_min_val_loss' + '.pt')
                    except AttributeError:
                        pass

        with open(self.folder_name + "/state_visitation", 'wb') as handle:
            pickle.dump(self.unique_states, handle)

        print(f"DONE experiment. Visited {len(self.unique_states)} unique states in total.")

    def do_episodes(self, number=1, is_learning=True):
        all_rewards = []
        for _ in range(number):
            reward = self._do_episode(is_learning=is_learning)
            all_rewards.append(reward)
            self.score_agent_window = self._update_window(self.score_agent_window, self.score_agent)
            self.steps_agent_window = self._update_window(self.steps_agent_window, self.last_episode_steps)
            self.episode_num += 1
        return all_rewards

    def evaluate(self, number=10, test_case=None):
        evaluation_score = 0
        for num in range(number):
            ep_reward = self._do_episode(is_learning=False, evaluate=True, test_case=test_case)
            evaluation_score += ep_reward
        self.score_agent = evaluation_score
        return evaluation_score

    def _do_episode(self, is_learning=True, evaluate=False, test_case=None):
        """
        For online training, returns the score from interacting with environment
        For offline training, returns the loss
        """
        rewards = []
        term = False
        self.fps = 0

        if not self.online_ql:
            if evaluate:
                if test_case is not None:
                    self._episode_reset()
                    actions = self.test_states[test_case]
                    for action in actions:
                        if action == self.env.actions.forward:
                            last_state, _, _, _ = self.env.step(action, k_fwd=1)
                        else:
                            last_state, _, _, _ = self.env.step(action, k_fwd=0)
                    while not term:
                        reward, term = self._step(evaluate=evaluate)
                        rewards.extend(reward)
                        if not term and self.last_episode_steps >= self.episode_max_len:
                            print('Reaching maximum number of steps in the current episode.')
                            term = True
                else:
                    val_loss = self.ai.eval_network(static_val=True)
                    if val_loss < self.same_min_val_loss:
                        self.same_min_val_loss = val_loss
                        self.best_ai_same_val = deepcopy(self.ai)

                    val_loss = self.ai.eval_network(static_val=False)
                    if val_loss < self.min_val_loss:
                        self.min_val_loss = val_loss
                        self.best_ai = deepcopy(self.ai)

                    return val_loss
            else:
                train_loss = self.ai.learn()
                return train_loss

        else:
            # If we have test case specified, we pull this trajectory from test_states for evaluation
            if test_case is not None:
                self._episode_reset()
                actions = self.test_states[test_case]
                for action in actions:
                    if action == self.env.actions.forward:
                        last_state, _, _, _ = self.env.step(action, k_fwd=1)
                    else:
                        last_state, _, _, _ = self.env.step(action, k_fwd=0)
            else:
                self._episode_reset(rand_start=True)

            while not term:
                reward, term = self._step(evaluate=evaluate)
                rewards.extend(reward)
                if (self.ai.transitions.size >= self.replay_min_size and is_learning and \
                    self.last_episode_steps % self.ai.learning_frequency == 0):
                    self.ai.learn()
                # self.score_agent += reward
                if not term and self.last_episode_steps >= self.episode_max_len:
                    print('Reaching maximum number of steps in the current episode.')
                    term = True

        disc_factors = np.power(self.ai.gamma, range(len(rewards)))
        ep_reward = np.dot(disc_factors, rewards)
        return ep_reward

    def _step(self, evaluate=False):
        self.last_episode_steps += 1

        action = self.ai.get_action(self.last_state, evaluate)
        new_obs, reward, game_over, _ = self.env.step(action)
        env_reward = reward
        new_obs = new_obs['image']

        if self.sq_learning:
            k = len(reward)
            disc_factors = np.power(self.ai.gamma, range(len(reward)))
            reward = np.dot(disc_factors, reward)
        else:
            k = 1
            disc_factors = np.power(self.ai.gamma, range(len(reward)))
            reward = np.dot(disc_factors, reward)
            # reward = sum(reward)

        if not evaluate:
            self.ai.transitions.add(s=self.last_state.astype('float32'), a=action, r=reward,
                                    s2=new_obs.astype('float32'), term=game_over, k=k)
            if self.annealing:
                if self.total_training_steps >= self.replay_min_size:
                    self.anneal_count += 1
                    self.ai.anneal_eps(self.total_training_steps - self.replay_min_size)
            self.total_training_steps += 1

        self.last_state = new_obs
        self.unique_states[tuple(self.last_state)] = self.unique_states.get(tuple(self.last_state), 0) + 1
        return env_reward, game_over

    def _episode_reset(self, rand_start=False):
        self.last_episode_steps = 0
        self.score_agent = 0
        self.score_computer = 0
        s = self.env.reset()
        if rand_start:
            print("Randomizing initial state...")
            s = self.randomize_init_state()
        self.last_state = s['image']

    @staticmethod
    def _plot_and_write(plot_dict, loc, x_label="", y_label="", title="", kind='line', legend=True,
                        moving_average=False):
        for key in plot_dict:
            plot(data={key: plot_dict[key]}, loc=loc + ".pdf", x_label=x_label, y_label=y_label, title=title,
                 kind=kind, legend=legend, index_col=None, moving_average=moving_average)
            write_to_csv(data={key: plot_dict[key]}, loc=loc + ".csv")

    @staticmethod
    def _create_folder(folder_location, folder_name):
        i = 0
        while os.path.exists(folder_location + folder_name + str(i)):
            i += 1
        folder_name = folder_location + folder_name + str(i)
        os.makedirs(folder_name)
        os.mkdir(folder_name + '/ai')
        os.mkdir(folder_name + '/checkpoints')
        return folder_name

    @staticmethod
    def _update_window(window, new_value):
        window[:-1] = window[1:]
        window[-1] = new_value
        return window
