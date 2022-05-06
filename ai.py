import numpy as np
from utils.utils import ExperienceReplay
from utils.model import BCQ_FC, DQN_FC, BCQ_Conv, DQN_Conv
import torch
import torch.optim as optim
import torch.nn.functional as F


class AI(object):
    def __init__(self, nb_actions, gamma=.99,
                 learning_rate=0.00025, epsilon=0.05, final_epsilon=0.05, test_epsilon=0.0, annealing_steps=1000,
                 minibatch_size=32, replay_max_size=100, update_freq=50, learning_frequency=1, ddqn=False,
                 network_size='dqn_fc', normalize=1., rng=None, device=None, sq_learning=True, bcq_threshold=0):
        self.rng = rng
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.start_epsilon = epsilon
        self.test_epsilon = test_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = annealing_steps
        self.minibatch_size = minibatch_size
        self.network_size = network_size
        self.update_freq = update_freq
        self.update_counter = 0
        self.normalize = normalize
        self.learning_frequency = learning_frequency
        self.replay_max_size = replay_max_size
        self.transitions = ExperienceReplay(capacity=self.replay_max_size)
        self.ddqn = ddqn
        self.device = device
        self.network = self._build_network()
        self.target_network = self._build_network()
        self.weight_transfer(from_model=self.network, to_model=self.target_network)
        self.network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, amsgrad=True)
        self.sq_learning = sq_learning
        self.total_qupdates = 0
        self.bcq_threshold = bcq_threshold
        self.val_transitions = ExperienceReplay(capacity=self.replay_max_size)
        self.same_val_transitions = ExperienceReplay(capacity=self.replay_max_size)

    def _build_network(self):
        if self.network_size == 'bcq_conv':
            self.model_type = "bcq"
            return BCQ_Conv()
        elif self.network_size == "dqn_conv":
            self.model_type = "dqn"
            return DQN_Conv()
        elif self.network_size == "bcq_fc":
            self.model_type = "bcq"
            return BCQ_FC()
        elif self.network_size == "dqn_fc":
            self.model_type = "dqn"
            return DQN_FC()
        else:
            raise ValueError('Invalid network_size.')

    def eval_network(self, static_val=False):
        if static_val:
            s, a, r, s2, t, k = self.same_val_transitions.sample(self.val_transitions.size)
        else:
            s, a, r, s2, t, k = self.val_transitions.sample(self.val_transitions.size)
        s = torch.FloatTensor(s).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        t = torch.FloatTensor(np.float32(t)).to(self.device)
        k = torch.FloatTensor(k).to(self.device)

        with torch.no_grad():
            if self.model_type == "dqn":
                q = self.network(s / self.normalize)
                q2 = self.target_network(s2 / self.normalize).detach()
                q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1)
                if self.ddqn:
                    q2_net = self.network(s2 / self.normalize).detach()
                    q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
                else:
                    q2_max = torch.max(q2, 1)[0]

                bellman_target = r + (self.gamma ** k) * q2_max.detach() * (1 - t)
                errs = (bellman_target - q_pred).unsqueeze(1)
                quad = torch.min(torch.abs(errs), 1)[0]
                lin = torch.abs(errs) - quad
                loss = torch.sum(0.5 * quad.pow(2) + lin)

            else:
                q, imt, i = self.network(s2 / self.normalize)
                imt = imt.exp()
                imt = (imt / imt.max(1, keepdim=True)[0] > self.bcq_threshold).float()
                next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)
                q, imt, i = self.target_network(s2 / self.normalize)
                target_Q = (r + (1 - t) * (self.gamma ** k) * q.gather(1, next_action).squeeze()).unsqueeze(1)

                curr_Q, imt, i = self.network(s / self.normalize)
                curr_Q = curr_Q.gather(1, a.unsqueeze(1))
                q_loss = F.smooth_l1_loss(curr_Q, target_Q)
                i_loss = F.nll_loss(imt, a.reshape(-1))
                loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        return loss.item()

    def train_on_batch(self, s, a, r, s2, t, k):
        s = torch.FloatTensor(s).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        t = torch.FloatTensor(np.float32(t)).to(self.device)
        k = torch.FloatTensor(k).to(self.device)

        if self.model_type == "dqn":
            q = self.network(s / self.normalize)
            q2 = self.target_network(s2 / self.normalize).detach()
            q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1)
            if self.ddqn:
                q2_net = self.network(s2 / self.normalize).detach()
                q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
            else:
                q2_max = torch.max(q2, 1)[0]

            bellman_target = r + (self.gamma ** k) * q2_max.detach() * (1 - t)

            errs = (bellman_target - q_pred).unsqueeze(1)
            quad = torch.min(torch.abs(errs), 1)[0]
            lin = torch.abs(errs) - quad
            loss = torch.sum(0.5 * quad.pow(2) + lin)

        else:
            with torch.no_grad():
                q, imt, i = self.network(s2 / self.normalize)
                imt = imt.exp()
                imt = (imt / imt.max(1, keepdim=True)[0] > self.bcq_threshold).float()
                next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)
                q, imt, i = self.target_network(s2 / self.normalize)
                target_Q = (r + (1 - t) * (self.gamma ** k) * q.gather(1, next_action).squeeze()).unsqueeze(1)

            curr_Q, imt, i = self.network(s / self.normalize)
            curr_Q = curr_Q.gather(1, a.unsqueeze(1))
            q_loss = F.smooth_l1_loss(curr_Q, target_Q)
            i_loss = F.nll_loss(imt, a.reshape(-1))
            loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_q(self, s):
        s = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        if self.model_type == "dqn":
            return self.network(s / self.normalize).detach().cpu().numpy()
        else:
            q, imt, i = self.network(s / self.normalize)
            return q.detach().cpu().numpy()

    def get_max_action(self, s):
        s = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        if self.network_size not in ["bcq", "bcq_fc"]:
            q = self.network(s / self.normalize).detach()
            action = [q.argmax(1).item()]
        else:
            with torch.no_grad():
                q, imt, i = self.network(s / self.normalize)
                imt = imt.exp()
                imt = (imt / imt.max(1, keepdim=True)[0] > self.bcq_threshold).float()
                # print(imt)
                action = [(imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True).item()]
        return action

    def get_2ndmax_action(self, s):
        s = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        if self.model_type == "dqn":
            q = self.network(s / self.normalize).detach()
            action = torch.argsort(q).squeeze()[-2].item()
        else:
            with torch.no_grad():
                q, imt, i = self.network(s / self.normalize)
                imt = imt.exp()
                imt = (imt / imt.max(1, keepdim=True)[0] > self.bcq_threshold).float()
                masked_values = (imt * q + (1 - imt) * -1e8)
                action = torch.argsort(masked_values).squeeze()[-2].item()
        return action

    def get_action(self, states, evaluate):
        # get action WITH e-greedy exploration
        eps = self.epsilon if not evaluate else self.test_epsilon
        if self.rng.binomial(1, eps):
            return self.rng.randint(self.nb_actions)
        else:
            return self.get_max_action(states)[0]

    def learn(self):
        """ Learning from one minibatch """
        assert self.minibatch_size <= self.transitions.size, 'not enough data in the pool'
        s, a, r, s2, term, k = self.transitions.sample(self.minibatch_size)
        loss = self.train_on_batch(s, a, r, s2, term, k)
        if self.update_counter == self.update_freq:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
            self.update_counter = 0
        else:
            self.update_counter += 1
        self.total_qupdates += 1
        return loss

    def anneal_eps(self, step):
        if self.epsilon > self.final_epsilon:
            decay = (self.start_epsilon - self.final_epsilon) * step / self.decay_steps
            self.epsilon = self.start_epsilon - decay
        if step >= self.decay_steps:
            self.epsilon = self.final_epsilon

    def dump_network(self, weights_file_path):
        torch.save(self.network.state_dict(), weights_file_path)

    def load_weights(self, weights_file_path, target=False):
        self.network.load_state_dict(torch.load(weights_file_path))
        if target:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)

    @staticmethod
    def weight_transfer(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())

    def __getstate__(self):
        _dict = {k: v for k, v in self.__dict__.items()}
        del _dict['device']  # is not picklable
        del _dict['transitions']  # huge object (if you need the replay buffer, save it with np.save)
        return _dict
