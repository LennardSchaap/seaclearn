import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from gym.spaces.utils import flatdim

from utils import init


class Policy_continuous(nn.Module):
    def __init__(self, obs_space, action_space, base=None, base_kwargs=None):
        super(Policy_continuous, self).__init__()

        obs_shape = obs_space.shape

        if base_kwargs is None:
            base_kwargs = {}

        num_actions = flatdim(action_space)
        self.model = MLPBase(obs_shape[0], num_actions, **base_kwargs)
        
    @property
    def is_recurrent(self):
        return self.model.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.model.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def add_noise(self, action, variance, noise_clip=2.0):

        noise = (variance**0.5)*torch.randn_like(action)
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        action += noise
        action = torch.clamp(action, -1, 1)

        return action

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, (alpha, beta), rnn_hxs = self.model(inputs, rnn_hxs, masks)

        # action = self.add_noise(action, variance)

        # Create alpha beta distribution
        # print(alpha, beta)
        beta_dist = D.Beta(alpha, beta)

        # Use rsample to obtain samples with reparameterization for gradient flow
        action = beta_dist.rsample() # Action is now a value between [0, 1]

        # Calculate log probabilities
        action = torch.clamp(action, 0.001, 0.999)
        action_log_probs = beta_dist.log_prob(action)

        # Map action to a range between [-1, 1]
        action = action * 2 - 1

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.model(inputs, rnn_hxs, masks)
        return value
    
    def evaluate_actions(self, inputs, rnn_hxs, masks, action): 
        value, (alpha, beta), rnn_hxs = self.model(inputs, rnn_hxs, masks)

        # Map action to a range between [0, 1]
        action = (action + 1) / 2

        # Create beta distribution
        beta_dist = D.Beta(alpha, beta)

        # Calculate log probabilities for the original action, but clamp the action to avoid numerical issues
        action = torch.clamp(action, 0.001, 0.999)
        action_log_probs = beta_dist.log_prob(action)

        # Calculate the distribution entropy for the beta distribution
        dist_entropy = beta_dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class SoftplusAddOne(nn.Module):
    def forward(self, x):
        return torch.nn.functional.softplus(x) + 1.0

class MLPBase(NNBase):
    def __init__(self, num_inputs, num_outputs, recurrent=False, hidden_size=256):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            # nn.Sequential(
                init_(nn.Linear(hidden_size, 2 * num_outputs)), # Output alpha and beta values
                SoftplusAddOne() # Add 1 so alpha and beta are > 1
            ### http://proceedings.mlr.press/v70/chou17a/chou17a.pdf ###
            # )
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        alpha, beta = hidden_actor.chunk(2, dim=-1)

        return self.critic_linear(hidden_critic), (alpha, beta), rnn_hxs
