import numpy as np
import torch
from hw8.roble.infrastructure import pytorch_util as ptu

from hw8.roble.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from hw8.roble.policies.argmax_policy import ArgMaxPolicy
from hw8.roble.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['alg']['batch_size']
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['alg']['ac_dim']
        self.learning_starts = agent_params['alg']['learning_starts']
        self.learning_freq = agent_params['alg']['learning_freq']
        self.target_update_freq = agent_params['alg']['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        # self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.critic = DQNCritic(**agent_params)
        self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env']['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0
        
        # For prediction model (RND)
        self.predictor_network = None
        self.target_network = None
        if 'use_rnd' in agent_params.get('alg', {}) and agent_params['alg']['use_rnd']:
            # Initialize predictor and target networks for RND
            self._init_prediction_networks(agent_params)
  
    def _init_prediction_networks(self, agent_params):
        """Initialize predictor and target networks for Random Network Distillation (RND)"""
        from torch import nn
        
        # Get observation dimension
        ob_dim = agent_params['alg']['ob_dim']
        
        # Get network parameters from agent_params
        rnd_output_size = agent_params['alg'].get('rnd_output_size', 5)
        rnd_n_layers = agent_params['alg'].get('rnd_n_layers', 2)
        rnd_size = agent_params['alg'].get('rnd_size', 400)
        
        # Create predictor network (trained to predict target network outputs)
        predictor_layers = []
        predictor_layers.append(nn.Linear(ob_dim, rnd_size))
        predictor_layers.append(nn.ReLU())
        
        for _ in range(rnd_n_layers - 1):
            predictor_layers.append(nn.Linear(rnd_size, rnd_size))
            predictor_layers.append(nn.ReLU())
            
        predictor_layers.append(nn.Linear(rnd_size, rnd_output_size))
        self.predictor_network = nn.Sequential(*predictor_layers).to(ptu.device)
        
        # Create target network (fixed, random weights)
        target_layers = []
        target_layers.append(nn.Linear(ob_dim, rnd_size))
        target_layers.append(nn.ReLU())
        
        for _ in range(rnd_n_layers - 1):
            target_layers.append(nn.Linear(rnd_size, rnd_size))
            target_layers.append(nn.ReLU())
            
        target_layers.append(nn.Linear(rnd_size, rnd_output_size))
        self.target_network = nn.Sequential(*target_layers).to(ptu.device)
        
        # Initialize target network with random weights and freeze it
        for param in self.target_network.parameters():
            param.requires_grad = False
            
        # Initialize optimizer for predictor network
        self.predictor_optimizer = self.optimizer_spec.constructor(
            self.predictor_network.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
    
    def get_prediction_error(self, obs):
        """
        Calculate prediction error for Random Network Distillation (RND)
        
        Args:
            obs: Observations to calculate prediction error for
            
        Returns:
            Prediction error (L2 distance between predictor and target outputs)
        """
        if self.predictor_network is None or self.target_network is None:
            return np.zeros(len(obs))
            
        # Convert observations to tensor
        obs_tensor = ptu.from_numpy(obs)
        
        # Get predictions from both networks
        with torch.no_grad():
            target_output = self.target_network(obs_tensor)
            predictor_output = self.predictor_network(obs_tensor)
            
            # Calculate MSE between outputs
            prediction_error = torch.sum((target_output - predictor_output) ** 2, dim=1)
            
            # Return as numpy array
            return ptu.to_numpy(prediction_error)
    
    def train_prediction_model(self, next_obs):
        """
        Train the predictor network to predict the output of the target network
        
        Args:
            next_obs: Next observations to train on
            
        Returns:
            Loss value
        """
        if self.predictor_network is None or self.target_network is None:
            return 0.0
            
        # Convert observations to tensor
        next_obs_tensor = ptu.from_numpy(next_obs)
        
        # Get target network output (no gradient needed)
        with torch.no_grad():
            target_output = self.target_network(next_obs_tensor)
            
        # Get predictor network output
        predictor_output = self.predictor_network(next_obs_tensor)
        
        # Calculate MSE loss
        loss = torch.mean(torch.sum((target_output - predictor_output) ** 2, dim=1))
        
        # Optimize predictor network
        self.predictor_optimizer.zero_grad()
        loss.backward()
        self.predictor_optimizer.step()
        
        return ptu.to_numpy(loss)
        
    def add_to_replay_buffer(self, paths):
        if paths is None:
            return
            
        # If paths is provided, add them to the replay buffer
        # (This case is unlikely to occur for this agent, but included for completeness)
        for path in paths:
            for i in range(len(path["observation"])):
                idx = self.replay_buffer.store_frame(path["observation"][i])
                self.replay_buffer.store_effect(
                    idx,
                    path["action"][i],
                    path["reward"][i],
                    path["terminal"][i]
                )
 
    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        eps = self.exploration.value(self.t)

        # Use epsilon-greedy exploration
        if np.random.random() < eps:
            # Take a random action
            action = self.env.action_space.sample()
        else:
            # Use the policy to select an action
            obs = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(obs)

        # Take a step in the environment
        next_obs, reward, done, info = self.env.step(action)
        
        # Store the effect of the action
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)
        
        # Update last_obs
        self.last_obs = next_obs.copy()
        
        # Reset the environment if the episode is done
        if done:
            self.last_obs = self.env.reset()

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
            # Fill in the call to the update function using the appropriate tensors
            log = self.critic.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n
            )

            # Update the target network periodically 
            # HINT: your critic already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1
        self.t += 1
        return log