import numpy as np



class ArgMaxPolicy(object):

    def __init__(self, critic, use_boltzmann=False):
        self.critic = critic
        self.use_boltzmann = use_boltzmann

    def set_critic(self, critic):
        self.critic = critic

    def get_action(self, obs):
        # MJ: changed the dimension check to a 3
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output

        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output (get it from hw3)
        # NOTE: you should adapt your code so that it considers the boltzmann distribution case

        # Get Q-values from the critic network
        q_values = self.critic.qa_values(observation)

        if self.use_boltzmann:
            # Use Boltzmann distribution for exploration
            # Convert Q-values to probabilities using softmax
            # Add small epsilon for numerical stability
            q_values = q_values - np.max(q_values, axis=1, keepdims=True)  # For numerical stability
            exp_q = np.exp(q_values)
            probabilities = exp_q / np.sum(exp_q, axis=1, keepdims=True)
            
            # Sample actions according to these probabilities
            action = self.sample_discrete(probabilities)
        else:
            # Greedy action selection - take action with highest Q-value
            action = np.argmax(q_values, axis=1)

        # If it's a single observation, return a scalar action
        if len(obs.shape) <= 3:
            return int(action[0])  # Convert to int
        
        # For batch observations, convert array elements to integers
        return np.array([int(a) for a in action])

    def sample_discrete(self, p):
        # https://stackoverflow.com/questions/40474436/how-to-apply-numpy-random-choice-to-a-matrix-of-probability-values-vectorized-s
        c = p.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        choices = (u < c).argmax(axis=1)
        return choices

    ####################################
    ####################################