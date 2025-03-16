import os
import time
import sys
print(sys.path)
import hydra, json

from hw8.roble.infrastructure.rl_trainer import RL_Trainer
from hw8.roble.agents.explore_or_exploit_agent import ExplorationOrExploitationAgent
from hw8.roble.agents.dqn_agent import DQNAgent
from hw8.roble.infrastructure.dqn_utils import get_env_kwargs, PiecewiseSchedule, ConstantSchedule
from hw8.roble.infrastructure import pytorch_util as ptu

from omegaconf import DictConfig, OmegaConf

class Q_Trainer(object):

    def __init__(self, params):
        self.params = params

        env_args = get_env_kwargs(params['env']['env_name'])

        # get agent class
        if self.params['alg']['rl_alg'] == 'explore_or_exploit':
            agent_class = ExplorationOrExploitationAgent
        elif self.params['alg']['rl_alg'] == 'dqn':
            agent_class = DQNAgent
        else:
            print("Pick a rl_alg first")
            sys.exit()

        self.params['alg']['train_batch_size'] = params['alg']['batch_size']

        # append env args to params
        self.params = {**self.params, **env_args}

        print(self.params)

        self.rl_trainer = RL_Trainer(self.params, agent_class=agent_class)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.params['alg']['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )

def my_app(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))
    import os
    print("Command Dir:", os.getcwd())
    params = vars(cfg)
    # params.extend(env_args)
    for key, value in cfg.items():
        params[key] = value

    params['eps'] = 0.2
    params['exploit_weight_schedule'] = ConstantSchedule(1.0)

    if params['env']['env_name']=='PointmassEasy-v0':
        params['env']['max_episode_length']=50
    if params['env']['env_name']=='PointmassMedium-v0':
        params['env']['max_episode_length']=150
    if params['env']['env_name']=='PointmassHard-v0':
        params['env']['max_episode_length']=100
    if params['env']['env_name']=='PointmassVeryHard-v0':
        params['env']['max_episode_length']=200
    
    if params['alg']['use_rnd']:
        params['explore_weight_schedule'] = PiecewiseSchedule([(0,1), (params['alg']['num_exploration_steps'], 0)], outside_value=0.0)
    else:
        params['explore_weight_schedule'] = ConstantSchedule(0.0)

    if params['alg']['unsupervised_exploration']:
        params['explore_weight_schedule'] = ConstantSchedule(1.0)
        params['exploit_weight_schedule'] = ConstantSchedule(0.0)
        
        if not params['alg']['use_rnd']:
            params['alg']['learning_starts'] = params['alg']['num_exploration_steps']
    
    print ("params: ", params)
    
    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################


    logdir_prefix = 'hw8_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    exp_name = logdir_prefix + cfg.env.exp_name + '_' + cfg.env.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, exp_name)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.logging.logdir = logdir
        cfg.logging.exp_name = exp_name

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = Q_Trainer(params)
    
    # Print GPU information after initialization
    print("\n\n\nGPU INFORMATION:")
    ptu.print_gpu_info()
    print("\n\n\n")
    
    trainer.run_training_loop()

if __name__ == "__main__":
    import os
    print("Command Dir:", os.getcwd())
    my_main()