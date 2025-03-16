from collections import OrderedDict
import pickle
import os
import sys
import time
from tqdm import tqdm
import wandb
import gym
from gym import wrappers
import numpy as np
import torch
import matplotlib.pyplot as plt
from hw8.roble.infrastructure import pytorch_util as ptu
from hw8.roble.infrastructure.atari_wrappers import ReturnWrapper

from hw8.roble.infrastructure import utils
from hw8.roble.infrastructure.logger import Logger
from hw8.roble.infrastructure import pytorch_util as ptu


from hw8.roble.agents.explore_or_exploit_agent import ExplorationOrExploitationAgent
from hw8.roble.infrastructure.dqn_utils import (
    get_wrapper_by_name,
    register_custom_envs,
)

# register all of our envs
import hw8.roble.envs

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
os.environ["SDL_VIDEODRIVER"] = "dummy"


class RL_Trainer(object):

    def __init__(self, params, agent_class=None):

        #############
        ## INIT
        #############

        # Get params, create logger
        # Add this near where the logger is initialized
        
        logdir = params["logging"]["logdir"]
        
        # Set up both logging systems
        # Custom logger for metrics
        self.logger = Logger(logdir)
        
        # # Fix: Update log_to_stdout to handle all three arguments
        # def log_to_stdout(value, key, itr):
        #     original_log = self.logger.log_scalar
        #     logging.info(f"[Iteration {itr}] {key}: {value}")
        #     original_log(value, key, itr)
        
        # self.logger.log_scalar = log_to_stdout

        self.params = params

        # Set random seeds
        seed = self.params["logging"]["random_seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params["alg"]["no_gpu"],
            gpu_id=self.params["alg"]["gpu_id"],
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        register_custom_envs()
        self.env = gym.make(self.params["env"]["env_name"])
        self.eval_env = gym.make(self.params["env"]["env_name"])
        print(self.params["env"]["env_name"])
        if not ("pointmass" in self.params["env"]["env_name"]):
            import matplotlib

            matplotlib.use("Agg")
            # self.env.set_logdir(self.params['logging']['logdir'] + '/expl_')
            # self.eval_env.set_logdir(self.params['logging']['logdir'] + '/eval_')

        if self.params["logging"]["video_log_freq"] > 0:
            self.episode_trigger = (
                lambda episode: episode % self.params["logging"]["video_log_freq"] == 0
            )
        else:
            self.episode_trigger = lambda episode: False

        if "env_wrappers" in self.params:
            # These operations are currently only for Atari envs
            self.env = wrappers.RecordEpisodeStatistics(self.env, deque_size=1000)
            self.env = ReturnWrapper(self.env)
            # self.env = wrappers.RecordVideo(self.env, os.path.join(self.params['logging']['logdir'], "gym"), episode_trigger=self.episode_trigger)
            self.env = wrappers.Monitor(
                self.env,
                os.path.join(self.params["logging"]["logdir"], "gym"),
                video_callable=self.episode_trigger,
                force=True,
            )

            self.env = params["env_wrappers"](self.env)

            self.eval_env = wrappers.RecordEpisodeStatistics(
                self.eval_env, deque_size=1000
            )
            self.eval_env = ReturnWrapper(self.eval_env)
            # self.eval_env = wrappers.RecordVideo(self.eval_env, os.path.join(self.params['logging']['logdir'], "gym"), episode_trigger=self.episode_trigger)
            self.eval_env = wrappers.Monitor(
                self.eval_env,
                os.path.join(self.params["logging"]["logdir"], "gym"),
                video_callable=self.episode_trigger,
                force=True,
            )
            self.eval_env = params["env_wrappers"](self.eval_env)

            self.mean_episode_reward = -float("nan")
            self.best_mean_episode_reward = -float("inf")
        if (
            "non_atari_colab_env" in self.params
            and self.params["logging"]["video_log_freq"] > 0
        ):
            self.env = wrappers.RecordVideo(
                self.env,
                os.path.join(self.params["logging"]["logdir"], "gym"),
                episode_trigger=self.episode_trigger,
            )
            self.eval_env = wrappers.RecordVideo(
                self.eval_env,
                os.path.join(self.params["logging"]["logdir"], "gym"),
                episode_trigger=self.episode_trigger,
            )

            self.mean_episode_reward = -float("nan")
            self.best_mean_episode_reward = -float("inf")
        self.env.seed(seed)
        self.eval_env.seed(seed)

        # Maximum length for episodes
        self.params["env"]["max_episode_length"] = (
            self.params["env"]["max_episode_length"] or self.env.spec.max_episode_steps
        )
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params["env"]["max_episode_length"]

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params["alg"]["discrete"] = discrete

        # Observation and action sizes

        ob_dim = (
            self.env.observation_space.shape
            if img
            else self.env.observation_space.shape[0]
        )
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params["alg"]["ac_dim"] = ac_dim
        self.params["alg"]["ob_dim"] = ob_dim

        # simulation timestep, will be used for video saving
        if "model" in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        elif "env_wrappers" in self.params:
            self.fps = 30  # This is not actually used when using the Monitor wrapper
        elif "video.frames_per_second" in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata["video.frames_per_second"]
        else:
            self.fps = 10

        #############
        ## AGENT
        #############

        self.agent = agent_class(self.env, self.params)



    def run_training_loop(
        self,
        n_iter,
        collect_policy,
        eval_policy,
        buffer_name=None,
        initial_expertdata=None,
        relabel_with_expert=False,
        start_relabel_with_expert=1,
        expert_policy=None,
        use_wandb=True,
    ):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        :param use_wandb: whether to use wandb for logging instead of tensorboard
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = (
            1000 if isinstance(self.agent, ExplorationOrExploitationAgent) else 1
        )

        # Initialize wandb if requested
        if use_wandb:
            try:
                import wandb

                if not wandb.run:
                    # Create a simplified config with only essential parameters
                    simple_config = {
                        "env_name": self.params["env"]["env_name"],
                        "max_episode_length": self.params["env"]["max_episode_length"],
                        "batch_size": self.params["alg"]["batch_size"],
                        "batch_size_initial": self.params["alg"]["batch_size_initial"],
                        "n_iter": n_iter,
                        "learning_rate": self.params["alg"].get("learning_rate", "N/A"),
                        "gamma": self.params["alg"].get("gamma", "N/A"),
                        "double_q": self.params["alg"].get("double_q", False),
                    }

                    # Add exploration parameters if available
                    if "agent_params" in self.params and self.params["agent_params"]:
                        agent_params = self.params["agent_params"]
                        if "use_rnd" in agent_params:
                            simple_config["use_rnd"] = agent_params["use_rnd"]
                        if "use_density_model" in agent_params:
                            simple_config["use_density_model"] = agent_params[
                                "use_density_model"
                            ]
                        if "num_exploration_steps" in agent_params:
                            simple_config["num_exploration_steps"] = agent_params[
                                "num_exploration_steps"
                            ]

                    # Extract project and run name
                    project = self.params.get("wandb", {}).get("project", "hw8")
                    run_name = self.params.get("wandb", {}).get(
                        "name",
                        f"{self.params['env']['exp_name']}_{time.strftime('%Y%m%d_%H%M%S')}",
                    )
                    log_dir = self.params["logging"]["logdir"]

                    print(f"Initializing wandb with project={project}, name={run_name}")
                    wandb.init(
                        project=project,
                        name=run_name,
                        config=simple_config,
                        dir=log_dir,
                    )
                    print("Successfully initialized wandb for logging")
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                print("Using tensorboard for logging instead.")
                use_wandb = False

        # Add a variable to control evaluation frequency

        for itr in tqdm(range(n_iter)):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************" % itr)

            # decide if videos should be rendered/logged at this iteration
            if (
                itr % self.params["logging"]["video_log_freq"] == 0
                and self.params["logging"]["video_log_freq"] != -1
            ):
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params["logging"]["scalar_log_freq"] == -1:
                self.logmetrics = False
            elif itr % self.params["logging"]["scalar_log_freq"] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            if isinstance(self.agent, ExplorationOrExploitationAgent):
                self.agent.step_env()
                envsteps_this_batch = 1
                train_video_paths = None
                paths = None
            else:
                use_batchsize = self.params["alg"]["batch_size"]
                if itr == 0:
                    use_batchsize = self.params["alg"]["batch_size_initial"]
                    # Safety check to ensure we don't have a zero batch size
                    if use_batchsize <= 0:
                        print("Warning: batch_size_initial was set to", use_batchsize)
                        print(
                            "Using regular batch_size instead:",
                            self.params["alg"]["batch_size"],
                        )
                        use_batchsize = self.params["alg"]["batch_size"]
                paths, envsteps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr, initial_expertdata, collect_policy, use_batchsize
                    )
                )

            if isinstance(self.agent, ExplorationOrExploitationAgent):
                if (not self.agent.offline_exploitation) or (
                    self.agent.t <= self.agent.num_exploration_steps
                ):
                    self.total_envsteps += envsteps_this_batch
            else:
                # For DQNAgent or other agents that don't have these attributes
                self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr >= start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # # add collected data to replay buffer
            # if isinstance(self.agent, ExplorationOrExploitationAgent):
            #     # For ExplorationOrExploitationAgent, data is already added in step_env()
            #     # The add_to_replay_buffer call is kept for interface consistency
            #     if (not self.agent.offline_exploitation) or (
            #         self.agent.t <= self.agent.num_exploration_steps
            #     ):
            #         self.agent.add_to_replay_buffer(paths)  # paths is None here
            # else:
            #     # For other agents like DQNAgent, add the collected paths to the replay buffer
            #                 self.agent.add_to_replay_buffer(paths)

            if not isinstance(self.agent, ExplorationOrExploitationAgent):
                self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            # if itr % print_period == 0:
            #     print("\nTraining agent...")
            all_logs = self.train_agent()

            # Log densities and output trajectories


            # log/save
            if self.logvideo or self.logmetrics:
                # Perform basic logging every iteration
                # 

                # Only perform full evaluation (including policy evaluation) every eval_frequency iterations
                should_evaluate = (
                    itr % self.params["logging"]["eval_frequency"] == 0
                ) or (itr == self.params["alg"]["n_iter"] - 1)


                if isinstance(self.agent, ExplorationOrExploitationAgent):
                    if use_wandb:
                        if should_evaluate:
                            print("\nBeginning logging procedure...")
                            self.dump_density_graphs(itr)
                            
                            self.perform_logging_wandb(
                                itr, paths, eval_policy, train_video_paths, all_logs
                            )

                    else:
                        self.perform_dqn_logging(all_logs, evaluate=should_evaluate)
                else:
                    if use_wandb:
                        if should_evaluate:
                            self.perform_logging_wandb(
                                itr, paths, eval_policy, train_video_paths, all_logs
                            )

                    else:
                        self.perform_logging(
                            itr,
                            paths,
                            eval_policy,
                            train_video_paths,
                            all_logs,
                            evaluate=should_evaluate,
                        )

                # Save parameters on evaluation iterations
                if should_evaluate and self.params["logging"]["save_params"]:
                    self.agent.save(
                        "{}/agent_itr_{}.pt".format(
                            self.params["logging"]["logdir"], itr
                        )
                    )

        # Finish wandb run if it was used
        if use_wandb:
            try:
                import wandb

                if wandb.run:
                    wandb.finish()
            except ImportError:
                pass

    ####################################
    ####################################

    def collect_training_trajectories(
        self,
        itr,
        initial_expertdata,
        collect_policy,
        num_transitions_to_sample,
        save_expert_data_to_disk=False,
    ):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # TODO collect `batch_size` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
        # print("\nCollecting data to be used for training...")

        paths, envsteps_this_batch = utils.sample_trajectories(
            self.env,
            collect_policy,
            num_transitions_to_sample,
            self.params["env"]["max_episode_length"],
        )

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN

        train_video_paths = None
        if self.logvideo:
            # print("\nCollecting train rollouts to be used for saving videos...")
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(
                self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True
            )
        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        # print("\nTraining agent using sampled data from replay buffer...")
        all_logs = []
        for train_step in range(self.params["alg"]["num_agent_train_steps_per_iter"]):

            obs, acs, rews, next_obs, dones = self.agent.sample(
                self.params["alg"]["train_batch_size"]
            )

            train_log = self.agent.train(
                obs,
                acs,
                rews,
                next_obs,
                dones,
            )
            all_logs.append(train_log)

        return all_logs

    ####################################
    ####################################

    def do_relabel_with_expert(self, expert_policy, paths):
        raise NotImplementedError
        # hw1/hw2, can ignore it b/c it's not used for this hw

    ####################################
    ####################################

    def perform_dqn_logging(self, all_logs, evaluate=True):
        if evaluate:
            last_log = all_logs[-1]

            episode_rewards = self.env.get_episode_rewards()
            if len(episode_rewards) > 0:
                self.mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                self.best_mean_episode_reward = max(
                    self.best_mean_episode_reward, self.mean_episode_reward
                )

            logs = OrderedDict()

            logs["Train_EnvstepsSoFar"] = self.agent.t
            print("Timestep %d" % (self.agent.t,))
            if self.mean_episode_reward > -5000:
                logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
            print("mean reward (100 episodes) %f" % self.mean_episode_reward)
            if self.best_mean_episode_reward > -5000:
                logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
            print("best mean reward %f" % self.best_mean_episode_reward)

            if self.start_time is not None:
                time_since_start = time.time() - self.start_time
                print("running time %f" % time_since_start)
                logs["TimeSinceStart"] = time_since_start

            logs.update(last_log)

            eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
                self.eval_env,
                self.agent.eval_policy,
                self.params["alg"]["eval_batch_size"],
                self.params["env"]["max_episode_length"],
            )

            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Buffer size"] = self.agent.replay_buffer.num_in_buffer

            sys.stdout.flush()

            for key, value in logs.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, self.agent.t)
            print("Done logging...\n\n")

            self.logger.flush()

    def perform_logging(
        self, itr, paths, eval_policy, train_video_paths, all_logs, evaluate=True
    ):

        if evaluate:
            # collect eval trajectories, for logging
            print("\nCollecting data for eval...")
            eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
                self.eval_env,
                eval_policy,
                self.params["alg"]["eval_batch_size"],
                self.params["env"]["max_episode_length"],
            )

            # save eval rollouts as videos in tensorboard event file
            if self.logvideo and train_video_paths != None:
                print("\nCollecting video rollouts eval")
                eval_video_paths = utils.sample_n_trajectories(
                    self.eval_env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True
                )

                # save train/eval videos
                print("\nSaving train rollouts as videos...")
                self.logger.log_paths_as_videos(
                    train_video_paths,
                    itr,
                    fps=self.fps,
                    max_videos_to_save=MAX_NVIDEO,
                    video_title="train_rollouts",
                )
                self.logger.log_paths_as_videos(
                    eval_video_paths,
                    itr,
                    fps=self.fps,
                    max_videos_to_save=MAX_NVIDEO,
                    video_title="eval_rollouts",
                )

            # save eval metrics
            if self.logmetrics:
                # returns, for logging
                train_returns = [path["reward"].sum() for path in paths]
                eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

                # episode lengths, for logging
                train_ep_lens = [len(path["reward"]) for path in paths]
                eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

                # decide what to log
                logs = OrderedDict()
                logs["Eval_AverageReturn"] = np.mean(eval_returns)
                logs["Eval_StdReturn"] = np.std(eval_returns)
                logs["Eval_MaxReturn"] = np.max(eval_returns)
                logs["Eval_MinReturn"] = np.min(eval_returns)
                logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

                logs["Train_AverageReturn"] = np.mean(train_returns)
                logs["Train_StdReturn"] = np.std(train_returns)
                logs["Train_MaxReturn"] = np.max(train_returns)
                logs["Train_MinReturn"] = np.min(train_returns)
                logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

                logs["Train_EnvstepsSoFar"] = self.total_envsteps
                logs["TimeSinceStart"] = time.time() - self.start_time
                logs.update(last_log)

                if itr == 0:
                    self.initial_return = np.mean(train_returns)
                logs["Initial_DataCollection_AverageReturn"] = self.initial_return

                # perform the logging
                for key, value in logs.items():
                    print("{} : {}".format(key, value))
                    try:
                        self.logger.log_scalar(value, key, itr)
                    except:
                        pdb.set_trace()
                print("Done logging...\n\n")

                self.logger.flush()

    def dump_density_graphs(self, itr):
        import matplotlib.pyplot as plt

        self.fig = plt.figure()
        filepath = lambda name: self.params["logging"][
            "logdir"
        ] + "/curr_{}.png".format(name)

        num_states = self.agent.replay_buffer.num_in_buffer - 2
        states = self.agent.replay_buffer.obs[:num_states]
        if num_states <= 0:
            return

        H, xedges, yedges = np.histogram2d(
            states[:, 0], states[:, 1], range=[[0.0, 1.0], [0.0, 1.0]], density=True
        )
        plt.imshow(np.rot90(H), interpolation="bicubic")
        plt.colorbar()
        plt.title("State Density")
        self.fig.savefig(filepath("state_density"), bbox_inches="tight")

        plt.clf()
        ii, jj = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
        obs = np.stack([ii.flatten(), jj.flatten()], axis=1)

        # Use get_prediction_error instead of forward_np
        density = self.agent.exploration_model.get_prediction_error(obs)
        density = density.reshape(ii.shape)
        plt.imshow(density[::-1])
        plt.colorbar()
        plt.title("RND Value")
        self.fig.savefig(filepath("rnd_value"), bbox_inches="tight")

        plt.clf()
        exploitation_values = self.agent.exploitation_critic.qa_values(obs).mean(-1)
        exploitation_values = exploitation_values.reshape(ii.shape)
        plt.imshow(exploitation_values[::-1])
        plt.colorbar()
        plt.title("Predicted Exploitation Value")
        self.fig.savefig(filepath("exploitation_value"), bbox_inches="tight")

        plt.clf()
        exploration_values = self.agent.exploration_critic.qa_values(obs).mean(-1)
        exploration_values = exploration_values.reshape(ii.shape)
        plt.imshow(exploration_values[::-1])
        plt.colorbar()
        plt.title("Predicted Exploration Value")
        self.fig.savefig(filepath("exploration_value"), bbox_inches="tight")

    def perform_logging_wandb(
        self, itr, paths, eval_policy, train_video_paths, all_logs
    ):
        """
        Perform logging for Weights & Biases (wandb)

        Args:
            itr: iteration number
            paths: paths collected during training
            eval_policy: policy used for evaluation
            train_video_paths: video paths collected during training
            all_logs: logs from training
        """

        if all_logs:
            last_log = all_logs[-1]

            # Log the last training metrics
            for key, value in last_log.items():
                wandb.log({f"train/{key}": value}, step=itr)

        if self.logmetrics or self.logvideo:

            eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
                # self.eval_env,
                self.env,
                eval_policy,
                self.params["alg"]["eval_batch_size"],
                self.params["env"]["max_episode_length"],
            )

        if self.logmetrics and len(eval_paths) > 0:
            # Log stats from evaluation
            returns = [eval_path["reward"].sum() for eval_path in eval_paths]
            ep_lengths = [len(eval_path["reward"]) for eval_path in eval_paths]

            metrics_dict = {
                "eval/AverageReturn": np.mean(returns),
                "eval/StdReturn": np.std(returns),
                "eval/MaxReturn": np.max(returns),
                "eval/MinReturn": np.min(returns),
                "eval/AverageEpLen": np.mean(ep_lengths),
                "eval/TotalEnvInteracts": self.total_envsteps,
                "eval/Iteration": itr,
            }

            wandb.log(metrics_dict, step=itr)

            # Log stats from exploration
            if isinstance(self.agent, ExplorationOrExploitationAgent):
                # Log exploration-specific metrics
                if (
                    hasattr(self.agent, "exploration_stats")
                    and self.agent.exploration_stats
                ):
                    for key, value in self.agent.exploration_stats.items():
                        wandb.log({f"exploration/{key}": value}, step=itr)

        if isinstance(self.agent, ExplorationOrExploitationAgent) and itr > self.params["alg"]["num_exploration_steps"]:
            eval_q_metrics = self.agent.compare_eval_q_values(eval_paths)
            wandb.log(eval_q_metrics, step=itr)

        if self.logmetrics:
            # returns, for logging
            # train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            # train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            # logs["Train_AverageReturn"] = np.mean(train_returns)
            # logs["Train_StdReturn"] = np.std(train_returns)
            # logs["Train_MaxReturn"] = np.max(train_returns)
            # logs["Train_MinReturn"] = np.min(train_returns)
            # logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            # if itr == 0:
            #     self.initial_return = np.mean(train_returns)
            # logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            self.logger.flush()

        # Log videos

        if self.logvideo and len(eval_paths) > 0:
            # Create videos directory if it doesn't exist
            videos_dir = os.path.join(self.params["logging"]["logdir"], "videos")
            os.makedirs(videos_dir, exist_ok=True)

            # Save eval video
            eval_video_path = os.path.join(videos_dir, f"eval_video_itr_{itr}.mp4")
            video_saved = self.logger.save_video_to_path(eval_paths, eval_video_path)

            # Log video to wandb
            if video_saved:
                wandb.log(
                    {"eval/video": wandb.Video(eval_video_path, fps=10, format="mp4")},
                    step=itr,
                )

            # If we have training videos, log those too
            if train_video_paths and len(train_video_paths) > 0:
                train_video_path = os.path.join(
                    videos_dir, f"train_video_itr_{itr}.mp4"
                )
                train_video_saved = self.logger.save_video_to_path(
                    train_video_paths, train_video_path
                )
                if train_video_saved:
                    wandb.log(
                        {
                            "train/video": wandb.Video(
                                train_video_path, fps=10, format="mp4"
                            )
                        },
                        step=self.total_envsteps,
                    )

        # Log density plots for ExplorationOrExploitationAgent

        if isinstance(self.agent, ExplorationOrExploitationAgent) and hasattr(
            self.agent, "density_model"
        ):
            # Create density_plots directory if it doesn't exist
            density_plots_dir = os.path.join(
                self.params["logging"]["logdir"], "density_plots"
            )
            os.makedirs(density_plots_dir, exist_ok=True)

            # Generate density plot
            density_plot_path = os.path.join(
                density_plots_dir, f"density_itr_{itr}.png"
            )

            # Create and save the density plot
            fig = plt.figure(figsize=(10, 8))
            self.agent.density_model.plot_density(fig)
            plt.savefig(density_plot_path)
            plt.close(fig)

            # Log the density plot to wandb
            wandb.log(
                {"exploration/density_plot": wandb.Image(density_plot_path)},
                step=self.total_envsteps,
            )

        # Flush to make sure all data is sent to wandb
        try:
            wandb.log({})
        except Exception as e:
            print(f"Error flushing wandb logs: {e}")

    def _convert_params_for_wandb(self, params, visited=None):
        """
        Convert params dictionary into a format that wandb can serialize.
        Handles nested dictionaries, lists, and non-serializable objects.
        Prevents infinite recursion from circular references.

        Args:
            params: The parameters dictionary to convert
            visited: Set of object ids already visited (to detect cycles)

        Returns:
            A wandb-compatible configuration dictionary
        """
        if params is None:
            return None

        # Initialize visited set if this is the top-level call
        if visited is None:
            visited = set()

        # Check for circular references
        obj_id = id(params)
        if obj_id in visited:
            return "<circular reference>"

        # Add current object to visited set
        visited.add(obj_id)

        if isinstance(params, dict):
            return {
                k: self._convert_params_for_wandb(v, visited.copy())
                for k, v in params.items()
            }

        elif isinstance(params, list):
            return [
                self._convert_params_for_wandb(item, visited.copy()) for item in params
            ]

        elif isinstance(params, tuple):
            return tuple(
                self._convert_params_for_wandb(item, visited.copy()) for item in params
            )

        elif hasattr(params, "__dict__") and not isinstance(params, type):
            # Convert objects to dictionaries, but avoid class objects
            try:
                return self._convert_params_for_wandb(vars(params), visited.copy())
            except:
                return str(params)

        elif callable(params):
            # Handle functions or callable objects
            return (
                f"<function: {params.__name__}>"
                if hasattr(params, "__name__")
                else str(params)
            )

        else:
            # Test if the object is JSON serializable
            try:
                import json

                json.dumps(params)
                return params
            except (TypeError, OverflowError):
                # If not serializable, convert to string
                return str(params)
