from ast import If
import os
from functools import partial

import gym
import jax
import numpy as np
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags

from experiments.configs.ensemble_config import add_redq_config
from backend.agents import agents
from backend.common import wandb
from backend.common.evaluation import evaluate_and_record_videos_ibrl_robosuite
from backend.common.wandb import WandBLogger
from backend.data.replay_buffer import ReplayBuffer, ReplayBuffer_IBRL
from backend.envs.adroit_binary_dataset import get_hand_dataset_with_mc_calculation
from backend.envs.d4rl_dataset import (
    get_d4rl_dataset,
    get_d4rl_dataset_with_mc_calculation,
)
from backend.envs.env_common import get_env_type, make_gym_env
from backend.utils.timer_utils import Timer
from backend.utils.train_utils import concatenate_batches, subsample_batch
from backend.envs.env_common import calc_return_to_go

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB
import robomimic.utils.env_utils as EnvUtils
from backend.envs.wrappers import RobosuiteStateWrapper
import h5py


import collections
import pickle
from copy import deepcopy
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
FLAGS = flags.FLAGS

# env
flags.DEFINE_string("env", "antmaze-large-diverse-v2", "Environemnt to use")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", -1.0, "Reward bias.")
flags.DEFINE_float("discount", 0.99, "discount")

flags.DEFINE_float(
    "clip_action",
    0.99999,
    "Clip actions to be between [-n, n]. This is needed for tanh policies.",
)

# training
flags.DEFINE_integer("num_eval_rollouts", 500, "Number of offline epochs.")
flags.DEFINE_integer("num_q_est_steps", 1_000_000, "Number of policy training steps.")
flags.DEFINE_integer("num_online_steps", 1_000_000, "Number of policy training steps.")

# agent
flags.DEFINE_string("bc_agent", "calql", "what BC agent to use")
flags.DEFINE_string("agent", "calql", "what RL agent to use")

# experiment house keeping
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "save_dir",
    os.path.expanduser("~/q2rl_log"),
    "Directory to save the logs and checkpoints",
)
flags.DEFINE_string("resume_path_bc", "", "Path to resume for bc policies")
flags.DEFINE_string("resume_path_agent", "", "Path to resume for RL agent")
flags.DEFINE_integer("log_interval", 5_000, "Log every n steps")
flags.DEFINE_integer("eval_interval", 20_000, "Evaluate every n steps")
flags.DEFINE_integer("video_interval", 20_000, "Evaluate every n steps")
flags.DEFINE_integer("save_interval", 50_000, "Evaluate every n steps")
flags.DEFINE_integer("batch_size", 256, "batch size for training")
flags.DEFINE_integer("utd", 1, "update-to-data ratio of the critic")
flags.DEFINE_bool("render", False, "Render the environment")


flags.DEFINE_integer(
    "n_eval_trajs", 20, "Number of trajectories to use for each evaluation."
)

# wandb
flags.DEFINE_string("exp_name", "", "Experiment name for wandb logging")
flags.DEFINE_string("project", None, "Wandb project folder")
flags.DEFINE_string("group", None, "Wandb group of the experiment")
flags.DEFINE_bool("debug", False, "If true, no logging to wandb")
flags.DEFINE_string("demo_path", None, "path to load the demo replay buffer from")
flags.DEFINE_string("data_filter_key", None, "Key to filter data")
flags.DEFINE_bool("get_demo_buffer", False, "Load demo trajs into demo buffer")
flags.DEFINE_integer("replay_buffer_capacity", int(2e6), "Replay buffer capacity")
flags.DEFINE_float("max_traj_length", 200, "Maximum trajectory length.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def main(_):
    """
    wandb and logging
    """
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "q2rl" or FLAGS.project,
            "group": FLAGS.group,
            "exp_descriptor": f"{FLAGS.group}_{FLAGS.env}_{FLAGS.agent}_seed{FLAGS.seed}",
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=dict(),
        random_str_in_identifier=True,
        disable_online_logging=FLAGS.debug,
    )

    save_dir = os.path.join(
        FLAGS.save_dir,
        f"{FLAGS.group}",
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    """
    env
    """
    torch_device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    robomimic_agent, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=FLAGS.resume_path_bc, device=torch_device, verbose=True)
    rm_env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        render=FLAGS.render, 
        render_offscreen=True, 
        verbose=True,
        )
    rm_env.env.ignore_done = (
        False  # Fix a hardcoded ignore_done=True in robomimic env init
    )
    rm_env.env.reward_shaping = (
        False  #Reward shaping should be false
    )
    rm_env.env.horizon = (
        FLAGS.max_traj_length  #Changing horizon of the env
    )
    suite_env = rm_env.env
    finetune_env = RobosuiteStateWrapper(suite_env, reward_scale=FLAGS.reward_scale, reward_bias=FLAGS.reward_bias)

    ##Robosuite eval env 
    _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=FLAGS.resume_path_bc, device=torch_device, verbose=True)
    eval_rm_env, _ = FileUtils.env_from_checkpoint(
    ckpt_dict=ckpt_dict, 
    render=FLAGS.render, 
    render_offscreen=True, 
    verbose=True,
    )
    eval_rm_env.env.ignore_done = (
        False  # Fix a hardcoded ignore_done=True in robomimic env init
    )
    eval_rm_env.env.reward_shaping = (
        True  #Reward shaping should be true for eval env
    )
    eval_rm_env.env.horizon = (
        FLAGS.max_traj_length  #Changing horizon of the env
    )
    eval_suite_env = eval_rm_env.env
    eval_env = RobosuiteStateWrapper(eval_suite_env, reward_scale=FLAGS.reward_scale, reward_bias=FLAGS.reward_bias)

    def get_robomimic_obs(obs):
        robomimic_obs = collections.OrderedDict()
        robomimic_obs["robot0_eef_pos"] = obs[:3]
        robomimic_obs["robot0_eef_quat"] = obs[3:7]
        robomimic_obs["robot0_gripper_qpos"] = obs[7:9]
        robomimic_obs["object"] = obs[9:]
        robomimic_obs = ObsUtils.process_obs_dict(robomimic_obs)
        return robomimic_obs
    
    robomimic_agent.start_episode()
    observation, info = finetune_env.reset()

    """
    Initialize agent
    """
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, construct_rng = jax.random.split(rng)

   
    online_replay_buffer_type = ReplayBuffer_IBRL
    online_replay_buffer = online_replay_buffer_type(
        finetune_env.observation_space,
        finetune_env.action_space,
        capacity=FLAGS.replay_buffer_capacity,
        seed=FLAGS.seed,
        discount=FLAGS.discount,
    )

    agent = agents[FLAGS.agent].create(
        rng=construct_rng,
        observations=finetune_env.observation_space.sample(),
        actions=finetune_env.action_space.sample(),
        encoder_def=None,
        shared_encoder=False,
        **FLAGS.config.agent_kwargs,
    )


    """
    eval function
    """
    def evaluate_and_log_results(
        eval_env,
        policy_fn,
        bc_fn,
        eval_func,
        step_number,
        wandb_logger,
        n_eval_trajs=FLAGS.n_eval_trajs,
    ):
        stats, trajs = eval_func(
            policy_fn,
            bc_fn,
            eval_env,
            n_eval_trajs,
        )

        eval_info = {
            "average_return": stats["average_return"],
            "average_traj_length": stats["average_traj_length"],
            "success_rate": stats["success_rate"],
            "bc_actions": stats["bc_actions"],
        }
        
        wandb_logger.log({"evaluation": eval_info}, step=step_number)
    
    ## If we want to initialize IBRL replay buffer with demo data
    if FLAGS.get_demo_buffer:
        print_green("Loading demo data into the replay buffer...")
        f = h5py.File(FLAGS.demo_path, "r")
        for demo in f["mask"][FLAGS.data_filter_key][:]:
            obs_object = f["data"][demo]["obs"]["object"][:]
            obs_robot0_eef_pos = f["data"][demo]["obs"]["robot0_eef_pos"][:]
            obs_robot0_eef_quat = f["data"][demo]["obs"]["robot0_eef_quat"][:]
            obs_robot0_gripper_qpos = f["data"][demo]["obs"]["robot0_gripper_qpos"][:]
            obs = np.concatenate([obs_robot0_eef_pos, obs_robot0_eef_quat, obs_robot0_gripper_qpos, obs_object], axis=-1)
            actions = f["data"][demo]["actions"][:]
            rewards = f["data"][demo]["rewards"][:] * FLAGS.reward_scale + FLAGS.reward_bias
            dones = f["data"][demo]["dones"][:]

            for i in range(len(actions)):
                transition = dict(
                    observations=obs[i],
                    next_observations=obs[i+1] if i < len(actions) - 1 else obs[i],
                    actions=actions[i],
                    rewards=rewards[i],
                    masks=1.0 - dones[i],
                    dones=dones[i],
                    next_bc_actions=actions[i+1] if i < len(actions) - 1 else actions[i],
                    bc_actions=actions[i],
                )
                online_replay_buffer.insert(transition)    
    
    step = agent.state.step
    observation, info = finetune_env.reset()
    done = False

    timer = Timer()
    for _ in tqdm.tqdm(range(step, FLAGS.num_online_steps)):
        rng, action_rng = jax.random.split(rng)
        ##bc actions
        bc_action = robomimic_agent(ob=get_robomimic_obs(observation))
        action, _ = agent.sample_actions(observation, bc_actions=bc_action, seed=action_rng)
        
        next_observation, reward, done, truncated, info = finetune_env.step(
            action
        )

        next_bc_action = robomimic_agent(ob=get_robomimic_obs(next_observation))
        transition = dict(
            observations=observation,
            next_observations=next_observation,
            actions=action,
            rewards=reward,
            masks=1.0 - done,
            dones=1.0 if (done or truncated) else 0,
            next_bc_actions=next_bc_action,
            bc_actions=bc_action,
        )
        online_replay_buffer.insert(transition)

        observation = next_observation
        if done or truncated:
            observation, info = finetune_env.reset()
            done = False

        
        batch = online_replay_buffer.sample(FLAGS.batch_size)

        agent, update_info = agent.update_high_utd(
                            batch,
                            utd_ratio=FLAGS.utd,
                        )
        

        if step % FLAGS.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_logger.log({"training": update_info}, step=step)
        
        if step % FLAGS.eval_interval == 0:
            timer.tick("total")
            with timer.context("eval step"):
                
                policy_fn = partial(
                    agent.sample_actions, argmax=True
                )
                bc_fn = robomimic_agent
               
                eval_func = partial(
                    evaluate_and_record_videos_ibrl_robosuite,
                    clip_action=FLAGS.clip_action,
                    step_number=step,
                    record_video=True if step % FLAGS.video_interval == 0 else False,
                    wandb_logger=wandb_logger,
                )
              
                evaluate_and_log_results(
                    eval_env=eval_env,
                    policy_fn=policy_fn,
                    bc_fn=bc_fn,
                    eval_func=eval_func,
                    step_number=step,
                    wandb_logger=wandb_logger,
                )
            timer.tock("total")
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)



        if step % FLAGS.save_interval == 0 or step == FLAGS.num_q_est_steps:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=step, keep=30
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        
        """
        Advance Step
        """
        step += 1
    

    print("Done training!")

if __name__ == "__main__":
    app.run(main)
