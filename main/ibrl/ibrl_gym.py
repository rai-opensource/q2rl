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
from backend.common.evaluation import evaluate_and_record_videos_ibrl
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


flags.DEFINE_integer(
    "n_eval_trajs", 20, "Number of trajectories to use for each evaluation."
)

# wandb
flags.DEFINE_string("exp_name", "", "Experiment name for wandb logging")
flags.DEFINE_string("project", None, "Wandb project folder")
flags.DEFINE_string("group", None, "Wandb group of the experiment")
flags.DEFINE_bool("debug", False, "If true, no logging to wandb")
flags.DEFINE_bool("get_new_rollouts", False, "get new rollouts")
flags.DEFINE_integer("replay_buffer_capacity", int(2e6), "Replay buffer capacity")

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
    # do not clip adroit actions online following CalQL repo
    # https://github.com/nakamotoo/Cal-QL
    env_type = get_env_type(FLAGS.env)
    finetune_env = make_gym_env(
        env_name=FLAGS.env,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        scale_and_clip_action=env_type in ("antmaze", "kitchen", "locomotion"),
        action_clip_lim=FLAGS.clip_action,
        seed=FLAGS.seed,
    )
    eval_env = make_gym_env(
        env_name=FLAGS.env,
        scale_and_clip_action=env_type in ("antmaze", "kitchen", "locomotion"),
        action_clip_lim=FLAGS.clip_action,
        seed=FLAGS.seed + 1000,
    )


    """
    Initialize agent
    """
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, construct_rng = jax.random.split(rng)
    bc_agent = agents["bc"].create(
        rng=construct_rng,
        observations=finetune_env.observation_space.sample(),
        actions=finetune_env.action_space.sample(),
        encoder_def=None,
    )

    online_replay_buffer_type = ReplayBuffer_IBRL
    online_replay_buffer = online_replay_buffer_type(
        finetune_env.observation_space,
        finetune_env.action_space,
        capacity=FLAGS.replay_buffer_capacity,
        seed=FLAGS.seed,
        discount=FLAGS.discount,
    )


    assert os.path.exists(FLAGS.resume_path_bc), "resume path does not exist"
    bc_agent = checkpoints.restore_checkpoint(FLAGS.resume_path_bc, target=bc_agent)

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
            "average_return": np.mean([np.sum(t["rewards"]) for t in trajs]),
            "average_traj_length": np.mean([len(t["rewards"]) for t in trajs]),
        }
        if env_type == "adroit-binary":
            # adroit
            eval_info["success_rate"] = np.mean(
                [any(d["goal_achieved"] for d in t["infos"]) for t in trajs]
            )
        elif env_type == "kitchen":
            # kitchen
            eval_info["num_stages_solved"] = np.mean([t["rewards"][-1] for t in trajs])
            eval_info["success_rate"] = np.mean([t["rewards"][-1] for t in trajs]) / 4

        wandb_logger.log({"evaluation": eval_info}, step=step_number)
    
    step = agent.state.step
   
    observation, info = finetune_env.reset()
    done = False
    timer = Timer()
    sample_info = None
    for _ in tqdm.tqdm(range(step, FLAGS.num_online_steps)):

        rng, action_rng = jax.random.split(rng)
        ##bc actions
        bc_action = bc_agent.sample_actions(observation, argmax=True)
        action, _ = agent.sample_actions(observation, bc_actions=bc_action, seed=action_rng)
        
        next_observation, reward, done, truncated, info = finetune_env.step(action)
        next_bc_action = bc_agent.sample_actions(next_observation, argmax=True)

        transition = dict(
            observations=observation,
            next_observations=next_observation,
            actions=action,
            rewards=reward,
            masks=1.0 - done,
            dones=1.0 if (done or truncated) else 0,
            next_bc_actions=next_bc_action,
            bc_actions=bc_action
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
                bc_fn = partial(
                    bc_agent.sample_actions, argmax=True
                )
               
                eval_func = partial(
                    evaluate_and_record_videos_ibrl,
                    clip_action=FLAGS.clip_action,
                    step_number=step,
                    record_video=False,
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
