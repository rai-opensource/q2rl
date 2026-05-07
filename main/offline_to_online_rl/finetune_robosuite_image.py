import os
from functools import partial

import gym
import h5py
import jax
import numpy as np
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags
from copy import deepcopy

from experiments.configs.ensemble_config import add_redq_config
from backend.agents import agents
from backend.common.evaluation import evaluate_and_record_videos_baseline_robsuite
from backend.common.wandb import WandBLogger
from backend.data.replay_buffer import ReplayBuffer, ReplayBufferMC
from backend.envs.adroit_binary_dataset import get_hand_dataset_with_mc_calculation
from backend.envs.d4rl_dataset import (
    get_d4rl_dataset,
    get_d4rl_dataset_with_mc_calculation,
)
from backend.envs.env_common import get_env_type, make_gym_env
from backend.utils.timer_utils import Timer
from backend.utils.train_utils import concatenate_batches, subsample_batch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB
import robomimic.utils.env_utils as EnvUtils
from backend.envs.wrappers import RobosuiteImageWrapper
import collections


FLAGS = flags.FLAGS

# env
flags.DEFINE_string("env", "antmaze-large-diverse-v2", "Environemnt to use")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", -1.0, "Reward bias.")
flags.DEFINE_float("max_traj_length", 200, "Maximum trajectory length.")
flags.DEFINE_float(
    "clip_action",
    0.99999,
    "Clip actions to be between [-n, n]. This is needed for tanh policies.",
)

# training
flags.DEFINE_integer("num_offline_steps", 1_000_000, "Number of offline epochs.")
flags.DEFINE_integer("num_online_steps", 500_000, "Number of online epochs.")
flags.DEFINE_float(
    "offline_data_ratio",
    0.0,
    "How much offline data to retain in each online batch update",
)
flags.DEFINE_string(
    "online_sampling_method",
    "mixed",
    """Method of sampling data during online update: mixed or append.
    `mixed` samples from a mix of offline and online data according to offline_data_ratio.
    `append` adds offline data to replay buffer and samples from it.""",
)
flags.DEFINE_bool(
    "online_use_cql_loss",
    True,
    """When agent is CQL/CalQL, whether to use CQL loss for the online phase (use SAC loss if False)""",
)
flags.DEFINE_integer(
    "warmup_steps", 0, "number of warmup steps (WSRL) before performing online updates"
)

# agent
flags.DEFINE_string("agent", "calql", "what RL agent to use")
flags.DEFINE_integer("utd", 1, "update-to-data ratio of the critic")
flags.DEFINE_integer("batch_size", 256, "batch size for training")
flags.DEFINE_integer("replay_buffer_capacity", int(5e4), "Replay buffer capacity")
flags.DEFINE_bool("use_redq", False, "Use an ensemble of Q-functions for the agent")

# experiment house keeping
flags.DEFINE_integer("seed", 20, "Random seefd.")
flags.DEFINE_string(
    "save_dir",
    os.path.expanduser("~/q2rl_log"),
    "Directory to save the logs and checkpoints",
)
flags.DEFINE_string("resume_path", "", "Path to resume from")
flags.DEFINE_string("resume_path_bc", "", "Path to resume for bc policies")
flags.DEFINE_bool("render", False, "Render the environment")

flags.DEFINE_integer("log_interval", 5_000, "Log every n steps")
flags.DEFINE_integer("eval_interval", 20_000, "Evaluate every n steps")
flags.DEFINE_integer("save_interval", 100_000, "Save every n steps.")
flags.DEFINE_integer(
    "n_eval_trajs", 20, "Number of trajectories to use for each evaluation."
)
flags.DEFINE_bool("deterministic_eval", True, "Whether to use deterministic evaluation")

# wandb
flags.DEFINE_string("exp_name", "", "Experiment name for wandb logging")
flags.DEFINE_string("project", None, "Wandb project folder")
flags.DEFINE_string("group", None, "Wandb group of the experiment")
flags.DEFINE_bool("debug", False, "If true, no logging to wandb")
flags.DEFINE_bool("get_demo_buffer", False, "Load demo trajs into demo buffer")
flags.DEFINE_string("demo_path", None, "path to load the demo replay buffer from")
flags.DEFINE_string("data_filter_key", None, "Key to filter data")
flags.DEFINE_integer("video_interval", 20_000, "Evaluate every n steps")


config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    """
    house keeping
    """
    assert FLAGS.online_sampling_method in [
        "mixed",
        "append",
    ], "incorrect online sampling method"

    if FLAGS.use_redq:
        FLAGS.config.agent_kwargs = add_redq_config(FLAGS.config.agent_kwargs)

    min_steps_to_update = FLAGS.batch_size * (1 - FLAGS.offline_data_ratio)
   
    """
    wandb and logging
    """
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "q2rl" or FLAGS.project,
            "group": FLAGS.group,
            "exp_descriptor": f"{FLAGS.exp_name}_{FLAGS.env}_{FLAGS.agent}_seed{FLAGS.seed}",
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        random_str_in_identifier=True,
        disable_online_logging=FLAGS.debug,
    )

    save_dir = os.path.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
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
    finetune_env = RobosuiteImageWrapper(suite_env, reward_scale=FLAGS.reward_scale, reward_bias=FLAGS.reward_bias)

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
        True  #Reward shaping should be true
    )
    eval_rm_env.env.horizon = (
        FLAGS.max_traj_length  #Changing horizon of the env
    )
    eval_suite_env = eval_rm_env.env
    eval_env = RobosuiteImageWrapper(eval_suite_env, reward_scale=FLAGS.reward_scale, reward_bias=FLAGS.reward_bias)

    """
    replay buffer
    """
    dataset_buffer_type = ReplayBufferMC 
    dataset_buffer = dataset_buffer_type(
        finetune_env.observation_space,
        finetune_env.action_space,
        capacity=FLAGS.replay_buffer_capacity,
        seed=FLAGS.seed,
        discount=FLAGS.config.agent_kwargs.discount
    )

    """
    load dataset
    """
    if FLAGS.get_demo_buffer:
        f = h5py.File(FLAGS.demo_path, "r")
        for demo in f["mask"][FLAGS.data_filter_key][:]:
            obs_robot0_eef_pos = f["data"][demo]["obs"]["robot0_eef_pos"][:]
            obs_robot0_eef_quat = f["data"][demo]["obs"]["robot0_eef_quat"][:]
            obs_robot0_gripper_qpos = f["data"][demo]["obs"]["robot0_gripper_qpos"][:]
            state = np.concatenate([obs_robot0_eef_pos, obs_robot0_eef_quat, obs_robot0_gripper_qpos], axis=-1)
            front = f["data"][demo]["obs"]["agentview_image"][:]
            wrist = f["data"][demo]["obs"]["robot0_eye_in_hand_image"][:]
            obs = []
            for i in range(len(state)):
                observation = collections.OrderedDict()
                observation["state"] = state[i]
                observation["front"] = front[i]
                observation["wrist"] = wrist[i]
                obs.append(observation)
            actions = f["data"][demo]["actions"][:]
            rewards = f["data"][demo]["rewards"][:] * FLAGS.reward_scale + FLAGS.reward_bias
            dones = f["data"][demo]["dones"][:]

            for i in range(len(actions)):
                rl_obs = obs[i]
                next_obs = obs[i+1] if i < len(actions) - 1 else obs[i]
                rl_next_obs = next_obs

                transition = dict(
                    observations=rl_obs,
                    next_observations=rl_next_obs,
                    actions=actions[i],
                    rewards=rewards[i],
                    masks=1.0 - dones[i],
                    dones=dones[i],
                    )
                dataset_buffer.insert(transition)
    

    """
    replay buffer
    """
    replay_buffer_type = ReplayBufferMC
    replay_buffer = replay_buffer_type(
        finetune_env.observation_space,
        finetune_env.action_space,
        capacity=FLAGS.replay_buffer_capacity,
        seed=FLAGS.seed,
        discount=FLAGS.config.agent_kwargs.discount,
    )

    """
    Initialize agent
    """
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, construct_rng = jax.random.split(rng)
    example_batch = dataset_buffer.sample(1)
    agent = agents[FLAGS.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        actions=example_batch["actions"],
        encoder_def=None,
        **FLAGS.config.agent_kwargs,
        image_keys=["front","wrist"],
    )

    if FLAGS.resume_path != "":
        assert os.path.exists(FLAGS.resume_path), "resume path does not exist"
        agent = checkpoints.restore_checkpoint(FLAGS.resume_path, target=agent)
        

    """
    eval function
    """

    def evaluate_and_log_results(
        eval_env,
        policy_fn,
        eval_func,
        step_number,
        wandb_logger,
        n_eval_trajs=FLAGS.n_eval_trajs,
    ):
        stats, trajs = eval_func(
            policy_fn=policy_fn,
            env=eval_env,
            num_episodes=n_eval_trajs,
        )

        eval_info = {
            "average_return": stats["average_return"],
            "average_traj_length": stats["average_traj_length"],
            "success_rate": stats["success_rate"],
        }

        wandb_logger.log({"evaluation": eval_info}, step=step_number)

    """
    training loop
    """
    timer = Timer()
    step = int(agent.state.step)  # 0 for new agents, or load from pre-trained
    is_online_stage = False
    observation, info = finetune_env.reset()
    done = False  # env done signal

    for _ in tqdm.tqdm(range(step, FLAGS.num_offline_steps + FLAGS.num_online_steps)):
        """
        Switch from offline to online
        """
        if not is_online_stage and step >= FLAGS.num_offline_steps:
            logging.info("Switching to online training")
            is_online_stage = True

            # option for CQL and CalQL to change the online alpha, and whether to use CQL regularizer
            if FLAGS.agent in ("cql", "calql"):
                online_agent_configs = {
                    "cql_alpha": FLAGS.config.agent_kwargs.get(
                        "online_cql_alpha", None
                    ),
                    "use_cql_loss": FLAGS.online_use_cql_loss,
                }
                agent.update_config(online_agent_configs)

        timer.tick("total")

        """
        Env Step
        """
        with timer.context("env step"):
            if is_online_stage:
                rng, action_rng = jax.random.split(rng)
                action = agent.sample_actions(observation, seed=action_rng)
                next_observation, reward, done, truncated, info = finetune_env.step(
                    action
                )

                transition = dict(
                    observations=observation,
                    next_observations=next_observation,
                    actions=action,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=1.0 if (done or truncated) else 0,
                )
                replay_buffer.insert(transition)

                observation = next_observation
                if done or truncated:
                    observation, info = finetune_env.reset()
                    done = False

        """
        Updates
        """
        with timer.context("update"):
            # offline updates
            if not is_online_stage:
                batch = dataset_buffer.sample(FLAGS.batch_size)
                agent, update_info = agent.update(
                    batch,
                )

            # online updates
            else:
                if step - FLAGS.num_offline_steps <= max(
                    FLAGS.warmup_steps, min_steps_to_update
                ):
                    # no updates during warmup
                    pass
                else:
                    # do online updates, gather batch
                    if FLAGS.online_sampling_method == "mixed":
                        # batch from a mixing ratio of offline and online data
                        batch_size_offline = int(
                            FLAGS.batch_size * FLAGS.offline_data_ratio
                        )
                        batch_size_online = FLAGS.batch_size - batch_size_offline
                        online_batch = replay_buffer.sample(batch_size_online)
                        offline_batch = dataset_buffer.sample(batch_size_offline)
                        # update with the combined batch
                        batch = concatenate_batches([online_batch, offline_batch])
                    elif FLAGS.online_sampling_method == "append":
                        # batch from online replay buffer
                        batch = replay_buffer.sample(FLAGS.batch_size)
                    else:
                        raise RuntimeError("Incorrect online sampling method")

                    # update
                    if FLAGS.utd > 1:
                        agent, update_info = agent.update_high_utd(
                            batch,
                            utd_ratio=FLAGS.utd,
                        )
                    else:
                        agent, update_info = agent.update(
                            batch,
                        )

        """
        Advance Step
        """
        step += 1

        """
        Evals
        """
        eval_steps = (
            FLAGS.num_offline_steps,  # finish offline training
            FLAGS.num_offline_steps + 1,  # start of online training
            FLAGS.num_offline_steps + FLAGS.num_online_steps,  # end of online training
        )
        if step % FLAGS.eval_interval == 0 or step in eval_steps:
            logging.info("Evaluating...")
            with timer.context("evaluation"):
                policy_fn = partial(
                    agent.sample_actions, argmax=FLAGS.deterministic_eval
                )
                eval_func = partial(
                        evaluate_and_record_videos_baseline_robsuite,
                        bc_fn=None,
                        clip_action=FLAGS.clip_action,
                        step_number=step,
                        record_video=True if step % FLAGS.video_interval == 0 else False,
                        wandb_logger=wandb_logger,
                    )

                evaluate_and_log_results(
                    eval_env=eval_env,
                    policy_fn=policy_fn,
                    eval_func=eval_func,
                    step_number=step,
                    wandb_logger=wandb_logger,
                )

        """
        Save Checkpoint
        """
        if step % FLAGS.save_interval == 0 or step == FLAGS.num_offline_steps:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=step, keep=30
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        """
        Logging
        """
        if step % FLAGS.log_interval == 0:
            # check if update_info is available (False during warmup)
            if "update_info" in locals():
                update_info = jax.device_get(update_info)
                wandb_logger.log({"training": update_info}, step=step)

            wandb_logger.log({"timer": timer.get_average_times()}, step=step)


if __name__ == "__main__":
    app.run(main)
