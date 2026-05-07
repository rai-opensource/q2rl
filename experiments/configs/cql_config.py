from ml_collections import ConfigDict
import numpy as np

from experiments.configs import sac_config


def get_config(updates=None):
    config = sac_config.get_config()

    config.cql_n_actions = 10
    config.cql_action_sample_method = "uniform"
    config.cql_max_target_backup = True
    config.cql_importance_sample = True
    config.cql_autotune_alpha = False
    config.cql_alpha_lagrange_init = 1.0
    config.cql_alpha_lagrange_otpimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )
    config.cql_target_action_gap = 1.0
    config.cql_temp = 1.0
    config.cql_alpha = 5.0
    config.cql_clip_diff_min = -np.inf
    config.cql_clip_diff_max = np.inf
    config.use_td_loss = True  # set this to False to essentially do BC
    config.use_cql_loss = True  # set this to False to default to SAC

    # Cal-QL
    config.use_calql = False
    config.calql_bound_random_actions = False

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    return config
