from ml_collections import ConfigDict

from experiments.configs import sac_config


def get_config(updates=None):
    config = sac_config.get_config()

    config.critic_ensemble_size = 10
    config.critic_subsample_size = 2

    config.policy_network_kwargs.use_layer_norm = True
    config.critic_network_kwargs.use_layer_norm = True

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    return config
