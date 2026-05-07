from ml_collections import ConfigDict


def add_redq_config(config, updates=None):
    # use ensemble and layer norm
    config.critic_ensemble_size = 10
    config.critic_subsample_size = 2
    config.policy_network_kwargs.use_layer_norm = True
    config.critic_network_kwargs.use_layer_norm = True

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())

    return config
