from ml_collections import ConfigDict


def get_config(updates=None):
    config = ConfigDict()
    config.discount = 0.99
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.soft_target_update_rate = 5e-3
    config.critic_ensemble_size = 2
    config.critic_subsample_size = None
    config.autotune_entropy = True
    config.temperature_init = 1.0

    # arch
    config.critic_network_kwargs = ConfigDict(
        {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        }
    )
    config.policy_network_kwargs = ConfigDict(
        {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        }
    )
    config.policy_kwargs = ConfigDict(
        {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
        }
    )

    config.actor_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 1e-4,
        }
    )
    config.critic_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )
    config.temperature_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 1e-4,
        }
    )

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())

    return config
