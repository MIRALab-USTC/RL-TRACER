
def handle_config(TrainConfig):
    if not TrainConfig.default_setting:
        return TrainConfig

    if 'tracer' not in TrainConfig.experiment:
        return TrainConfig

    '''TRACER Hyperparameters'''
    key = TrainConfig.task_name.split('-')[0]
    TrainConfig.num_q = 5

    if TrainConfig.corrupt_cfg.corruption_mode == 'random':
        if TrainConfig.corrupt_cfg.corrupt_obs and \
                TrainConfig.corrupt_cfg.corrupt_act and \
                TrainConfig.corrupt_cfg.corrupt_reward and \
                TrainConfig.corrupt_cfg.corrupt_dynamics:
            TrainConfig.Agent.sigma = 0.1
            TrainConfig.Agent.obser_sigma = 0.1
            TrainConfig.Agent.quantile = 0.25
            TrainConfig.Agent.critic_target_update_freq = 2
            TrainConfig.Agent.learning_rate = 3e-4

        elif TrainConfig.corrupt_cfg.corrupt_obs:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.obser_sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.25,
                'walker2d': 0.25,
                'hopper': 0.25, 
            }[key]
            TrainConfig.Agent.critic_target_update_freq = {
                'halfcheetah': 1,
                'walker2d': 1,
                'hopper': 1, 
            }[key]
            TrainConfig.Agent.learning_rate = {
                'halfcheetah': 3e-4,
                'walker2d': 3e-4,
                'hopper': 3e-4,
            }[key]

        elif TrainConfig.corrupt_cfg.corrupt_act:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.obser_sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.25,
                'walker2d': 0.25,
                'hopper': 0.25,
            }[key]
            TrainConfig.Agent.critic_target_update_freq = {
                'halfcheetah': 1,
                'walker2d': 1,
                'hopper': 1, 
            }[key]
            TrainConfig.Agent.learning_rate = {
                'halfcheetah': 3e-4,
                'walker2d': 3e-4,
                'hopper': 3e-4,
            }[key]

        elif TrainConfig.corrupt_cfg.corrupt_reward:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.obser_sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.25,
                'walker2d': 0.5,
                'hopper': 0.5,
            }[key]
            TrainConfig.Agent.critic_target_update_freq = {
                'halfcheetah': 1,
                'walker2d': 1,
                'hopper': 3, 
            }[key]
            TrainConfig.Agent.learning_rate = {
                'halfcheetah': 3e-4,
                'walker2d': 3e-4,
                'hopper': 3e-4,
            }[key]

        elif TrainConfig.corrupt_cfg.corrupt_dynamics:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 1.0,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.obser_sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.25,
                'walker2d': 0.25,
                'hopper': 0.25,
            }[key]
            TrainConfig.Agent.critic_target_update_freq = {
                'halfcheetah': 1,
                'walker2d': 1,
                'hopper': 3, 
            }[key]
            TrainConfig.Agent.learning_rate = {
                'halfcheetah': 3e-4,
                'walker2d': 3e-4,
                'hopper': 3e-4,
            }[key]

    elif TrainConfig.corrupt_cfg.corruption_mode == 'adversarial':
        if TrainConfig.corrupt_cfg.corrupt_obs and \
                TrainConfig.corrupt_cfg.corrupt_act and \
                TrainConfig.corrupt_cfg.corrupt_reward and \
                TrainConfig.corrupt_cfg.corrupt_dynamics:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.001,
            }[key]
            TrainConfig.Agent.obser_sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.01,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.25,
                'walker2d': 0.25,
                'hopper': 0.5, 
            }[key]
            TrainConfig.Agent.critic_target_update_freq = {
                'halfcheetah': 1,
                'walker2d': 1,
                'hopper': 1, 
            }[key]
            TrainConfig.Agent.learning_rate = {
                'halfcheetah': 3e-4,
                'walker2d': 3e-4,
                'hopper': 3e-4,
            }[key]

        elif TrainConfig.corrupt_cfg.corrupt_obs:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 1.0,
                'hopper': 0.5,
            }[key]
            TrainConfig.Agent.obser_sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.25,
                'hopper': 0.25, 
            }[key]
            TrainConfig.Agent.critic_target_update_freq = {
                'halfcheetah': 1,
                'walker2d': 1,
                'hopper': 2, 
            }[key]
            TrainConfig.Agent.learning_rate = {
                'halfcheetah': 3e-4,
                'walker2d': 3e-4,
                'hopper': 1e-3,
            }[key]

        elif TrainConfig.corrupt_cfg.corrupt_act:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 1.0,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.obser_sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.01,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.25,
            }[key]
            TrainConfig.Agent.critic_target_update_freq = {
                'halfcheetah': 1,
                'walker2d': 1,
                'hopper': 2, 
            }[key]
            TrainConfig.Agent.learning_rate = {
                'halfcheetah': 3e-4,
                'walker2d': 3e-4,
                'hopper': 1e-4,
            }[key]

        elif TrainConfig.corrupt_cfg.corrupt_reward:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.obser_sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.5,
            }[key]
            TrainConfig.Agent.critic_target_update_freq = {
                'halfcheetah': 1,
                'walker2d': 1,
                'hopper': 2, 
            }[key]
            TrainConfig.Agent.learning_rate = {
                'halfcheetah': 3e-4,
                'walker2d': 3e-4,
                'hopper': 3e-4,
            }[key]

        elif TrainConfig.corrupt_cfg.corrupt_dynamics:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.obser_sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.25,
                'hopper': 0.5,
            }[key]
            TrainConfig.Agent.critic_target_update_freq = {
                'halfcheetah': 1,
                'walker2d': 1,
                'hopper': 1, 
            }[key]
            TrainConfig.Agent.learning_rate = {
                'halfcheetah': 3e-4,
                'walker2d': 3e-4,
                'hopper': 3e-4,
            }[key]

    return TrainConfig
