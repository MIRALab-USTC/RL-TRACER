
def handle_config(TrainConfig):
    if 'obs' in TrainConfig.corrupt_cfg.corruption_type:
        TrainConfig.corrupt_cfg.corrupt_obs = True
    if 'act' in TrainConfig.corrupt_cfg.corruption_type:
        TrainConfig.corrupt_cfg.corrupt_act = True
    if 'rew' in TrainConfig.corrupt_cfg.corruption_type:
        TrainConfig.corrupt_cfg.corrupt_reward = True
    if 'dynamics' in TrainConfig.corrupt_cfg.corruption_type:
        TrainConfig.corrupt_cfg.corrupt_dynamics = True

    if not TrainConfig.default_setting:
        return TrainConfig

    if 'riql' not in TrainConfig.experiment:
        return TrainConfig

    '''RIQL Hyperparameters'''
    key = TrainConfig.task_name.split('-')[0]
    TrainConfig.num_q = 5

    if TrainConfig.corrupt_cfg.corruption_mode == 'random':
        if TrainConfig.corrupt_cfg.corrupt_obs:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.25,
                'hopper': 0.25, 
            }[key]
            if key == 'hopper':
                TrainConfig.num_q = 3

        elif TrainConfig.corrupt_cfg.corrupt_act:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.5,
                'walker2d': 0.5,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.25,
                'walker2d': 0.1,
                'hopper': 0.25,
            }[key]
            if key == 'halfcheetah':
                TrainConfig.num_q = 3

        elif TrainConfig.corrupt_cfg.corrupt_reward:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 3.0,
                'walker2d': 3.0,
                'hopper': 1.0,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.25,
                'walker2d': 0.1,
                'hopper': 0.25,
            }[key]
            if key == 'hopper':
                TrainConfig.num_q = 3

        elif TrainConfig.corrupt_cfg.corrupt_dynamics:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 3.0,
                'walker2d': 1.0,
                'hopper': 1.0,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.25,
                'walker2d': 0.25,
                'hopper': 0.5,
            }[key]
            if key == 'walker2d':
                TrainConfig.num_q = 3
    
    elif TrainConfig.corrupt_cfg.corruption_mode == 'adversarial':
        if TrainConfig.corrupt_cfg.corrupt_obs:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 0.1,
                'walker2d': 1.0,
                'hopper': 1.0,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.25,
                'hopper': 0.25, 
            }[key]
        elif TrainConfig.corrupt_cfg.corrupt_act:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 1.0,
                'walker2d': 1.0,
                'hopper': 1.0,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.25,
            }[key]
        elif TrainConfig.corrupt_cfg.corrupt_reward:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 1.0,
                'walker2d': 3.0,
                'hopper': 0.1,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.1,
                'hopper': 0.25,
            }[key]
        elif TrainConfig.corrupt_cfg.corrupt_dynamics:
            TrainConfig.Agent.sigma = {
                'halfcheetah': 1.0,
                'walker2d': 1.0,
                'hopper': 1.0,
            }[key]
            TrainConfig.Agent.quantile = {
                'halfcheetah': 0.1,
                'walker2d': 0.25,
                'hopper': 0.5,
            }[key]
    return TrainConfig
