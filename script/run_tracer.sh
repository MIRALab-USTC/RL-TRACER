# loop j
algo=(tracer)
obser_model_type=(gaussian) # (gaussian, mean)

# loop i
task_name=(hopper-medium-replay-v2)
cuda_id=(6)
seed=(1)
batch_size=16
max_epochs=3000 # * 1000 step
use_wandb=false
project=Robust_Offline_RL_Corruption # Robust_Offline_RL_Mixed_Corruption, Robust_Offline_RL
use_corruption=true
corruption_type=obs_act_rew_dynamics # obs_act_rew_dynamics, obs
corruption_mode=adversarial # random, adversarial
corruption_rate=0.3
corruption_range=1.0
norm_state=true
learning_rate=0.0003
# whether to use default hyperparams
default_setting=false
sigma=0.1
obser_sigma=0.1
quantile=0.25
num_q=5
enable_entropy=true
inv_beta=(0.0001 0.01)

for ((j=0;j<${#algo[@]};j++))
    do
    algo_name=${algo[j]}-Q${num_q}-sigma${sigma}_obs${obser_sigma}-quan${quantile}-LR${learning_rate}

    for ((i=0;i<${#cuda_id[@]};i++))
    do
        nohup python -u train.py \
            task_name=${task_name[i]} \
            task=offline \
            algo=${algo[j]} \
            max_epochs=${max_epochs} \
            batch_size=${batch_size} \
            use_corruption=${use_corruption} \
            corrupt_cfg.corruption_type=${corruption_type} \
            corrupt_cfg.corruption_mode=${corruption_mode} \
            corrupt_cfg.corruption_rate=${corruption_rate} \
            corrupt_cfg.corruption_range=${corruption_range} \
            norm_data.norm_state=${norm_state} \
            train_agent=true \
            default_setting=${default_setting} \
            num_q=${num_q} \
            eval_freq=50 \
            actor_lr=${learning_rate} \
            alpha_lr=${learning_rate} \
            critic_lr=${learning_rate} \
            agent_hidden_dim=256 \
            Agent.sigma=${sigma} \
            Agent.quantile=${quantile} \
            Agent.model_lr=${learning_rate} \
            Agent.obser_sigma=${obser_sigma} \
            Agent.obser_beta=[${inv_beta[0]},${inv_beta[1]}] \
            Agent.obser_model_type=${obser_model_type[j]} \
            Agent.enable_entropy=${enable_entropy} \
            use_tb=false \
            use_wandb=${use_wandb} \
            cuda_id=${cuda_id[i]} \
            algo_name=${algo_name} \
            project=${project} \
            seed=${seed[i]} > ./log/${algo_name}-${task_name[i]}-${corruption_mode}_${corruption_type}-s${seed[i]}.log 2>&1 &
    done
done
