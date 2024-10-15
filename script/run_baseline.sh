task_name=(walker2d-medium-expert-v2 walker2d-medium-expert-v2 walker2d-medium-expert-v2 walker2d-medium-expert-v2)
cuda_id=(0 1 2 3)
seed=(1 2 3 4)
batch_size=256
max_epochs=3000 # * 1000 step
algo=(iql riql)
use_corruption=true
corruption_type=obs
corruption_rate=0.3
corruption_range=1.0
norm_state=(false true)

for ((j=0;j<${#algo[@]};j++))
    do
    algo_name=${algo[j]}-corrupt_${corruption_type}_${corruption_range}_${corruption_rate}

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
            corrupt_cfg.corruption_mode=random \
            corrupt_cfg.corruption_rate=${corruption_rate} \
            corrupt_cfg.corruption_range=${corruption_range} \
            norm_data.norm_state=${norm_state[j]} \
            train_agent=true \
            eval_freq=10 \
            actor_lr=3e-4 \
            alpha_lr=3e-4 \
            critic_lr=3e-4 \
            agent_hidden_dim=256 \
            use_tb=true \
            use_wandb=true \
            cuda_id=${cuda_id[i]} \
            algo_name=${algo_name} \
            project=Robust_Offline_RL \
            seed=${seed[i]} > ${algo_name}-${task_name[i]}-${corruption_type}-s${seed[i]}.log 2>&1 &
    done
done
