import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE

from .utils import eval_mode
# from offlinerl.utils.env import get_env


def d4rl_score(task, rew_mean):
    score = (rew_mean - REF_MIN_SCORE[task]) / (REF_MAX_SCORE[task] - REF_MIN_SCORE[task]) * 100
    return score


def d4rl_norm_score(env, eval_returns):
    normalized_score = env.get_normalized_score(eval_returns) * 100.0
    return normalized_score


def d4rl_eval_fn(env, env_type, task_name, video, max_episode_steps, num_eval_episodes=100):

    def d4rl_eval(agent, epoch, print_log=False):
        episode_returns = []
        episode_lengths = []
        # evaluate the agent under num_eval_episodes episodes
        for episode in range(num_eval_episodes):
            s, d, ep_ret, ep_len = env.reset(), False, 0, 0
            video.init(enabled=(episode == 0))
            while not (d or (ep_len == max_episode_steps)):
                with eval_mode(agent):
                    a = agent.select_action(s, deterministic=True)
                s, r, d, _ = env.step(a)
                video.record(env)
                ep_ret += r
                ep_len += 1

            video.save('%d.mp4' % epoch)
            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            if print_log:
                print("return: ", ep_ret, " score: ", d4rl_norm_score(env, np.array(ep_ret)))

        rew_mean = np.mean(episode_returns)
        len_mean = np.mean(episode_lengths)

        # score = d4rl_score(task_name, rew_mean)
        score = d4rl_norm_score(env, np.array(episode_returns))

        eval_info = {f'TestEpRet{env_type}': rew_mean,
                     f'TestEpRetStd{env_type}': np.std(episode_returns),
                     f'TestEpLen{env_type}': len_mean,
                     f'TestScore{env_type}': np.mean(score),
                     f'TestScoreStd{env_type}': np.std(score)}
        return eval_info
    
    return d4rl_eval
