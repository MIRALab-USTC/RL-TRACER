import time
import torch
import random
import numpy as np
from pathlib import Path
import hydra
import yaml
from tqdm import tqdm
# import os
# os.environ['MUJOCO_GL'] = 'egl'

import gym
import d4rl
import dmc2gym

import wandb
from model.fake_env import FakeEnv
from common import utils
from common import attack
from common import riql_config
from common import tracer_config
from common.logger_tb import Logger
from common.logx import EpochLogger
from common.video import VideoRecorder
from common.evaluate import d4rl_eval_fn

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


TASK = {
    "halfcheetah": "HalfCheetah-v2",
    "hopper": "Hopper-v2",
    "walker2d": "Walker2d-v2"
}


class Workspace:

    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = Path.cwd()
        self.current_path = Path(__file__).resolve().parent
        self.device = torch.device('cuda:%d' % cfg.cuda_id if 'cuda' in cfg.device else 'cpu')
        self.model_dir = self.work_dir / 'model'
        if self.cfg.save_agent:
            self.model_dir.mkdir(exist_ok=True)
        self._init_eval_env()
        self._init_log()
        self._init_offline_dataset(self.cfg.Buffer)
        self._init_eval_fn()
        self._init_agent(self.cfg.Agent)

    def _init_eval_env(self):
        # initialize eval environment
        self.eval_env = gym.make(self.cfg.task_name)
        self.eval_env.seed(self.cfg.seed)
        self.env_type = ''

        # set parameters
        self.cfg.state_dim = self.eval_env.observation_space.shape[0]
        self.cfg.action_dim = int(np.prod(self.eval_env.action_space.shape))
        self.cfg.action_limit = float(self.eval_env.action_space.high[0])
        try: self.max_episode_steps = self.eval_env._max_episode_steps
        except: self.max_episode_steps = self.cfg.max_episode_steps
        # initialize video saver
        video_dir = self.work_dir / 'video'
        self.video = VideoRecorder(
            str(video_dir) if self.cfg.save_video else None, height=448, width=448)

    def _init_log(self):
        self.cfg = riql_config.handle_config(self.cfg)
        self.cfg = tracer_config.handle_config(self.cfg)

        work_dir = str(self.work_dir)
        exp_name = work_dir.split('/')[-2]
        logger_kwargs = dict(output_dir=work_dir, exp_name=exp_name)
        logsp = EpochLogger(**logger_kwargs)
        with open(self.work_dir / '.hydra' / 'config.yaml') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        logsp.save_config(config)
        logtb = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.logger = dict(tb=logtb, sp=logsp)
        if self.cfg.use_wandb:
            utils.wandb_init(utils.asdict(self.cfg))

    def _init_offline_dataset(self, buf_cfg):
        if self.cfg.load_agent and not self.cfg.train_agent:
            state_mean = np.load(str(self.work_dir / "state_mean.npy"))
            state_std = np.load(str(self.work_dir / "state_std.npy"))
        else:
            # load offline dataset
            dataset = d4rl.qlearning_dataset(self.eval_env)
            print("-------------------------------------------------")
            print("------------ Infor of Offline Dataset -----------")
            print("Dataset Length: %d" % (len(dataset)))
            print("Dataset Keys: ", dataset.keys())
            print("Dataset Obs Shape: ", dataset["observations"].shape)
            print("Dataset Act Shape: ", dataset["actions"].shape)
            print("Dataset Rew Shape: ", dataset["rewards"].shape)
            print("Dataset Next Obs Shape: ", dataset["next_observations"].shape)
            print("---------------------- END ----------------------")
            print("-------------------------------------------------")

            # corrupt offline dataset
            attack_indexes = None
            if self.cfg.use_corruption and (self.cfg.corrupt_cfg.corruption_type != 'none'):
                attack_indexes = np.zeros(dataset["rewards"].shape)
                dataset, indexes = attack.attack_dataset(self.cfg.corrupt_cfg, dataset)
                attack_indexes[indexes] = 1.0

            # normalize offline data
            if self.cfg.norm_data.norm_state:
                state_mean, state_std = utils.compute_mean_std(
                        np.concatenate([dataset["observations"],
                                        dataset['next_observations']], axis=0), eps=1e-3)
            else:
                state_mean, state_std = 0, 1

            print('state mean: ', state_mean)
            print('state std: ', state_std)
            np.save(str(self.work_dir / "state_mean.npy"), state_mean)
            np.save(str(self.work_dir / "state_std.npy"), state_std)

            dataset["observations"] = utils.normalize(
                    dataset["observations"], state_mean, state_std)
            dataset["next_observations"] = utils.normalize(
                    dataset["next_observations"], state_mean, state_std)

            # initialize the real buffer
            buf_cfg.capacity = len(dataset["observations"])
            buffer = hydra.utils.instantiate(buf_cfg)
            buffer.add_attack_indexes(attack_indexes)
            # initialize the fake environment
            self.fake_env = FakeEnv(dataset=dataset,
                                    buffer=buffer,
                                    device=self.device,
                                    config=None)

        # normalize eval env
        self.eval_env = utils.wrap_env(self.eval_env,
                                       state_mean=state_mean,
                                       state_std=state_std)

    def _init_eval_fn(self):
        self.eval_fn = d4rl_eval_fn(self.eval_env,
                                    env_type='',
                                    task_name=self.cfg.task_name,
                                    video=self.video,
                                    max_episode_steps=self.max_episode_steps,
                                    num_eval_episodes=self.cfg.num_eval_episodes)

    def _init_agent(self, agent_cfg):
        if self.cfg.train_agent or self.cfg.load_agent:
            self.agent = hydra.utils.instantiate(agent_cfg)
            return True
        self.agent = None
        return False

    def train(self):
        self._start_time = start_time = time.time()
        print("-------------------------------------------------------------------------------")
        print("| Policy: {} | Env: {} | Seed: {}".format(self.cfg.experiment, self.cfg.task_name, self.cfg.seed))
        print("-------------------------------------------------------------------------------")
        self.load()

        if self.cfg.train_agent:
            self.train_agent()

        write_dir = self.current_path / "main_results/"
        write_dir.mkdir(parents=True, exist_ok=True)
        self.log_in_the_end(str(write_dir / self.cfg.task_name), max_episode=10, print_log=True)

        self.eval_env.close()

    def train_agent(self):
        self._start_time = start_time = time.time()
        info_dict = dict()
        print("| Training Agent...")
        for epoch in range(1, self.cfg.max_epochs+1):

            # train agent by both the real and fake data from fake environment
            for step in range(1, self.cfg.num_trains_per_train_loop+1):
                self.agent.total_time_steps += 1

                # update the agent
                for i in range(self.cfg.num_train_loops_per_epoch):
                    save_log = (step+i) % self.cfg.log_agent_freq == 0
                    loss_info = self.agent.update(fake_env=self.fake_env, logger=self.logger,
                                                  save_log=save_log)
                    if save_log and self.cfg.use_wandb:
                        wandb.log({"epoch": epoch, **loss_info})
                    info_dict.update(loss_info)
                # print log
                if step % 1000 == 0:
                    pstep = '%d' % step if step // 1000 == 1 else '%d ' % step
                    print("| Train Epoch: %d | Step: %s | LossQ: %.2f | Qvals: %.2f | TQvals: %.2f | LossPi: %.2f | HPi: %.2f | Time: %s" % (
                        epoch, pstep, info_dict['LossQ'], info_dict['Qvals'], info_dict['TQvals'], info_dict['LossPi'], info_dict['HPi'], utils.calc_time(start_time)))

            # evaluate the agent
            if self.cfg.eval and (epoch == 1 or epoch % self.cfg.eval_freq == 0):
                eval_info = self.eval_fn(self.agent, epoch)
                self.logger['sp'].store(**eval_info)

                # print and write log
                epoch_fps = epoch * self.cfg.eval_freq / (time.time() - start_time)
                self.agent.print_log(self.logger['sp'], epoch, self.env_type, start_time, epoch_fps)

                with self.logger['tb'].log_and_dump_ctx(self.agent.total_time_steps, ty='eval') as log:
                    log('epoch', epoch)
                    log('step', self.agent.total_time_steps)
                    log('total_time', time.time() - start_time)
                    for k, v in eval_info.items():
                        log(k, v)

                if self.cfg.use_wandb:
                    eval_save_info = dict()
                    for k, v in eval_info.items():
                        eval_save_info['eval/' + k] = v
                    wandb.log({"epoch": epoch,
                               "step": self.agent.total_time_steps,
                               "total_time": time.time() - start_time,
                               **eval_save_info})

            # save the agent
            if epoch % self.cfg.save_agent_freq == 0 and self.cfg.save_agent:
                self.agent.save(self.model_dir, epoch)


    def log_in_the_end(self, write_dir, max_episode=100, print_log=False):
        # evaluate the agent with an episode number: max_episode
        eval_info = self.eval_fn(self.agent, self.cfg.max_epochs + 1, print_log=print_log)
        eval_info["step"] = self.agent.total_time_steps
        eval_info["epoch"] = self.cfg.max_epochs
        eval_info["total_time"] = time.time() - self._start_time
        utils.save_log_in_csv(eval_info, write_dir, max_episode, self.cfg)

    def load(self):
        if self.cfg.load_agent:
            self.agent.load(self.cfg.load_dir, self.cfg.load_agent_step)


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    # torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(torch.get_num_threads())
    utils.set_seed_everywhere(cfg.seed)
    workspace = Workspace(cfg)
    # import pdb
    # pdb.set_trace()

    workspace.train()
    wandb.finish()


if __name__ == '__main__':
    main()
