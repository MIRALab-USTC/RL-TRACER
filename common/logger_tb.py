# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import datetime
from collections import defaultdict

import numpy as np
import torch
import torchvision
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter


FORMAT_CONFIG = {
    'rl': {
        'train': [('step', 'S', 'int'),
                  ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                  ('episode_reward', 'R', 'float'),
                  ('buffer_size', 'BS', 'int'), ('fps', 'FPS', 'float'),
                  ('total_time', 'T', 'time')],
        'eval': [('step', 'S', 'int'), ('Epoch', 'E', 'int'),
                 ('TestEpLen', 'L', 'int'),
                 ('TestEpRet', 'R', 'float'),
                 ('TestScore', 'Score', 'float'),
                 ('total_time', 'T', 'time')]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['step']) >= data['step']:
                    break
                rows.append(row)
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open('a')
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        if 'frame' in data:
            data['frame'] = step
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb=True, config='rl', comment=''):
        self._log_dir = log_dir
        log_dir /= 'logs'
        log_dir.mkdir(exist_ok=True)
        if use_tb:
            if comment != '':
                log_dir /= ('%s' % comment)
            self._sw = SummaryWriter(str(log_dir))
        else:
            self._sw = None
        self._train_mg = MetersGroup(self._log_dir / 'train.csv',
                                     formating=FORMAT_CONFIG[config]['train'])
        self._eval_mg = MetersGroup(self._log_dir / 'eval.csv',
                                    formating=FORMAT_CONFIG[config]['eval'])

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_image(self, key, image, step):
        if self._sw is not None:
            assert image.dim() == 3, print(image.dim())
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._sw.add_image(key, grid, step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def save_log_dict(self, info_name, info_dict, step):
        for key, value in info_dict.items():
            if not isinstance(value, float) and not isinstance(value, int):
                assert value.ndim <= 2, print(info_name, key, value.ndim)
                if 'param' in key:
                    self.log_histogram(f'{info_name}/{key}', value, step)
                value = value.mean().item()
            self.log(f'{info_name}/{key}', value, step)

    def log(self, key, value, step, n=1):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step):
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias') and param.bias is not None:
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_histogram(self, key, histogram, step):
        assert key.startswith('train') or key.startswith('eval')
        if histogram.ndim == 2:
            histogram = histogram.mean(-1)
        self._try_sw_log_histogram(key, histogram, step)

    def save_log_imgs(self, info_name, info_dict, step):
        for key, value in info_dict.items():
            if not isinstance(value, float):
                if value.ndim == 4:
                    value = value.mean(0)
                assert value.ndim == 3, print(info_name, key, value.ndim)
            self.log_image(f'{info_name}/{key}', value, step)

    def log_image(self, key, image, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_image(key, image, step)

    def log_images(self, key, image, step):
        if self._sw is not None:
            if image.dim() == 3 or image.ndim() == 3:
                image = image.view(-1, 3, *image.size()[1:])
            if (image > 1.0).any():
                image /= 255.0
            if (image < 0.0).any():
                image += 0.5
            self._sw.add_images(key, image, step)

    def log_video(self, key, frames, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step)

    def dump(self, step, ty=None):
        if ty is None or ty == 'eval':
            self._eval_mg.dump(step, 'eval')
        if ty is None or ty == 'train':
            self._train_mg.dump(step, 'train')

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f'{self._ty}/{key}', value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)
