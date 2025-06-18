import random

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

import sevenn._keys as KEY
from sevenn.model_build import build_E3_equivariant_model
from sevenn.scripts.processing_continue import processing_continue
from sevenn.scripts.processing_dataset import processing_dataset
from sevenn.scripts.train import init_loaders
from sevenn.sevenn_logger import Logger
from sevenn.train.trainer import RehearsalTrainer

from .process_dataset_rehearsal import process_dataset_rehearsal
from .processing_epoch_with_rehearsal import processing_epoch_with_rehearsal


def init_mem_loaders(memory, config):
    mem_batch_size = config[KEY.MEM_BATCH_SIZE]
    is_ddp = config[KEY.IS_DDP]

    if is_ddp:
        dist.barrier()
        mem_sampler = DistributedSampler(
            memory,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )
        memory_loader = DataLoader(
            memory, batch_size=mem_batch_size, sampler=mem_sampler
        )
    else:
        memory_loader = DataLoader(
            memory, batch_size=mem_batch_size, shuffle=True
        )
    return memory_loader


def train_rehearsal(config, working_dir: str):
    """
    Main program flow
    """
    Logger().timer_start('total')
    seed = config[KEY.RANDOM_SEED]
    random.seed(seed)
    torch.manual_seed(seed)

    # config updated
    if config[KEY.CONTINUE][KEY.CHECKPOINT] is not False:
        state_dicts, start_epoch, init_csv = processing_continue(config)
    else:
        state_dicts, start_epoch, init_csv = None, 1, True

    # config updated
    # Note that continue and dataset cannot be separated completely
    data_lists = processing_dataset(config, working_dir)
    loaders = init_loaders(*data_lists, config)

    # This part is for initializing memory batch of dataset
    assert config[KEY.LOAD_MEMORY_PATH] is not False, 'No memory dataset given'

    memory_set = process_dataset_rehearsal(config, Logger)
    Logger().write(f'Memory ratio: {config[KEY.MEM_RATIO]}\n')
    Logger().write(f'Memory batch size: {config[KEY.MEM_BATCH_SIZE]}\n')
    mem_dat_list = memory_set.to_list()
    random.shuffle(mem_dat_list)
    mem_dat_list = mem_dat_list[:int(len(mem_dat_list) * config[KEY.MEM_RATIO])]
    mem_loader = init_mem_loaders(mem_dat_list, config)

    Logger().write('\nModel building...\n')
    model = build_E3_equivariant_model(config)
    Logger().write('Model building was successful\n')

    trainer = RehearsalTrainer(model, config)

    if state_dicts is not None:
        trainer.load_state_dicts(*state_dicts, strict=False)

    Logger().print_model_info(model, config)

    Logger().write('Trainer initialized, ready to training\n')
    Logger().bar()

    processing_epoch_with_rehearsal(
        trainer, config, loaders, mem_loader, start_epoch, init_csv, working_dir,
    )
    Logger().timer_end('total', message='Total wall time')
