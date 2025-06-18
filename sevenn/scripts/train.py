import random

import torch
import torch.distributed as dist
import torch.nn
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

import sevenn._keys as KEY
from sevenn.model_build import build_E3_equivariant_model
from sevenn.sevenn_logger import Logger
from sevenn.train.trainer import Trainer

from .processing_continue import processing_continue
from .processing_dataset import processing_dataset
from .processing_epoch import processing_epoch


def init_loaders(train, valid, _, config):
    batch_size = config[KEY.BATCH_SIZE]
    is_ddp = config[KEY.IS_DDP]
    if is_ddp:
        dist.barrier()
        train_sampler = DistributedSampler(
            train,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=config[KEY.TRAIN_SHUFFLE],
        )
        valid_sampler = DistributedSampler(
            valid, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )
        train_loader = DataLoader(
            train, batch_size=batch_size, sampler=train_sampler
        )
        valid_loader = DataLoader(
            valid, batch_size=batch_size, sampler=valid_sampler
        )
    else:
        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=config[KEY.TRAIN_SHUFFLE]
        )
        valid_loader = DataLoader(valid, batch_size=batch_size)
    return train_loader, valid_loader, None


def radial_embeddings_std_mean(config, data_loader):
    """
    Compute mean, std of weight_nn inputs
    """
    from sevenn.model_build import init_edge_embedding
    from sevenn.nn.edge_embedding import EdgePreprocess
    Logger().writeline('Compute std, mean of radial embedding')
    edge_preprocess = EdgePreprocess(is_stress=False)
    edge_embedding = init_edge_embedding(config)
    with torch.no_grad():
        edge_embedded = []
        for batch in data_loader:
            batch = edge_preprocess(batch)
            out = edge_embedding(batch)
            edge_embedded.append(out[KEY.EDGE_EMBEDDING])
        # shapes (*, radial_basis_num)
        edge_embedded = torch.cat(edge_embedded)
        std, mean = torch.std_mean(edge_embedded)
    Logger().format_k_v('std, mean', f'{std:.4f}, {mean:.4f}', write=True)
    return std, mean


def compute_fisher_information(config, trainer, loader):
    import os
    if config[KEY.BATCH_SIZE] != 1:
        raise ValueError('Batch size must be 1 to compute Fisher Information.')
    fname_fisher = 'fisher_sevenn.pt'
    fname_opt = 'opt_params_sevenn.pt'
    if os.path.isfile(fname_fisher) or os.path.isfile(fname_opt):
        raise ValueError(f'{fname_fisher} or {fname_opt} already exist!'
                         ' abort computation to avoid overwrite')
    Logger().write('Calculating Fisher information and'
                   ' optimized parameters for EWC...\n')

    loss_thr = config[KEY.CONTINUE][KEY.LOSS_THR]
    fisher_info, optim_param, calc_num =\
        trainer.compute_fisher_matrix(loader, loss_thr)

    torch.save(fisher_info, fname_fisher)
    torch.save(optim_param, fname_opt)

    Logger().write('Calculation finished.'
                   f'{calc_num} configurations'
                   ' from trainingset were used.\n')
    Logger().write(f'Files {fname_fisher} and {fname_opt}'
                   ' are generated.\n')
    Logger().bar()


# TODO: E3_equivariant model assumed
def train(config, working_dir: str):
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
    loaders = init_loaders(*data_lists, config=config)

    if config[KEY.STANDARDIZE_RADIAL_EMBEDDING] is not False:
        std, mean = radial_embeddings_std_mean(config, data_loader=loaders[0])
        config[KEY._CONV_KWARGS].update({
            'weight_shift': mean,
            'weight_scale': 1 / std,
        })

    Logger().write('\nModel building...\n')
    model = build_E3_equivariant_model(config)
    assert isinstance(model, torch.nn.Module)

    Logger().write('Model building successful\n')

    trainer = Trainer(model, config)
    if state_dicts is not None:
        assert isinstance(state_dicts, tuple)
        trainer.load_state_dicts(*state_dicts, strict=False)

    Logger().print_model_info(model, config)
    # log_model_info(model, config)

    Logger().write('Trainer initialized, ready to training\n')
    Logger().bar()

    if config[KEY.CONTINUE][KEY.CALC_FISHER]:
        compute_fisher_information(config, trainer, loaders[0])
        return

    processing_epoch(
        trainer, config, loaders, start_epoch, init_csv, working_dir
    )
    Logger().timer_end('total', message='Total wall time')
