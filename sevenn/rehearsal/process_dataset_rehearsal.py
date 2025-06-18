import sevenn._keys as KEY
from sevenn.scripts.processing_dataset import dataset_load
from sevenn.train.dataset import AtomGraphDataset


def process_dataset_rehearsal(config, logger):
    cutoff = config[KEY.CUTOFF]

    logger().write('\nTry to use rehearsal from load_memory_path\n')
    memory_set = AtomGraphDataset({}, cutoff)
    for file in config[KEY.LOAD_MEMORY_PATH]:
        # print(file)
        memory_set.augment(dataset_load(file, config))
    memory_set.group_by_key()
    memory_set.unify_dtypes()

    memory_set.toggle_requires_grad_of_data(KEY.EDGE_VEC, True)

    logger().format_k_v(
        '\nLoaded memory set  size is', memory_set.len(), write=True
    )

    logger().write('Memory set loading was successful\n')

    if memory_set.x_is_one_hot_idx is False:
        memory_set.x_to_one_hot_idx(config[KEY.TYPE_MAP])

    return memory_set
