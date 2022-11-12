import logging

import tensorflow as tf


def setup_gpu(gpu_id=0, allow_memory_growth=True):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            logging.info(f'Trying to configure GPU {gpu_id}: allow_memory_growth={allow_memory_growth}')
            gpu = gpus[gpu_id] if gpu_id <= len(gpus) else gpus[0]
            if len(gpus) > 1:
                tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            if allow_memory_growth:
                tf.config.experimental.set_memory_growth(gpu, allow_memory_growth)

        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            logging.error('GPU configuration failed: GPU_ID={gpu_id}, allow_memory_growth={allow_memory_growth}')
            logging.error(e)