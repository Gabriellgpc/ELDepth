import os
from functools import partial

import click
import pandas as pd
import tensorflow as tf
from wandb.keras import WandbCallback

import wandb
from src.data.dataloader import DataGenerator
from src.helpers import setup_gpu
from src.helpers.utils import read_configuration, seed_everything
from src.losses import depth_final_loss
from src.lr_schedules import cosineAnnealingScheduler
from src.models import UNet
from src.optimizer.gc_adam import GCAdam
from src.viz.plot_images import visualize_depth_map


def get_callbacks(config):
    callbacks = []

    wandb_cb = WandbCallback(log_weights=False,
                             save_graph=False,
                             save_model=False,
                             save_weights_only=False,
                             compute_flops=True,
                            #  log_batch_frequency=5,
                             )
    callbacks.append(wandb_cb)

    # earlystopping
    earlystop = tf.keras.callbacks.EarlyStopping(patience=config.trainer.patience,
                                                 monitor='val_loss',
                                                 restore_best_weights=True,
                                                 verbose=1,
                                                 )
    callbacks.append(earlystop)

    # lr schedule
    schedule = partial(cosineAnnealingScheduler,
                       initial_lr=config.trainer.initial_lr,
                       reset_step=5,
                       min_lr=config.trainer.min_lr)
    schedule_cb = tf.keras.callbacks.LearningRateScheduler( schedule )
    callbacks.append(schedule_cb)

    return callbacks

def get_datagenerator(config):
    train_path = config.dataloader.train
    validation_path = config.dataloader.validation

    filelist = []
    for root, dirs, files in os.walk(train_path):
        for file in files:
            filelist.append(os.path.join(root, file))
    filelist.sort()
    data = {
        "image": [x for x in filelist if x.endswith(".png")],
        "depth": [x for x in filelist if x.endswith("_depth.npy")],
        "mask":  [x for x in filelist if x.endswith("_depth_mask.npy")],
    }
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=config.random_seed)

    train_loader = DataGenerator(
        data=df[:16].reset_index(drop="true"),
        batch_size=config.dataloader.batch_size,
        dim=config.dataloader.dim)

    filelist = []
    for root, dirs, files in os.walk(validation_path):
        for file in files:
            filelist.append(os.path.join(root, file))

    filelist.sort()
    data = {
        "image": [x for x in filelist if x.endswith(".png")],
        "depth": [x for x in filelist if x.endswith("_depth.npy")],
        "mask":  [x for x in filelist if x.endswith("_depth_mask.npy")],
    }
    df = pd.DataFrame(data)
    validation_loader = DataGenerator(
        data=df[:16].reset_index(drop="true"),
        batch_size=config.dataloader.batch_size,
        dim=config.dataloader.dim)

    return train_loader, validation_loader

@click.command()
@click.option('--config-file', default='/workspace/options/exp_01.yaml')
def main(config_file):
    config = read_configuration(config_file)

    seed_everything(config.random_seed)

    wandb.init(entity='condados', project='ELDepth')

    setup_gpu(config.gpu)

    # ------------------
    #    Dataloaders
    # ------------------

    train_loader, validation_loader = get_datagenerator(config)

    # ------------------
    #     Callbcaks
    # ------------------
    callbacks = get_callbacks(config)

    # ------------------
    #     Optimizer
    # ------------------

    # optimizer = tf.keras.optimizers.Adam(learning_rate=config.trainer.initial_lr)
    optimizer = GCAdam(learning_rate=config.trainer.initial_lr)

    # ------------------
    #      Model
    # ------------------

    model = UNet()

    # Compile the model
    model.compile(optimizer, loss=depth_final_loss)

    x_dummy = tf.random.normal([1,*config.dataloader.dim, 3], 0.5, 0.5)
    model(x_dummy)
    print(model.summary())

    # ------------------
    #     Training
    # ------------------

    model.fit(
        train_loader,
        epochs=config.trainer.epochs,
        validation_data=validation_loader,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------
    #     W&B Logs
    # ------------------

    _, validation_loader = get_datagenerator(config)

    figsaved = '/workspace/tmp/visualize_depth_map.png'
    fig, _ = visualize_depth_map( next(iter(validation_loader)), test=True, model=model, save_at=figsaved)
    # wandb.log_artifact()
    wandb.log({'depth-map':fig})

    #snipper to cleanup cache: wandb artifact cache cleanup 1GB

if __name__=='__main__':
    main()