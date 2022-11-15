import os
from functools import partial

import click
import pandas as pd
import tensorflow as tf
from wandb.keras import WandbCallback

import wandb
from src.data.augmentation import *
# from src.data.dataloader import DataGenerator
from src.data.dataloader import build_tf_dataloader, get_tf_resize
from src.helpers import setup_gpu
from src.helpers.utils import read_configuration, seed_everything
from src.losses import *
from src.lr_schedules import cosineAnnealingScheduler
from src.models import XLSR, SqueezeUNet, UNet
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

    # read CSV
    train_df = pd.read_csv(config.dataloader.train_csv)
    validation_df = pd.read_csv(config.dataloader.train_csv)

    # train loader
    train_transforms = [get_tf_resize(config.dataloader.dim)]
    train_loader = build_tf_dataloader(
                        input_paths=train_df['image'].values,
                        depth_paths=train_df['depth'].values,
                        mask_paths=train_df['mask'].values,
                        batch_size=config.dataloader.batch_size,
                        transforms=train_transforms,
                        train=True
                        )

    # validation loader
    validation_transforms = [get_tf_resize(config.dataloader.dim)]
    validation_loader = build_tf_dataloader(
                        input_paths=validation_df['image'].values,
                        depth_paths=validation_df['depth'].values,
                        mask_paths=validation_df['mask'].values,
                        batch_size=config.dataloader.batch_size,
                        transforms=validation_transforms,
                        train=True #for debug or in case wish to use the validations steps
                        )

    return train_loader, validation_loader

def get_model(config):
    model_name = config.network['type'].lower()
    model = UNet()

    if model_name == 'squeezenet':
        hparams = config.network['SqueezeUNet']
        x = tf.keras.layers.Input(shape=(config.dataloader.dim[0], config.dataloader.dim[1], 3))
        out = SqueezeUNet(x, **hparams)
        model = tf.keras.models.Model(inputs=x, outputs=out, name='Squeeze-UNet')
    if model_name == 'xlsr':
        hparams = config.network['XLSR']
        model = XLSR(**hparams, name='XLSR')

    return model

def get_loss(config):
    loss_name = config.trainer.loss.lower()
    loss = depth_final_loss

    # custom_loss, mae, mse, charbonnier_loss, loss_SILlog, loss_iRMSE, loss_RMSE

    if loss_name == 'charbonnier_loss':
        loss = charbonnier_loss
    if loss_name == 'mae':
        loss = tf.keras.losses.mae
    if loss_name == 'mse':
        loss = tf.keras.losses.mse
    if loss_name == 'loss_silog':
        loss = loss_SILog
    if loss_name == 'loss_rmse':
        loss = loss_RMSE
    if loss_name == 'loss_irmse':
        loss = loss_iRMSE

    return loss

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

    model = get_model(config)

    # Compile the model
    loss = get_loss(config)
    model.compile(optimizer, loss=loss, metrics=[loss_SILog, loss_RMSE], weighted_metrics=[])

    x_dummy = tf.random.normal([1,*config.dataloader.dim, 3], 0.5, 0.5)
    model(x_dummy)
    print(model.summary())

    # ------------------
    #     Training
    # ------------------

    model.fit(
        train_loader,
        validation_data=validation_loader,
        epochs=config.trainer.epochs,
        steps_per_epoch=config.trainer.steps_per_epoch,
        validation_steps=config.trainer.validation_steps,
        validation_freq=config.trainer.validation_freq,
        callbacks=callbacks,
        max_queue_size=5,
        use_multiprocessing=True,
        workers=os.cpu_count(),
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