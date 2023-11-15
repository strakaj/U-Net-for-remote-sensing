import argparse
import os
from pathlib import Path
import numpy as np
import albumentations as A

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities.distributed import rank_zero_only

from src.backbones.txt_model import TimeTexture_flair
from src.datamodule import DataModule
from src.task_module import SegmentationTask
from src.utils_prints import print_config
from src.utils_dataset import read_config
from src.load_data import load_data

import segmentation_models_pytorch as smp
from src.loss_fcn import  FocalLossWithSmoothing

argParser = argparse.ArgumentParser()
argParser.add_argument("--config_file", help="Path to the .yml config file")
argParser.add_argument("--name", help="Wandb project name")
argParser.add_argument("--project", help="Wandb project name")
argParser.add_argument("--seed", type=int, default=2022, help="")

argParser.add_argument("--out_model_name", help="")
argParser.add_argument("--out_folder", help="")

argParser.add_argument("--num_workers", type=int, help="")
argParser.add_argument("--num_epochs", type=int, help="")
argParser.add_argument("--batch_size", type=int, help="")
argParser.add_argument("--lr", type=float, help="")
argParser.add_argument("--optimizer", type=str, help="")
argParser.add_argument("--encoder_name", type=str, help="")
argParser.add_argument("--scheduler", type=str, help="")
argParser.add_argument("--loss_name", type=str, help="")

argParser.add_argument("--gpus_per_node", type=int, help="")
argParser.add_argument("--strategy", help="null, ddp")

argParser.add_argument('--filter_clouds', default=False, action='store_true')
argParser.add_argument('--average_month', default=False, action='store_true')
argParser.add_argument('--use_augmentation', default=False, action='store_true')
argParser.add_argument("--drop_utae_modality", type=float, help="")
argParser.add_argument("--fold", default=-1, type=int, help="")
argParser.add_argument("--val_percent", default=0.8, type=float, help="")

argParser.add_argument("--limit_train_batches", default=1.0, type=float, help="")
argParser.add_argument("--limit_val_batches", default=1.0, type=float, help="")

argParser.add_argument("--unet_checkpoint", type=str, help="")


def main(config):
    seed_everything(config["seed"], workers=True)

    out_dir = Path(config["out_folder"], config["out_model_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    d_train, d_val, d_test = load_data(config)

    # Augmentation
    if config["use_augmentation"] == True:
        transform_set = A.ReplayCompose([A.VerticalFlip(p=0.5),
                                   A.HorizontalFlip(p=0.5),
                                   A.RandomRotate90(p=0.5)])
    else:
        transform_set = None

    config["transform_set"] = transform_set
    # Dataset definition
    data_module = DataModule(
        dict_train=d_train,
        dict_val=d_val,
        dict_test=d_test,
        config=config,
        drop_last=True,
        augmentation_set = transform_set 
    )


    model = TimeTexture_flair(config)

    #@rank_zero_only
    #def track_model():
    #    print(model)
    #track_model()

    # Optimizer
    if "optimizer" in config and config["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])

    # Scheduler
    scheduler = None
    if "scheduler" in config:
        if config["scheduler"] == "exp":
            scheduler_param = {"gamma": 0.9}
            scheduler = ExponentialLR(optimizer, **scheduler_param)
        elif config["scheduler"] == "cos_0":
            scheduler_param = {"max_lr": config["lr"], "epochs": config["num_epochs"], "steps_per_epoch": 1, "pct_start": 0.0}
            scheduler = OneCycleLR(optimizer, **scheduler_param)
        elif config["scheduler"] == "cos_1":
            scheduler_param = {"max_lr": config["lr"], "epochs": config["num_epochs"], "steps_per_epoch": 1, "pct_start": 0.2}
            scheduler = OneCycleLR(optimizer, **scheduler_param)
        elif config["scheduler"] == "multi_0":
            scheduler_param = {"milestones": [np.floor(config["num_epochs"] * 0.5), np.floor(config["num_epochs"] * 0.9)], "gamma": 0.1}
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)
        elif config["scheduler"] == "multi_1":
            scheduler_param = {"milestones": [np.floor(config["num_epochs"] * 0.5), np.floor(config["num_epochs"] * 0.9)], "gamma": 0.33}
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)
        else:
            scheduler = None
    else:
        scheduler = None

    # Loss
    with torch.no_grad():
        weights_aer = torch.FloatTensor(np.array(list(config['weights_aerial_satellite'].values()))[:,0])
        weights_sat = torch.FloatTensor(np.array(list(config['weights_aerial_satellite'].values()))[:,1])
    
    if "loss_name" in config:
        if config["loss_name"] == "focal":
            criterion_vhr = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, gamma=2, alpha=1)
            criterion_hr = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, gamma=2, alpha=1)
        elif config["loss_name"] == "focal2":
            criterion_vhr = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, gamma=2, alpha=0.25)
            criterion_hr = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, gamma=2, alpha=0.25)
        elif config["loss_name"] == "dice":
            criterion_vhr = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
            criterion_hr = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        elif config["loss_name"] == "ce_ls2":
            criterion_vhr = nn.CrossEntropyLoss(weight=weights_aer, label_smoothing=0.2)
            criterion_hr = nn.CrossEntropyLoss(weight=weights_sat, label_smoothing=0.2)
        elif config["loss_name"] == "focalS2":
            criterion_vhr = FocalLossWithSmoothing(13, gamma=2, alpha=1, lb_smooth=0.2)
            criterion_hr = FocalLossWithSmoothing(13, gamma=2, alpha=1, lb_smooth=0.2)
        else:
            criterion_vhr = nn.CrossEntropyLoss(weight=weights_aer)
            criterion_hr = nn.CrossEntropyLoss(weight=weights_sat)
    else:
        criterion_vhr = nn.CrossEntropyLoss(weight=weights_aer)
        criterion_hr = nn.CrossEntropyLoss(weight=weights_sat)

    
    seg_module = SegmentationTask(
        model=model,
        num_classes=config["num_classes"],
        criterion=nn.ModuleList([criterion_vhr, criterion_hr]),
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )

    if "unet_checkpoint" in config and config["unet_checkpoint"]:
        seg_module.model.load_unet(config["unet_checkpoint"])

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(out_dir,"checkpoints"),
        filename="ckpt-{epoch:02d}-{val_loss:.2f}"+'_'+config["out_model_name"],
        save_top_k=1,
        save_last=True,
        mode="min",
        save_weights_only=True, # can be changed accordingly
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=30, # if no improvement after 30 epoch, stop learning. 
        mode="min",
    )

    prog_rate = TQDMProgressBar(refresh_rate=config["progress_rate"])

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [
        ckpt_callback, 
        early_stop_callback,
        prog_rate,
        lr_monitor
    ]

    #Logger
    kwargs = {}
    if config.get("project", ""):
        if "name" in config:
            kwargs["name"] = config["name"]
        logger = WandbLogger(
            project=config["project"],
            **kwargs
        )
    else:
        logger = TensorBoardLogger(
            save_dir=out_dir,
            name=Path("tensorboard_logs"+'_'+config["out_model_name"]).as_posix()
        )

    loggers = [
        logger
    ]

    # Train 
    trainer = Trainer(
        accelerator=config["accelerator"],
        devices=config["gpus_per_node"],
        strategy=config["strategy"],
        num_nodes=config["num_nodes"],
        max_epochs=config["num_epochs"],
        num_sanity_val_steps=0,
        limit_train_batches=config["limit_train_batches"],
        limit_val_batches=config["limit_val_batches"],
        callbacks = callbacks,
        logger=loggers,
        enable_progress_bar = config["enable_progress_bar"],
    )

    trainer.fit(seg_module, datamodule=data_module)
    trainer.validate(seg_module, datamodule=data_module) 

    @rank_zero_only
    def print_finish():
        print('--  [FINISHED.]  --', f'output dir : {out_dir}', sep='\n') 
    print_finish()
 

if __name__ == "__main__":
    args = argParser.parse_args()
    args_dict = vars(args)

    config = read_config(args.config_file)
    args_for_update = {k:v for k,v in args_dict.items() if v is not None}
    config.update(args_for_update)

    assert config["num_classes"] == config["out_conv"][-1]

    print_config(config)
    main(config)
