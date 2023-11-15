import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_lightning import Trainer

from src.backbones.txt_model import TimeTexture_flair
from src.datamodule import DataModule
from src.ensemble_task_module import EnsembleSegmentationTask
from src.load_data import load_data
from src.prediction_writer import PredictionWriter
from src.utils_dataset import read_config
from src.utils_prints import print_inference_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

argParser = argparse.ArgumentParser()
argParser.add_argument("--config_file", help="Path to the .yml config file")
argParser.add_argument("--out_model_name", help="")
argParser.add_argument("--out_folder", help="")

argParser.add_argument("--num_workers", type=int, help="")

argParser.add_argument("--encoder_names", default=[], nargs='+', type=str, help="")
argParser.add_argument("--checkpoint_paths", default=[], nargs='+', type=str, help="")
argParser.add_argument("--output_from", type=str, default="logits", help="logits, probabilities")
argParser.add_argument("--flip", default=[], nargs='+', type=str, help="v, h")

argParser.add_argument('--filter_clouds', default=False, action='store_true')
argParser.add_argument('--average_month', default=False, action='store_true')
argParser.add_argument("--drop_utae_modality", type=float, help="")


class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model

    def forward(self, config, bpatch, bspatch, dates, metadata):
        logits = self.model(config, bpatch, bspatch, dates, metadata)

        return logits


if __name__ == "__main__":
    args = argParser.parse_args()
    args_dict = vars(args)

    config = read_config(args.config_file)
    args_for_update = {k: v for k, v in args_dict.items() if v and v is not None}
    config.update(args_for_update)

    out_dir = Path(config["out_folder"], config["out_model_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    d_train, d_val, d_test = load_data(config)

    # Dataset definition
    data_module = DataModule(
        dict_train=d_train,
        dict_val=d_val,
        dict_test=d_test,
        config=config,
        drop_last=False,
        augmentation_set=None
    )

    # Models
    models = []
    for name, checkpoint_path in zip(config["encoder_names"], config["checkpoint_paths"]):
        config["encoder_name"] = name
        # crate model
        model = TimeTexture_flair(config)
        model = Wrapper(model)

        # load weights into the model
        checkpoint = torch.load(checkpoint_path)["state_dict"]
        model.load_state_dict(checkpoint, strict=False)

        models.append(model)

    models = nn.ModuleList(models)
    model = EnsembleSegmentationTask(models, config, output_from=config["output_from"], flip=config.get("flip", []))
    model.eval()
    model.to(device)

    # Predict
    output_dir = os.path.join(out_dir, "predictions" + "_" + config["out_model_name"] + "-" + config["output_from"] + "_" + "".join(config.get("flip", []))).strip()
    writer_callback = PredictionWriter(
        output_dir=output_dir,
        write_interval="batch",
    )

    # Predict Trainer
    trainer = Trainer(
        accelerator=config["accelerator"],
        devices=config["gpus_per_node"],
        strategy=config["strategy"],
        num_nodes=config["num_nodes"],
        callbacks=[writer_callback],
        enable_progress_bar=config["enable_progress_bar"],
    )

    # Enable time measurement
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()

    trainer.predict(model, datamodule=data_module, return_predictions=False)

    if config['strategy'] != None:
        dist.barrier()
        torch.cuda.synchronize()
    ender.record()
    torch.cuda.empty_cache()

    inference_time_seconds = starter.elapsed_time(ender) / 1000.0
    print_inference_time(inference_time_seconds, config)
