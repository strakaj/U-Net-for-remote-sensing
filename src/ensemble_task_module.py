import pytorch_lightning as pl
import torch
from torchvision.transforms.functional import hflip, vflip


class EnsembleSegmentationTask(pl.LightningModule):
    def __init__(
            self,
            models,
            config,
            output_from="logits",
            flip = []

    ):
        super().__init__()
        self.models = models
        self.config = config
        self.output_from=output_from
        self.flip = flip

    # modified
    def forward(self, config, input_patch, input_spatch, input_dates, input_mtd):
        logits = self.model(config, input_patch, input_spatch, input_dates, input_mtd)
        return logits

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = []
        for model in self.models:
            _, logits = model(self.config, batch["patch"], batch['spatch'], batch['dates'], batch["mtd"])
            if self.output_from == "probabilities":
                logits = torch.softmax(logits, dim=1)
            output.append(logits)

            if "v" in self.flip:
                _, logits = model(self.config, vflip(batch["patch"]), vflip(batch['spatch']), batch['dates'], batch["mtd"])
                if self.output_from == "probabilities":
                    logits = torch.softmax(logits, dim=1)
                output.append(vflip(logits))

            if "h" in self.flip:
                _, logits = model(self.config, hflip(batch["patch"]), hflip(batch['spatch']), batch['dates'], batch["mtd"])
                if self.output_from == "probabilities":
                    logits = torch.softmax(logits, dim=1)
                output.append(hflip(logits))

        output = torch.stack(output, dim=0)
        if self.output_from == "logits":
            logits = torch.mean(output, axis=0)
            proba = torch.softmax(logits, dim=1)
        elif self.output_from == "probabilities":
            proba = torch.mean(output, axis=0)

        batch["preds"] = torch.argmax(proba, dim=1)

        return batch
