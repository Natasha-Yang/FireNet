import os
import sys
from pathlib import Path

import torch
import typer
from loguru import logger
from tqdm import tqdm
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero_only
import wandb

# Add the project root directory (FireNet) to the Python path
project_root = os.path.abspath(os.path.join(os.getcwd())) 
print(f"Project root: {project_root}")
sys.path.append(project_root)

from firenet.dataloader.FireSpreadDataModule import FireSpreadDataModule
from firenet.dataloader.FireSpreadDataset import FireSpreadDataset
from firenet.dataloader.utils import get_means_stds_missing_values
from firenet.models import BaseModel, SMPModel, ConvLSTMLightning, LogisticRegression  # noqa

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
torch.set_float32_matmul_precision('high')

app = typer.Typer()

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.default_root_dir",
                              "trainer.logger.init_args.save_dir")
        parser.link_arguments("model.class_path",
                              "trainer.logger.init_args.name")
        
        # added
        parser.add_argument('--config', type=str, default=None,
                            help="Path to config file.")
        


        parser.add_argument("--do_train", type=bool,
                            help="If True: skip training the model.")
        parser.add_argument("--do_predict", type=bool,
                            help="If True: compute predictions.")
        parser.add_argument("--do_test", type=bool,
                            help="If True: compute test metrics.")
        parser.add_argument("--do_validate", type=bool,
                            default=False, help="If True: compute val metrics.")
        parser.add_argument("--ckpt_path", type=str, default=None,
                            help="Path to checkpoint to load for resuming training, for testing and predicting.")

    def before_instantiate_classes(self):
        n_features = FireSpreadDataset.get_n_features(
            self.config.data.n_leading_observations,
            self.config.data.features_to_keep,
            self.config.data.remove_duplicate_features)
        self.config.model.init_args.n_channels = n_features

        train_years, _, _ = FireSpreadDataModule.split_fires(
            self.config.data.data_fold_id)
        _, _, missing_values_rates = get_means_stds_missing_values(train_years)
        fire_rate = 1 - missing_values_rates[-1]
        pos_class_weight = float(1 / fire_rate)
        self.config.model.init_args.pos_class_weight = pos_class_weight

    def before_fit(self):
        self.wandb_setup()

    def before_test(self):
        self.wandb_setup()

    def before_validate(self):
        self.wandb_setup()

    @rank_zero_only
    def wandb_setup(self):
        config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")
        cfg_string = self.parser.dump(self.config, skip_none=False)
        with open(config_file_name, "w") as f:
            f.write(cfg_string)
        wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
        wandb.define_metric("train_loss_epoch", summary="min")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("train_f1_epoch", summary="max")
        wandb.define_metric("val_f1", summary="max")


@app.command()
def main(
    features_path: Path = Path("data/processed/features.csv"),
    labels_path: Path = Path("data/processed/labels.csv"),
    model_path: Path = Path("models/model.pkl"),
    do_train: bool = True,
    do_validate: bool = False,
    do_test: bool = False,
    do_predict: bool = False,
    ckpt_path: str = None
):
    # Progress tracking
    logger.info("Starting model training...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Reached halfway point.")
    logger.success("Model training progress completed.")

    # Initiating LightningCLI
    cli = MyLightningCLI(BaseModel, FireSpreadDataModule, subclass_mode_model=True, save_config_kwargs={
        "overwrite": True}, parser_kwargs={"parser_mode": "yaml"}, run=False)
    cli.wandb_setup()

    if do_train:
        logger.info("Training the model...")
        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=ckpt_path)

    ckpt = ckpt_path if not do_train else "best"

    if do_validate:
        logger.info("Validating the model...")
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

    if do_test:
        logger.info("Testing the model...")
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

    if do_predict:
        logger.info("Generating predictions...")
        prediction_output = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path=ckpt)
        x_af = torch.cat(list(map(lambda tup: tup[0][:, -1, :, :].squeeze(), prediction_output)), axis=0)
        y = torch.cat(list(map(lambda tup: tup[1], prediction_output)), axis=0)
        y_hat = torch.cat(list(map(lambda tup: tup[2], prediction_output)), axis=0)
        fire_masks_combined = torch.cat([x_af.unsqueeze(0), y_hat.unsqueeze(0), y.unsqueeze(0)], axis=0)

        predictions_file_name = os.path.join(cli.config.trainer.default_root_dir, f"predictions_{wandb.run.id}.pt")
        torch.save(fire_masks_combined, predictions_file_name)

    logger.success("Modeling complete.")


if __name__ == "__main__":
    app()
