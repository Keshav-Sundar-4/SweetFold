import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import random

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from omegaconf import OmegaConf, listconfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profilers import PyTorchProfiler

# Assuming these are the correct import paths based on your structure
# Import DataConfig explicitly to check its type later
from boltz.data.module.training import BoltzTrainingDataModule, DataConfig
# Import your specific model class
from boltz.model.model import Boltz1

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class GradientDebugger(Callback):
    """
    This callback inspects gradients after the backward pass to find
    parameters that were not used in the computation graph, which is the
    root cause of the DDP "find_unused_parameters" error.
    """
    def on_after_backward(self, trainer, pl_module):
        # This hook runs after loss.backward() and before optimizer.step()
        
        # We only need to check on one rank, as the error is synchronized.
        if trainer.global_rank == 0:
            unused_params = []
            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is None:
                    unused_params.append(name)
            
            if unused_params:
                print("="*100, flush=True)
                print(f"🔥🔥🔥 DDP GRADIENT DEBUGGER | RANK {trainer.global_rank} 🔥🔥🔥", flush=True)
                print("Found parameters that require gradients but did not receive them.", flush=True)
                print("This is the direct cause of the DDP crash.", flush=True)
                print("The following parameters were likely part of a conditionally skipped module:", flush=True)
                for name in unused_params:
                    print(f"  - {name}", flush=True)
                print("="*100, flush=True)


# Point save_dir at your existing folder
csv_logger = CSVLogger(
    save_dir="/work/keshavsundar/work_sundar/glycan_test",
    name="boltz_losses"
)


@dataclass
class TrainConfig:
    """Train configuration.

    Attributes
    ----------
    data : DataConfig
        The data configuration.
    model : LightningModule # Or Boltz1
        The model configuration.
    output : str
        The output directory.
    trainer : Optional[dict]
        The trainer configuration.
    resume : Optional[str]
        The resume checkpoint.
    pretrained : Optional[str]
        The pretrained model.
    wandb : Optional[dict]
        The wandb configuration.
    disable_checkpoint : bool
        Disable checkpoint.
    matmul_precision : Optional[str]
        The matmul precision.
    find_unused_parameters : Optional[bool]
        Find unused parameters.
    save_top_k : Optional[int]
        Save top k checkpoints.
    validation_only : bool
        Run validation only.
    debug : bool
        Debug mode.
    strict_loading : bool
        Fail on mismatched checkpoint weights.
    load_confidence_from_trunk: Optional[bool]
        Load pre-trained confidence weights from trunk.

    """

    data: DataConfig
    model: LightningModule # Or Boltz1
    output: str
    trainer: Optional[dict] = None
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    wandb: Optional[dict] = None
    disable_checkpoint: bool = False
    matmul_precision: Optional[str] = None
    find_unused_parameters: Optional[bool] = False # Keep original default
    save_top_k: Optional[int] = 1
    validation_only: bool = False
    debug: bool = False
    strict_loading: bool = True
    load_confidence_from_trunk: Optional[bool] = False


def train(raw_config: str, args: list[str]) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run training.

    Parameters
    ----------
    raw_config : str
        The input yaml configuration.
    args : list[str]
        Any command line overrides.

    """
    # --- UNCHANGED CODE: CONFIG LOADING (OMITTED FOR BREVITY) ---
    # Load the configuration
    raw_config = omegaconf.OmegaConf.load(raw_config)
    # Apply input arguments
    args = omegaconf.OmegaConf.from_dotlist(args)
    raw_config = omegaconf.OmegaConf.merge(raw_config, args)
    # Instantiate the task configuration using hydra
    cfg_dict = hydra.utils.instantiate(raw_config)
    # Create the TrainConfig dataclass instance from the instantiated dict
    cfg = TrainConfig(**cfg_dict)
    # Set matmul precision
    if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)
    # Create trainer dict from config
    trainer_cfg = cfg.trainer 
    if trainer_cfg is None:
        trainer_cfg = {}
    # Flip some arguments in debug mode
    devices = trainer_cfg.get("devices", 1)
    num_devices = 1 
    if isinstance(devices, int):
        num_devices = devices
    elif isinstance(devices, (list, listconfig.ListConfig)):
        num_devices = len(devices)
    wandb_cfg = cfg.wandb 
    if cfg.debug:
        if isinstance(devices, int):
            devices = 1
        elif isinstance(devices, (list, listconfig.ListConfig)):
            devices = [devices[0]]
        trainer_cfg["devices"] = devices
        if isinstance(cfg.data, DataConfig) and hasattr(cfg.data, 'num_workers'):
                cfg.data.num_workers = 0
        elif isinstance(cfg.data, (dict, omegaconf.DictConfig)) and 'num_workers' in cfg.data:
                cfg.data['num_workers'] = 0
        if wandb_cfg:
            wandb_cfg = None 
    # Create objects
    if isinstance(cfg.data, DataConfig):
            data_module = BoltzTrainingDataModule(cfg.data)
    else: 
            data_module = BoltzTrainingDataModule(DataConfig(**cfg.data))
    # Model loading - use the already instantiated model from cfg
    model_module = cfg.model 
    # --- UNCHANGED CODE: PRETRAINED WEIGHT LOADING (OMITTED FOR BREVITY) ---
    if cfg.pretrained and not cfg.resume:
        if cfg.load_confidence_from_trunk:
            checkpoint = torch.load(cfg.pretrained, map_location="cpu")
            new_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("structure_module") and not key.startswith(
                    "distogram_module"
                ):
                    new_key = "confidence_module." + key
                    new_state_dict[new_key] = value
            new_state_dict.update(checkpoint["state_dict"])
            checkpoint["state_dict"] = new_state_dict
            random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            file_path = os.path.dirname(cfg.pretrained) + "/" + random_string + ".ckpt"
            print(f"Saving modified checkpoint to {file_path} created by broadcasting trunk of {cfg.pretrained} to confidence module.")
            torch.save(checkpoint, file_path)
        else:
            file_path = cfg.pretrained

        print(f"Loading pretrained weights from {file_path} into existing model structure.")
        try:
            
            model_module = type(model_module).load_from_checkpoint(
                file_path,
                map_location="cpu",
                strict=cfg.strict_loading, 
                **(model_module.hparams) 
            )


        except Exception as e:
            print(f"Error loading checkpoint with load_from_checkpoint: {e}")
            print("Attempting to load state dict directly...")
            checkpoint = torch.load(file_path, map_location="cpu")
            state_dict = checkpoint.get('state_dict', checkpoint) 
            load_result = model_module.load_state_dict(state_dict, strict=cfg.strict_loading)
            print(f"State dict load result: {load_result}")


        if cfg.load_confidence_from_trunk:
            os.remove(file_path) 

    elif cfg.resume:
        print(f"Attempting to resume training from: {cfg.resume}")
        pass 
    
    total_trainable = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    print(f"Total trainable parameters before freezing: {total_trainable}")

    # ========================== START: MODIFIED CODE ==========================
    # Create checkpoint callback
    callbacks = []

    # ADD THE GRADIENT DEBUGGER TO THE CALLBACKS LIST
    grad_debugger = GradientDebugger()
    callbacks.append(grad_debugger)
    
    dirpath = Path(cfg.output)
    dirpath.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
    if not cfg.disable_checkpoint:
        mc = ModelCheckpoint(
            monitor="val/lddt", 
            dirpath=dirpath / "checkpoints", 
            filename="{epoch}-{step}-{val/lddt:.4f}", 
            save_top_k=cfg.save_top_k, 
            save_last=True,
            mode="max", 
            every_n_epochs=1, 
        )
        callbacks.append(mc)
    # =========================== END: MODIFIED CODE ===========================

    # --- UNCHANGED CODE: LOGGER AND STRATEGY SETUP (OMITTED FOR BREVITY) ---
    loggers = []
    if wandb_cfg:
        resolved_wandb_cfg = {}
        if isinstance(wandb_cfg, omegaconf.DictConfig):
            resolved_wandb_cfg = OmegaConf.to_container(wandb_cfg, resolve=True)
        elif isinstance(wandb_cfg, dict):
            resolved_wandb_cfg = wandb_cfg

        wdb_logger = WandbLogger(
            name=resolved_wandb_cfg.get("name", "boltz_train"),
            group=resolved_wandb_cfg.get("group", None),
            save_dir=str(dirpath),
            project=resolved_wandb_cfg.get("project", "boltz"),
            entity=resolved_wandb_cfg.get("entity", None),
            log_model=False,
        )
        loggers.append(wdb_logger)

        @rank_zero_only
        def save_config_to_wandb() -> None:
            config_out = dirpath / "run.yaml"
            with config_out.open("w") as f:
                OmegaConf.save(raw_config, f)
            try:
                wdb_logger.experiment.save(str(config_out))
                resolved_config_dict = OmegaConf.to_container(raw_config, resolve=True)
                wdb_logger.experiment.config.update(resolved_config_dict)
            except Exception as e:
                print(f"Warning: Could not save config to wandb: {e}")

        save_config_to_wandb()

    else:
        tb_logger = TensorBoardLogger(
            save_dir=str(dirpath),      
            name="tb_logs",             
        )
        loggers.append(tb_logger)
    

    loggers.append(csv_logger)
    # Set up trainer strategy
    strategy = "auto"
    find_unused_ddp = False
    if num_devices > 1:
        find_unused_ddp = False 
        print(f"Using DDP strategy with find_unused_parameters={find_unused_ddp} (required due to frozen weights)")
        strategy = DDPStrategy(find_unused_parameters=find_unused_ddp)
    elif cfg.find_unused_parameters and num_devices <= 1:
        print(f"Warning: find_unused_parameters={cfg.find_unused_parameters} requested but only {num_devices} device(s).")
    
    trainer = pl.Trainer(
        default_root_dir=str(dirpath),
        strategy=strategy,
        callbacks=callbacks, # Pass the list containing our debugger
        logger=loggers,
        enable_checkpointing=not cfg.disable_checkpoint,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=10,
        **trainer_cfg,
    )

    # --- UNCHANGED CODE: VALIDATION AND FIT CALLS ---
    if not cfg.strict_loading:
        if hasattr(model_module, 'strict_loading'):
                model_module.strict_loading = False
        else:
                pass 
    if cfg.validation_only:
        print("Running validation only...")
        trainer.validate(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume or cfg.pretrained,
        )
    else:
        print("Starting training...")
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume, 
        )



if __name__ == "__main__":
    if len(sys.argv) < 2:
         print("Usage: python train.py <path_to_config.yaml> [hydra.overrides...]")
         sys.exit(1)
    arg1 = sys.argv[1] # Config path
    arg2 = sys.argv[2:] # Hydra overrides
    train(arg1, arg2)

