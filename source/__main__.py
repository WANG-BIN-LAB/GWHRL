
import hydra
import numpy as np
from omegaconf import DictConfig, open_dict

from source.dataset import dataset_factory
from source.models import model_factory
from source.components import lr_scheduler_factory, optimizers_factory, logger_factory
from source.training import training_factory
from datetime import datetime
import sys



import torch

torch.cuda.empty_cache()

def run_cv_experiment(cfg: DictConfig):
    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)


    trainer = training_factory(cfg, model, optimizers, lr_schedulers, dataloaders, logger)

    return trainer.train()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    try:
        results = run_cv_experiment(cfg)

        print("\n" + "=" * 80)
        print("CROSS-VALIDATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nFinal Results:")
        for metric, values in results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")
        print("=" * 80)

    except Exception as e:
        print(f"\n Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    debug_mode = True

    if debug_mode:
        # Ablation switch:
        #   biological: original biology-prior subnet partition
        #   random: random 8-subnet partition with biology-prior subnet sizes
        #   sequential: sequential equal partition, 8 subnets x 25 nodes
        ablation_partition = 'biological'

        sys.argv = [
            'source/__main__.py',
            '--multirun',
            'datasz=100p',
            'model=bnt',
            f'+model.subnet_partition={ablation_partition}',
            'dataset=ABIDE',
            'n_splits=10',
            'cv_epochs=200',
            'cv_lr=0.0005' ,
            'preprocess=mixup',
            'preprocess.alpha=0.2'
        ]

    main()
