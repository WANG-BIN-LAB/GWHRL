import gc
import random

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
import pandas as pd

from source.models.BNT.bnt import BrainNetworkTransformer
from source.models.BNT.bnt_ablation import (
    RandomSubnetBrainNetworkTransformer,
    SequentialEqualSubnetBrainNetworkTransformer,
)
from source.dataset.abide import load_abide_data
# from source.models.BNT.components.NTXent import LogContrastiveLoss


DEFAULT_SEED = 44

def set_seed(seed=DEFAULT_SEED):
    """Set Python, NumPy, and PyTorch seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)


def resolve_bnt_model(config: DictConfig):
    """
    Select the BNT variant used by N-fold training.

    Supported config examples:
        model.name=BrainNetworkTransformer model.subnet_partition=biological
        model.name=BrainNetworkTransformer model.subnet_partition=random
        model.name=BrainNetworkTransformer model.subnet_partition=sequential
        model.name=RandomSubnetBrainNetworkTransformer
        model.name=SequentialEqualSubnetBrainNetworkTransformer
    """
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "BrainNetworkTransformer")
    partition = model_cfg.get("subnet_partition", "biological")

    name_to_class = {
        "BrainNetworkTransformer": BrainNetworkTransformer,
        "RandomSubnetBrainNetworkTransformer": RandomSubnetBrainNetworkTransformer,
        "SequentialEqualSubnetBrainNetworkTransformer": SequentialEqualSubnetBrainNetworkTransformer,
    }

    partition_to_class = {
        "biological": BrainNetworkTransformer,
        "random": RandomSubnetBrainNetworkTransformer,
        "sequential": SequentialEqualSubnetBrainNetworkTransformer,
    }

    if model_name != "BrainNetworkTransformer":
        if model_name not in name_to_class:
            raise ValueError(f"Unsupported BNT model name: {model_name}")
        model_class = name_to_class[model_name]
    else:
        if partition not in partition_to_class:
            raise ValueError(f"Unsupported subnet_partition: {partition}")
        model_class = partition_to_class[partition]

    return model_class, model_class.__name__


def apply_mixup(time_series, node_feature, label, alpha=0.4):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = label.size(0)
    index = torch.randperm(batch_size).to(label.device)

    mixed_time_series = lam * time_series + (1 - lam) * time_series[index]
    mixed_node_feature = lam * node_feature + (1 - lam) * node_feature[index]

    y_a = label
    y_b = label[index]

    return mixed_time_series, mixed_node_feature, y_a, y_b, lam


class NFoldCrossValidationTrain:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.dataloaders = dataloaders

        self.n_splits = cfg.get('n_splits', 10)
        self.cv_epochs = cfg.get('cv_epochs', 200)
        self.cv_lr = cfg.get('cv_lr', 0.0005)
        self.seed = cfg.get('seed', DEFAULT_SEED)

        self.use_mixup = cfg.get('preprocess', {}).get('name', '') == 'continus_mixup'
        self.mixup_alpha = cfg.get('preprocess', {}).get('alpha', 0.2)
        self.model_class, self.model_name = resolve_bnt_model(cfg)

        self.log_path = Path(cfg.log_path)
        self.unique_id = cfg.unique_id

        self.save_dir = self.log_path / f"CV_{self.n_splits}fold_{self.unique_id}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.save_dir / "training.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info(f"Training logs will be saved to: {log_file}")

    def train_and_evaluate(self, time_series_data, pearson_data, labels_data):

        set_seed(self.seed)

        features_ts = torch.FloatTensor(time_series_data)
        features_pearson = torch.FloatTensor(pearson_data)
        labels = torch.LongTensor(labels_data)

        # best_accuracy = 0.0
        # best_model_state = None
        # best_metrics = {}

        model_initialized = False
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        results = {
            'train_accuracy': [],
            'val_accuracy': [],
            'f1': [],
            'auc': [],
            'sensitivity': [],
            'specificity': []
        }

        best_fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(features_ts, labels)):
            fold_seed = self.seed + fold
            set_seed(fold_seed)

            self.logger.info(f"\n{'=' * 70}")
            self.logger.info(f"Training Fold {fold + 1}/{self.n_splits}")
            self.logger.info(f"{'=' * 70}")
            self.logger.info(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
            self.logger.info(f"Label distribution - Train: {np.bincount(labels_data[train_idx])}, "
                             f"Val: {np.bincount(labels_data[val_idx])}")

            X_train_ts = features_ts[train_idx]
            X_train_pearson = features_pearson[train_idx]
            y_train = labels[train_idx]

            X_val_ts = features_ts[val_idx]
            X_val_pearson = features_pearson[val_idx]
            y_val = labels[val_idx]

            train_dataset = utils.TensorDataset(X_train_ts, X_train_pearson, y_train)
            val_dataset = utils.TensorDataset(X_val_ts, X_val_pearson, y_val)
            train_generator = torch.Generator()
            train_generator.manual_seed(fold_seed)

            train_loader = utils.DataLoader(
                train_dataset,
                batch_size=self.config.dataset.batch_size,
                shuffle=True,
                drop_last=False,
                generator=train_generator
            )
            val_loader = utils.DataLoader(
                val_dataset,
                batch_size=self.config.dataset.batch_size,
                shuffle=False,
                drop_last=False
            )

            model = self.model_class(self.config).cuda()

            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.cv_lr, weight_decay=1e-3)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cv_epochs, eta_min=1e-6
            )

            fold_best_accuracy = 0
            fold_best_metrics = {}

            patience = 50
            patience_counter = 0
            best_val_accuracy_for_early_stop = 0.0

            for epoch in range(self.cv_epochs):
                model.train()
                epoch_loss = 0.0

                n_batches = 0
                train_correct = 0
                train_total = 0

                for time_series, node_feature, label in train_loader:
                    time_series,node_feature,label = time_series.cuda(),node_feature.cuda(),label.cuda()

                    if self.use_mixup and epoch > 0:
                        time_series, node_feature, y_a, y_b, lam = apply_mixup(
                            time_series, node_feature, label, alpha=self.mixup_alpha
                        )

                        optimizer.zero_grad()
                        outputs, final_feat = model(time_series, node_feature, label)

                        cls_loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                        # # 对比损失（使用最终特征和原始FC矩阵）
                        # if lambda_contrast > 0:
                        #     contrast_loss = lam * contrastive_loss(final_feat, y_a) + (1 - lam) * contrastive_loss(final_feat, y_b)
                        # else:
                        #     contrast_loss = torch.tensor(0.0, device=cls_loss.device)
                    else:
                        optimizer.zero_grad()
                        outputs, final_feat = model(time_series, node_feature, label)
                        # loss = criterion(outputs, label)
                        cls_loss = criterion(outputs, label)
                        # if lambda_contrast > 0:
                        #     contrast_loss = contrastive_loss(final_feat, label)
                        # else:
                        #     contrast_loss = torch.tensor(0.0, device=cls_loss.device)


                    # loss = cls_loss + lambda_contrast * contrast_loss
                    loss = cls_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                    _, predicted = torch.max(outputs, 1)
                    train_total += label.size(0)
                    train_correct += (predicted == label).sum().item()

                    del time_series, node_feature, label, outputs
                    torch.cuda.empty_cache()

                scheduler.step()

                avg_train_loss = epoch_loss / n_batches
                train_accuracy = 100.0 * train_correct / train_total

                if epoch % 1 == 0 or epoch == self.cv_epochs - 1:
                    model.eval()
                    val_preds_prob = []
                    val_labels_true = []
                    val_correct = 0
                    val_total = 0
                    val_loss = 0.0
                    val_n_batches = 0

                    with torch.no_grad():
                        for time_series, node_feature, label in val_loader:
                            time_series = time_series.cuda()
                            node_feature = node_feature.cuda()
                            label = label.cuda()
                            output, _ = model(time_series, node_feature, label)

                            val_loss_batch = criterion(output, label)
                            val_loss += val_loss_batch.item()
                            val_n_batches += 1

                            _, predicted = torch.max(output, 1)

                            val_correct += (predicted == label).sum().item()
                            val_total += label.size(0)

                            probs = F.softmax(output, dim=1)[:, 1].cpu().numpy()
                            val_preds_prob.extend(probs.tolist())
                            val_labels_true.extend(label.cpu().numpy())

                            del time_series, node_feature, label, output
                            torch.cuda.empty_cache()

                    # val_accuracy = 100.0 * val_correct / val_total
                    avg_val_loss = val_loss / val_n_batches

                    # 计算指标
                    val_preds_class = (np.array(val_preds_prob) > 0.5).astype(int)
                    val_labels_array = np.array(val_labels_true)

                    try:
                        tn, fp, fn, tp = confusion_matrix(val_labels_array, val_preds_class, labels=[0, 1]).ravel()
                        val_accuracy = accuracy_score(val_labels_array, val_preds_class)
                        f1 = f1_score(val_labels_array, val_preds_class, zero_division=0)

                        if len(np.unique(val_labels_array)) > 1:
                            auc = roc_auc_score(val_labels_array, val_preds_prob)
                        else:
                            auc = 0.0

                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                    except Exception as e:
                        self.logger.warning(f"Error calculating metrics at epoch {epoch}: {e}")
                        val_accuracy, f1, auc, sensitivity, specificity = 0.0, 0.0, 0.0, 0.0, 0.0

                    if val_accuracy > fold_best_accuracy:
                        fold_best_accuracy = val_accuracy
                        fold_best_metrics = {
                            'epoch': epoch + 1,
                            'train_accuracy': train_accuracy,
                            'val_accuracy': val_accuracy * 100.0,
                            'f1': f1,
                            'auc': auc,
                            'sensitivity': sensitivity,
                            'specificity': specificity
                        }

                    if val_accuracy > best_val_accuracy_for_early_stop:
                        best_val_accuracy_for_early_stop = val_accuracy
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    self.logger.info(
                        f"Fold {fold + 1} | Epoch {epoch + 1:3d}/{self.cv_epochs} | "
                        # f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                        # f"Cls: {epoch_cls_loss/n_batches:.4f} | Con: {epoch_con_loss/n_batches:.4f} | "
                        f"Train Loss Total: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy * 100:.2f}% | "
                        f"Sens: {sensitivity:.4f} | Best Val Acc: {fold_best_accuracy * 100:.2f}%"
                    )

                    if patience_counter >= patience:
                        self.logger.info(f"\nEarly stopping triggered at epoch {epoch + 1} for Fold {fold + 1}")
                        self.logger.info(f"Validation accuracy did not improve for {patience} consecutive epochs.")
                        break

            best_fold_metrics.append(fold_best_metrics)

            self.logger.info(f"\nFold {fold + 1} Best Metrics:")
            self.logger.info(f"  Epoch: {fold_best_metrics['epoch'] +1 }")
            self.logger.info(f"  Train Accuracy: {fold_best_metrics['train_accuracy']:.2f}%")
            self.logger.info(f"  Val Accuracy: {fold_best_metrics['val_accuracy']:.2f}%")
            self.logger.info(f"  F1: {fold_best_metrics['f1']:.4f}")
            self.logger.info(f"  AUC: {fold_best_metrics['auc']:.4f}")
            self.logger.info(f"  Sensitivity: {fold_best_metrics['sensitivity']:.4f}")
            self.logger.info(f"  Specificity: {fold_best_metrics['specificity']:.4f}")

            for metric in results:
                results[metric].append(fold_best_metrics[metric])

            del model, optimizer, scheduler, criterion
            torch.cuda.empty_cache()
            gc.collect()

        self.logger.info("\n" + "=" * 70)
        self.logger.info("Final Cross-Validation Results")
        self.logger.info("=" * 70)

        final_stats = {}
        for metric in results:
            mean_val = np.mean(results[metric])
            std_val = np.std(results[metric])
            final_stats[metric] = {'mean': mean_val, 'std': std_val}
            self.logger.info(f"Mean {metric:11s}: {mean_val:.4f} ± {std_val:.4f}")

        self.logger.info("\n" + "=" * 70)
        self.logger.info("Model Complexity Summary")
        self.logger.info("=" * 70)
        self.logger.info(f"Total parameters:     {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Non-trainable parameters: {non_trainable_params:,}")
        self.logger.info(f"Memory footprint:     {total_params * 4 / (1024 ** 2):.2f} MB")
        self.logger.info("=" * 70)

        self.save_cv_results(results, final_stats, best_fold_metrics)

        return results

    def save_cv_results(self, results, final_stats, best_fold_metrics):
        summary_file = self.save_dir / "cv_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"{self.n_splits}-Fold Cross-Validation Summary\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Dataset: {self.config.dataset.name}\n")
            f.write(f"Epochs per fold: {self.cv_epochs}\n")
            f.write(f"Learning rate: {self.cv_lr}\n")
            f.write(f"Batch size: {self.config.dataset.batch_size}\n")
            f.write("=" * 70 + "\n\n")

            f.write("Overall Statistics:\n")
            f.write("-" * 70 + "\n")
            for metric_name, stats in final_stats.items():
                f.write(f"{metric_name.upper():15s}: Mean = {stats['mean']:.4f}, Std = {stats['std']:.4f}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("Detailed Results Per Fold\n")
            f.write("=" * 70 + "\n")

            for fold_idx, fold_metrics in enumerate(best_fold_metrics):
                f.write(f"\nFold {fold_idx + 1}:\n")
                f.write("-" * 70 + "\n")
                for key, value in fold_metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {key:15s}: {value:.4f}\n")
                    else:
                        f.write(f"  {key:15s}: {value}\n")

        self.logger.info(f"\nCross-validation summary saved to: {summary_file}")

        csv_data = []
        for fold_idx, fold_metrics in enumerate(best_fold_metrics):
            row = {'Fold': fold_idx + 1}
            row.update(fold_metrics)
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_file = self.save_dir / "cv_results.csv"
        df.to_csv(csv_file, index=False, float_format='%.4f')
        self.logger.info(f"Cross-validation results saved to: {csv_file}")


    def train(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING N-FOLD CROSS-VALIDATION TRAINING")
        self.logger.info("=" * 80)
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Dataset: {self.config.dataset.name}")
        self.logger.info(f"N-fold: {self.n_splits}")
        self.logger.info(f"Batch size: {self.config.dataset.batch_size}")
        self.logger.info(f"Epochs per fold: {self.cv_epochs}")
        self.logger.info(f"Learning rate: {self.cv_lr}")
        self.logger.info(f"Mixup alpha: {self.mixup_alpha}")

        if self.config.dataset.name == 'abide':
            all_time_series, all_pearson, all_labels, site = load_abide_data(self.config)

            if isinstance(all_time_series, torch.Tensor):
                all_time_series = all_time_series.cpu().numpy()
            if isinstance(all_pearson, torch.Tensor):
                all_pearson = all_pearson.cpu().numpy()
            if isinstance(all_labels, torch.Tensor):
                all_labels = all_labels.cpu().numpy()

            all_labels = all_labels.astype(np.int64)

        else:
            raise NotImplementedError(f"Dataset {self.config.dataset.name} not supported yet")

        total_samples = len(all_labels)
        self.logger.info(f"\nTotal samples: {total_samples}")
        self.logger.info(f"Label distribution: {np.bincount(all_labels)}")
        self.logger.info(f"Samples per fold (approx): {total_samples // self.n_splits}")

        results = self.train_and_evaluate(
            time_series_data=all_time_series,
            pearson_data=all_pearson,
            labels_data=all_labels
        )

        self.logger.info("\n N-Fold Cross-Validation Training Completed Successfully!")
        return results
