import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from itertools import combinations
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast

from utils.dataset_synapse import (Synapse_preloaded_dataset, Synapse_dataset,
                                    ACDC_preloaded_dataset, ACDCdataset, RandomGenerator)
from utils.utils import powerset  # , cal_params_flops
from utils.utils import one_hot_encoder
from utils.utils import DiceLoss
from utils.utils import val_single_volume


class WeightedFusion(nn.Module):
    """
    Computes per-pixel attention weights for each prediction from a set of decoder stages,
    then fuses them as a weighted sum. This preserves the original logit values without adding
    extra non-linearities that might distort class scores.
    """
    def __init__(self, num_stages, num_classes):
        super(WeightedFusion, self).__init__()
        self.weight_convs = nn.ModuleList(
            [nn.Conv2d(num_classes, 1, kernel_size=1) for _ in range(num_stages)]
        )

    def forward(self, predictions):
        weight_maps = [conv(pred) for pred, conv in zip(predictions, self.weight_convs)]
        weights = torch.stack(weight_maps, dim=1)
        weights = F.softmax(weights, dim=1)
        preds = torch.stack(predictions, dim=1)
        fused = torch.sum(weights * preds, dim=1)
        return fused


class CombinatorialMutationsLossModule(nn.Module):
    """
    Free-floating approach:
      - Each raw param -> a positive weight via Softplus.
      - No sum constraint.
      - The ratio w_i / w_j is determined by (theta_i - theta_j).
      - The total can be anything the model finds optimal.
    """
    def __init__(
        self,
        original_num_maps,
        num_classes,
        selecetd_num_maps,
        operations=None,
        use_learnable_weights=True,
        supervision='mutation',
        lc1=0.3,
        lc2=0.7
    ):
        super().__init__()
        if operations is None:
            operations = ['add', 'sub', 'mul', 'concat', 'weighted_fusion', 'avg', 'max']

        self.num_maps = selecetd_num_maps
        self.num_classes = num_classes
        self.operations = operations
        self.use_learnable_weights = use_learnable_weights
        self.supervision = supervision
        self.lc1 = lc1
        self.lc2 = lc2

        if use_learnable_weights:
            self.original_weights = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(original_num_maps)]
            )

        self.combination_indices = {}
        if use_learnable_weights:
            self.synthesized_weights = nn.ModuleDict()

        for op in self.operations:
            comb_list = []
            for k in range(2, selecetd_num_maps + 1):
                comb_list.extend(list(combinations(range(selecetd_num_maps), k)))
            self.combination_indices[op] = comb_list

            if use_learnable_weights:
                self.synthesized_weights[op] = nn.ParameterList(
                    [nn.Parameter(torch.zeros(1)) for _ in range(len(comb_list))]
                )

        # Weighted fusion modules
        self.weighted_fusion_modules = nn.ModuleDict({
            str(k): WeightedFusion(k, num_classes)
            for k in range(2, selecetd_num_maps + 1)
        })

        # FIX: pre-register the 1x1 convs used by the 'concat' op instead of
        # instantiating (and randomly re-initializing) a new nn.Conv2d on every
        # forward call. Previously this both stalled the GPU (fresh param alloc +
        # host->device copy every batch) and was a silent correctness bug, since
        # a conv created inside forward() is never added to self.parameters() and
        # therefore never trained by the optimizer.
        self.concat_convs = nn.ModuleDict()
        if 'concat' in self.operations:
            for idx, comb in enumerate(self.combination_indices['concat']):
                in_ch = num_classes * len(comb)
                self.concat_convs[str(idx)] = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def _compute_all_weights(self):
        if not self.use_learnable_weights:
            return None, None

        orig_vals = [F.softplus(w) for w in self.original_weights]

        synth_vals = {}
        for op, plist in self.synthesized_weights.items():
            synth_vals[op] = [F.softplus(p) for p in plist]

        return orig_vals, synth_vals

    def _generate_mutations(self, maps):
        """
        VECTORIZED map generation. `maps` is [M, B, C, H, W] (M = original
        number of decoder outputs). Returns:
          - fused_logits: python list of [B, C, H, W] tensors, one per
            synthesized combo, in the SAME order as before (grouped by op,
            then by combo index within that op) so downstream weight
            bookkeeping still lines up 1:1 by position.
        """
        device = maps.device
        M, B, C, H, W = maps.shape
        elementwise_ops = {'add', 'avg', 'mul', 'sub', 'max'}
        fused_logits = []

        for op in self.operations:
            combos = self.combination_indices[op]
            if not combos:
                continue

            by_k = {}
            for local_idx, comb in enumerate(combos):
                by_k.setdefault(len(comb), []).append((local_idx, comb))

            produced = {}

            if op in elementwise_ops:
                for k, items in by_k.items():
                    idx_tensor = torch.tensor([comb for _, comb in items], device=device, dtype=torch.long)
                    gathered = maps[idx_tensor]
                    if op == 'add':
                        combined = gathered.sum(dim=1)
                    elif op == 'avg':
                        combined = gathered.mean(dim=1)
                    elif op == 'mul':
                        combined = gathered.prod(dim=1)
                    elif op == 'max':
                        combined = gathered.max(dim=1).values
                    elif op == 'sub':
                        combined = gathered[:, 0] - gathered[:, 1:].sum(dim=1)
                    for j, (local_idx, _) in enumerate(items):
                        produced[local_idx] = combined[j]

            elif op == 'concat':
                for k, items in by_k.items():
                    cat_inputs, weights, biases = [], [], []
                    for local_idx, comb in items:
                        cat_inputs.append(torch.cat([maps[c] for c in comb], dim=1))
                        conv = self.concat_convs[str(local_idx)]
                        weights.append(conv.weight)
                        if conv.bias is not None:
                            biases.append(conv.bias)
                        else:
                            biases.append(torch.zeros(self.num_classes, device=device))
                    grouped_input = torch.cat(cat_inputs, dim=1)
                    grouped_weight = torch.cat(weights, dim=0)
                    grouped_bias = torch.cat(biases, dim=0)
                    grouped_out = F.conv2d(grouped_input, grouped_weight, grouped_bias, groups=len(items))
                    outs = grouped_out.split(self.num_classes, dim=1)
                    for (local_idx, _), out in zip(items, outs):
                        produced[local_idx] = out

            elif op in ('weighted_fusion', 'wf'):
                for k, items in by_k.items():
                    mod = self.weighted_fusion_modules[str(k)]
                    idx_tensor = torch.tensor([comb for _, comb in items], device=device, dtype=torch.long)
                    gathered = maps[idx_tensor]
                    Ck = gathered.shape[0]
                    weight_maps = []
                    for i in range(k):
                        stage_input = gathered[:, i].reshape(Ck * B, C, H, W)
                        wmap = mod.weight_convs[i](stage_input)
                        weight_maps.append(wmap.reshape(Ck, B, 1, H, W))
                    weights_stack = torch.stack(weight_maps, dim=1)
                    weights_stack = F.softmax(weights_stack, dim=1)
                    fused = (weights_stack * gathered).sum(dim=1)
                    for j, (local_idx, _) in enumerate(items):
                        produced[local_idx] = fused[j]
            else:
                raise ValueError(f"Unsupported op: {op}")

            for local_idx in range(len(combos)):
                fused_logits.append(produced[local_idx])

        return fused_logits

    def forward(self, output_maps, label_batch=None, ce_loss=None, dice_loss=None):
        device = output_maps[0].device
        compute_loss = label_batch is not None and ce_loss is not None and dice_loss is not None

        maps = torch.stack(output_maps, dim=0)
        M = maps.shape[0]

        fused_logits = self._generate_mutations(maps)

        if not compute_loss:
            return fused_logits

        if self.use_learnable_weights:
            orig_vals, synth_vals = self._compute_all_weights()
            weight_list = list(orig_vals)
            for op in self.operations:
                w_list = synth_vals[op]
                weight_list.extend(w_list[idx] for idx in range(len(w_list)))
        else:
            n_total = M + len(fused_logits)
            weight_list = [torch.ones(1, device=device) for _ in range(n_total)]

        weights_tensor = torch.cat(weight_list)

        all_logits = torch.stack(list(maps.unbind(0)) + fused_logits, dim=0)

        def per_map_losses(logits, label):
            ce = ce_loss(logits, label.long())
            dc = dice_loss(logits, label, softmax=True)
            return ce, dc

        try:
            from torch.func import vmap
            ces, dices = vmap(per_map_losses, in_dims=(0, None))(all_logits, label_batch)
        except Exception:
            ces_list, dices_list = [], []
            for i in range(all_logits.shape[0]):
                ce, dc = per_map_losses(all_logits[i], label_batch)
                ces_list.append(ce)
                dices_list.append(dc)
            ces = torch.stack(ces_list)
            dices = torch.stack(dices_list)

        combined_per_map = self.lc1 * ces + self.lc2 * dices
        weighted_per_map = combined_per_map * weights_tensor

        deep_supervision_loss = weighted_per_map[:M].sum()
        mutation_loss = weighted_per_map[M:].sum()

        if self.supervision in ['mutation', 'lomix']:
            final_loss = deep_supervision_loss + mutation_loss
        else:
            final_loss = deep_supervision_loss
        return final_loss, deep_supervision_loss, mutation_loss

    @torch.no_grad()
    def print_weights(self):
        if not self.use_learnable_weights:
            logging.info("No learnable weights. Using uniform weighting.")
            return

        raw_orig = [p.item() for p in self.original_weights]
        raw_synth = {}
        for op, plist in self.synthesized_weights.items():
            raw_synth[op] = [p.item() for p in plist]

        orig_vals, synth_vals = self._compute_all_weights()

        logging.info("Original Weights (raw): %s", " ".join(f"{x:.4f}" for x in raw_orig))
        logging.info("Original Weights (softplus): %s",
                      " ".join(f"{v.item():.4f}" for v in orig_vals))
        logging.info("   => sum(original) = %.4f", float(torch.stack(orig_vals).sum()))

        for op in self.operations:
            if op not in raw_synth:
                continue
            rvals = raw_synth[op]
            svals = synth_vals[op]
            logging.info("Synthesized Weights for '%s' (raw): %s", op,
                          " ".join(f"{rv:.4f}" for rv in rvals))
            logging.info("Synthesized Weights for '%s' (softplus): %s", op,
                          " ".join(f"{sv.item():.4f}" for sv in svals))
            logging.info("   => sum(%s) = %.4f", op, float(torch.stack(svals).sum()))

    def save_weights(self, save_path):
        torch.save(self.state_dict(), save_path)
        logging.info(f"Saved parameters to {save_path}")

    def load_weights(self, load_path):
        self.load_state_dict(torch.load(load_path))
        logging.info(f"Loaded parameters from {load_path}")


class ModelWithLoss(nn.Module):
    """
    Wraps the segmentation model and the loss module together so that a
    SINGLE DistributedDataParallel wrapper covers both sets of parameters.
    """
    def __init__(self, model, loss_module, supervision, lc1=0.3, lc2=0.7):
        super().__init__()
        self.model = model
        self.loss_module = loss_module
        self.supervision = supervision
        self.lc1 = lc1
        self.lc2 = lc2

    def forward(self, image_batch, label_batch, ce_loss, dice_loss):
        P = self.model(image_batch, mode='train')
        if not isinstance(P, list):
            P = [P]

        if self.supervision in ['mutation', 'lomix', 'deep_supervision']:
            loss, deep_supervision_loss, mutation_loss = self.loss_module(P, label_batch, ce_loss, dice_loss)
        else:
            loss_ce = ce_loss(P[-1], label_batch[:].long())
            loss_dice = dice_loss(P[-1], label_batch, softmax=True)
            loss = self.lc1 * loss_ce + self.lc2 * loss_dice
            deep_supervision_loss = torch.zeros((), device=image_batch.device)
            mutation_loss = torch.zeros((), device=image_batch.device)

        return loss, deep_supervision_loss, mutation_loss


def setup_distributed():
    """
    Reads the env vars torchrun sets (RANK, LOCAL_RANK, WORLD_SIZE) and
    initializes the NCCL process group.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # FIX: set the device BEFORE init, and pass device_id so NCCL knows
        # which GPU this process owns up front instead of inferring it the
        # first time a collective (e.g. dist.barrier()) runs. Without this,
        # PyTorch emits "barrier(): using the device under current context.
        # You can specify device_id in init_process_group to mute this
        # warning." on every barrier call. device_id is only accepted by
        # newer torch versions, so fall back gracefully on older ones.
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(backend="nccl", init_method="env://",
                                     device_id=torch.device(f"cuda:{local_rank}"))
        except TypeError:
            dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def is_main_process(rank):
    return rank == 0


@torch.no_grad()
def inference(args, model, best_performance, db_test, testloader, device, rank):
    if not is_main_process(rank):
        return best_performance

    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    # FIX: removed the per-volume tqdm bar (same reasoning as the training
    # loop) — Kaggle's non-tty captured output can't redraw a line in place,
    # so each throttled refresh became its own permanent printed line. This
    # now matches training's style: no per-item output, just the single
    # end-of-run summary line below.
    for i_batch, sampled_batch in enumerate(testloader):
        h, w = sampled_batch["image"].size()[2:]
        image, label = sampled_batch["image"], sampled_batch["label"]
        case_name = sampled_batch['case_name'][0]
        metric_i = val_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    print('[VAL] mean_dice: %.4f (best so far: %.4f)' % (performance, best_performance))
    model.train()
    return performance


@torch.no_grad()
def run_final_test(args, raw, device, rank, snapshot_path, test_dataset_cls, test_kwargs,
                    final_test_split_name, base_dir, writer, iter_num):
    """
    One-time evaluation on a genuinely held-out test split, run once after
    training finishes (not every epoch, unlike inference() above). Only
    called when final_test_split_name is not None — i.e. when the dataset
    actually has a test split distinct from whatever was used for per-epoch
    validation (Synapse doesn't; ACDC does).

    base_dir is passed in explicitly (rather than hardcoding args.volume_path)
    since ACDC's train/valid/test splits all live under ONE shared parent
    directory (base_dir/{train,valid,test}/...), unlike Synapse's convention
    of separate root_path (train) / volume_path (validation) directories.

    Loads the BEST checkpoint (best.pth) before evaluating, since the model
    left in memory at the end of training holds the LAST epoch's weights,
    not necessarily the best-performing ones. Falls back to last.pth with a
    warning if best.pth is missing for some reason.
    """
    if not is_main_process(rank):
        return None

    best_ckpt_path = os.path.join(snapshot_path, 'best.pth')
    last_ckpt_path = os.path.join(snapshot_path, 'last.pth')
    if os.path.exists(best_ckpt_path):
        ckpt_path = best_ckpt_path
    elif os.path.exists(last_ckpt_path):
        logging.info(f"[TEST] best.pth not found at {best_ckpt_path}; falling back to last.pth for final test.")
        ckpt_path = last_ckpt_path
    else:
        logging.info("[TEST] No checkpoint found (neither best.pth nor last.pth); skipping final test evaluation.")
        return None

    raw.model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    logging.info(f"[TEST] Loaded checkpoint for final test evaluation: {ckpt_path}")

    db_final_test = test_dataset_cls(base_dir=base_dir, split=final_test_split_name,
                                      list_dir=args.list_dir, **test_kwargs)
    final_testloader = DataLoader(db_final_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info(f"[TEST] {len(final_testloader)} final test iterations.")
    raw.model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(final_testloader):
        image, label = sampled_batch["image"], sampled_batch["label"]
        case_name = sampled_batch['case_name'][0]
        metric_i = val_single_volume(image, label, raw.model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_final_test)
    performance = np.mean(metric_list, axis=0)

    logging.info('[TEST] Final test performance: mean_dice : %f (checkpoint: %s)' % (performance, ckpt_path))
    print('[TEST] Final test mean_dice: %.4f (checkpoint: %s)' % (performance, ckpt_path))

    # Per-class breakdown: metric_list (before the np.mean(axis=0) collapse
    # above) is already the per-class dice, averaged across test cases —
    # one value per class val_single_volume iterated over. This codebase's
    # val_single_volume follows the common convention of looping
    # `for i in range(1, classes)`, i.e. skipping background (class 0), so
    # class_idx below is labeled starting at 1 to match. If your
    # val_single_volume includes background or orders classes differently,
    # adjust the starting index/labels here accordingly.
    per_class_dice = np.atleast_1d(metric_list)
    logging.info('[TEST] Per-class dice breakdown:')
    print('[TEST] Per-class dice breakdown:')
    for i, class_dice in enumerate(per_class_dice, start=1):
        logging.info('[TEST]   class_%d : %f' % (i, class_dice))
        print('[TEST]   class_%d: %.4f' % (i, class_dice))
        if writer is not None:
            writer.add_scalar(f'test/dice_class_{i}', class_dice, iter_num)

    if writer is not None:
        writer.add_scalar('test/mean_dice', performance, iter_num)

    results_path = os.path.join(snapshot_path, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"split: {final_test_split_name}\n")
        f.write(f"mean_dice: {performance:.6f}\n")
        f.write("per_class_dice:\n")
        for i, class_dice in enumerate(per_class_dice, start=1):
            f.write(f"  class_{i}: {class_dice:.6f}\n")
    logging.info(f"[TEST] Wrote final test results to {results_path}")

    return performance


def trainer_synapse(args, model, snapshot_path, supervision='lomix', operations=['add', 'mul', 'wf', 'concat'],
                     n_outs=4, use_learnable_weights=True, patience=30, min_delta=0.0):
    """
    patience: number of consecutive epochs with no validation mean_dice
        improvement (> min_delta) before training stops early. Set to a
        value >= args.max_epochs (or None-like large number) to effectively
        disable early stopping.
    min_delta: minimum increase in mean_dice to count as an "improvement"
        for early-stopping purposes (does not affect checkpoint-saving,
        which still saves on any performance >= best_performance as before).
    """
    rank, local_rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                             format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(f"Distributed: {is_distributed}, world_size: {world_size}")
        logging.info(f"Early stopping patience: {patience}, min_delta: {min_delta}")
    else:
        logging.basicConfig(level=logging.CRITICAL)

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size

    # Dispatch to the right dataset classes/kwargs based on args.dataset.
    # ACDC's classes take no nclass kwarg (unlike Synapse's 13->9 class
    # remap). val_split_name is the split used for PER-EPOCH validation and
    # checkpointing during training; final_test_split_name (if not None) is
    # a genuinely held-out split only evaluated ONCE after training
    # completes, on the best checkpoint.
    #
    # FIX: previously ACDC's per-epoch validation used split "test" — but
    # your data has train/valid/test as three distinct splits, so that was
    # silently using the held-out test set for validation every epoch (data
    # leakage into model-selection decisions) while "valid" sat unused. Now
    # "valid" is used for per-epoch validation, and "test" is reserved for
    # the one-time final evaluation below. Synapse has no genuinely separate
    # test split beyond test_vol (already used for validation), so
    # final_test_split_name is None for it — the post-training test pass is
    # skipped cleanly rather than just re-running the same validation split
    # again under a different label.
    dataset_name = getattr(args, 'dataset', 'Synapse')
    if dataset_name == 'ACDC':
        train_dataset_cls = ACDC_preloaded_dataset
        test_dataset_cls = ACDCdataset
        val_split_name = 'valid'
        final_test_split_name = 'test'
        train_kwargs = {}
        test_kwargs = {}
        # FIX: ACDC's train/valid/test all live under ONE shared parent dir
        # (e.g. data/ACDC/{train,valid,test}), unlike Synapse's convention of
        # a separate root_path (train) and volume_path (validation)
        # directory. Using args.root_path for every ACDC split means you
        # only need to pass --root_path data/ACDC — --volume_path is simply
        # unused for ACDC, instead of having to redundantly pass the same
        # path to both flags.
        eval_base_dir = args.root_path
    else:
        train_dataset_cls = Synapse_preloaded_dataset
        test_dataset_cls = Synapse_dataset
        val_split_name = 'test_vol'
        final_test_split_name = None
        train_kwargs = {'nclass': args.num_classes}
        test_kwargs = {'nclass': args.num_classes}
        eval_base_dir = args.volume_path

    db_train = train_dataset_cls(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                  transform=transforms.Compose(
                                      [RandomGenerator(output_size=[args.img_size, args.img_size])]),
                                  **train_kwargs)
    if is_main_process(rank):
        print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id + rank * 1000)

    if is_distributed:
        train_sampler = DistributedSampler(db_train, num_replicas=world_size, rank=rank, shuffle=True,
                                            seed=args.seed)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler,
                              num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,
                              persistent_workers=True, prefetch_factor=4, drop_last=is_distributed)

    if is_main_process(rank):
        db_test = test_dataset_cls(base_dir=eval_base_dir, split=val_split_name, list_dir=args.list_dir,
                                    **test_kwargs)
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    else:
        db_test, testloader = None, None

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    loss_module = CombinatorialMutationsLossModule(
        4, num_classes, selecetd_num_maps=n_outs, operations=operations,
        use_learnable_weights=use_learnable_weights
    )

    lc1, lc2 = 0.3, 0.7
    combined = ModelWithLoss(model, loss_module, supervision, lc1=lc1, lc2=lc2).to(device)

    if is_distributed:
        # FIX (corrected): my first attempt at this fix used
        # `grad.contiguous()`, which turned out to be a no-op here — see
        # below for why — so it never actually changed anything, which is
        # why the warning persisted.
        #
        # gradient_as_bucket_view only changes what happens AFTER DDP's
        # reducer copies a gradient into its bucket — it doesn't touch the
        # copy-in step itself, which is where the stride check that
        # triggers "Grad strides do not match bucket view strides" runs.
        # So that setting was never going to affect this warning either way.
        #
        # The real cause: cuDNN's backward pass for the depthwise/grouped
        # convs in the PVTv2 backbone's DWConv layers (groups=dim, weight
        # shape [C, 1, 3, 3] — e.g. [2048,1,3,3] in stage 4's MLP) produces
        # a gradient whose size-1 dimension has stride 1 instead of the
        # canonical 9. PyTorch's is_contiguous() treats size-1 dimensions as
        # stride-agnostic, so it reports this tensor as "already
        # contiguous" even though its strides don't match DDP's bucket
        # layout — which means `grad.contiguous()` short-circuits and
        # returns the exact same tensor, unchanged (verified: same
        # data_ptr(), same non-canonical strides). Only an unconditional
        # copy with an explicitly requested memory format actually
        # normalizes the strides; `.clone(memory_format=torch.contiguous_format)`
        # does this (plain `.contiguous()`, even with memory_format passed
        # explicitly, still short-circuits the same way).
        #
        # Scoped to only 4D conv weights shaped like a depthwise/grouped
        # conv (shape[1] == 1) so we're not paying a real copy on every
        # parameter's gradient every backward pass — just the handful that
        # actually need it.
        def _force_canonical_grad_strides(grad):
            return grad.clone(memory_format=torch.contiguous_format)

        for p in combined.parameters():
            if p.requires_grad and p.dim() == 4 and p.shape[1] == 1:
                p.register_hook(_force_canonical_grad_strides)

        combined = DDP(combined, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False)
        raw = combined.module
    else:
        raw = combined

    combined.train()
    optimizer = optim.AdamW(combined.parameters(), lr=base_lr, weight_decay=0.0001)

    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)

    writer = SummaryWriter(snapshot_path + '/log') if is_main_process(rank) else None
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    if is_main_process(rank):
        logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    # LR SCHEDULER: polynomial decay, the standard policy for this codebase
    # family (TransUNet/CASCADE/EMCAD-derived). Previously the training loop
    # set `lr_ = base_lr` unconditionally every iteration — a no-op that
    # never actually decayed the learning rate. LambdaLR now handles this
    # via scheduler.step() called once per iteration in the training loop.
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda it: (1.0 - it / max_iterations) ** 0.9
    )

    best_performance = 0.0
    epochs_no_improve = 0  # EARLY STOPPING: consecutive epochs with no val improvement
    iterator = tqdm(range(max_epoch), ncols=70) if is_main_process(rank) else range(max_epoch)

    for epoch_num in iterator:
        if is_distributed:
            train_sampler.set_epoch(epoch_num)

        # FIX: dropped the per-batch tqdm bar. In a real terminal it
        # redraws the SAME line via '\r', but Kaggle's captured (non-tty)
        # output can't do that — every throttled refresh became its own
        # permanent printed line, flooding the log once training got fast
        # (~2.5s/it means a new line roughly every second even at
        # mininterval=1.0). Iterating trainloader directly avoids that
        # entirely; live loss values are now shown on the OUTER per-epoch
        # bar's postfix instead (see below), which only advances once per
        # epoch by construction — so in this same non-tty fallback mode it
        # naturally prints exactly one new line per epoch.
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch = image_batch.to(device, non_blocking=True)
            label_batch = label_batch.squeeze(1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=use_amp):
                loss, deep_supervision_loss, mutation_loss = combined(image_batch, label_batch, ce_loss, dice_loss)

            scaler.scale(loss).backward()
            # FIX: track the AMP scale factor before/after scaler.update() so
            # we know whether scaler.step(optimizer) actually ran
            # optimizer.step() this iteration. When GradScaler detects an
            # inf/nan gradient (common in the first several iterations while
            # it's still calibrating its scale, or any time a batch produces
            # an overflow), it SKIPS the underlying optimizer.step() and
            # shrinks the scale instead. Calling scheduler.step() unconditionally
            # right after would then step the LR schedule for an iteration
            # where no actual optimizer update happened — which is exactly
            # what triggered the "lr_scheduler.step() before optimizer.step()"
            # warning. Comparing scale_before/after is the standard way to
            # detect a skipped step and only advance the scheduler on
            # iterations where optimizer.step() genuinely ran.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before

            # LR SCHEDULER: standard polynomial decay (this codebase's usual
            # "poly" policy) stepped once per iteration via a real
            # LambdaLR scheduler, instead of the previous dead
            # `lr_ = base_lr` (which never actually decayed).
            if optimizer_stepped:
                scheduler.step()
            lr_ = optimizer.param_groups[0]['lr']

            iter_num += 1

            if is_main_process(rank):
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                writer.add_scalar('info/deep_supervision_loss', deep_supervision_loss, iter_num)
                writer.add_scalar('info/mutation_loss', mutation_loss, iter_num)

                if iter_num % 50 == 0:
                    logging.info(
                        'iteration %d, epoch %d : loss : %f, deep_supervision_loss : %f, mutation_loss : %f, lr: %f' % (
                            iter_num, epoch_num, loss.item(), deep_supervision_loss.item(),
                            mutation_loss.item(), lr_))

                    # Log each combo's effective (post-softplus) learnable
                    # weight so their evolution can be tracked in
                    # TensorBoard. Throttled to the same cadence as the loss
                    # log line above rather than every iteration, since
                    # there are ~40+ of these and writing all of them every
                    # single step would be a lot of redundant I/O.
                    if raw.loss_module.use_learnable_weights:
                        with torch.no_grad():
                            for op, weights in raw.loss_module.synthesized_weights.items():
                                for i, weight in enumerate(weights):
                                    weight_val = F.softplus(weight).item()
                                    writer.add_scalar(f"weights/weight_{op}_{i}", weight_val, iter_num)

        if is_main_process(rank):
            # Update the OUTER epoch bar's postfix with the last batch's
            # loss values. Since this bar advances once per epoch (not per
            # batch), this is the only per-epoch redraw — exactly one new
            # line per epoch in non-tty output, none per batch.
            if hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'ds': f"{deep_supervision_loss.item():.4f}",
                    'mut': f"{mutation_loss.item():.4f}",
                    'lr': f"{lr_:.2e}",
                }, refresh=False)
            logging.info(
                'iteration %d, epoch %d : loss : %f, deep_supervision_loss : %f, mutation_loss : %f, lr: %f' % (
                    iter_num, epoch_num, loss.item(), deep_supervision_loss.item(), mutation_loss.item(), lr_))
            print('[TRAIN] epoch %d done -> loss: %.4f, deep_supervision_loss: %.4f, mutation_loss: %.4f, lr: %.2e' % (
                epoch_num, loss.item(), deep_supervision_loss.item(), mutation_loss.item(), lr_))

            raw.loss_module.print_weights()

            save_mode_path = os.path.join(snapshot_path, 'last.pth')
            torch.save(raw.model.state_dict(), save_mode_path)

        if is_distributed:
            dist.barrier()

        performance = inference(args, raw.model, best_performance, db_test, testloader, device, rank)
        prev_best = best_performance  # captured before checkpoint block below updates it, for early-stopping comparison

        if is_main_process(rank):
            save_interval = 50

            if best_performance <= performance:
                best_performance = performance
                save_mode_path = os.path.join(snapshot_path, 'best.pth')
                torch.save(raw.model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                raw.loss_module.save_weights(os.path.join(snapshot_path, 'combinatorial_loss_weights_best.pth'))

            if (epoch_num + 1) % save_interval == 0:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(raw.model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                raw.loss_module.save_weights(
                    os.path.join(snapshot_path, 'combinatorial_loss_weights_' + 'epoch_' + str(epoch_num) + '.pth'))

        if is_distributed:
            dist.barrier()

        # EARLY STOPPING: only rank 0 has a real `performance`/`prev_best`
        # (inference() is a no-op on other ranks), so the stop decision is
        # made on rank 0 and then broadcast to every rank. This is required
        # under DDP — if rank 0 alone decided to `break` while other ranks
        # kept looping, they'd call combined()'s forward/backward next epoch
        # expecting a gradient all-reduce from rank 0 that would never come,
        # hanging forever.
        stop_flag = 0
        if is_main_process(rank):
            if performance > prev_best + min_delta:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement in mean_dice for {epochs_no_improve} epoch(s) (patience={patience}).")
            if epochs_no_improve >= patience:
                stop_flag = 1
                logging.info(f"Early stopping triggered at epoch {epoch_num}: "
                              f"no mean_dice improvement for {patience} consecutive epochs.")
                print(f"[EARLY STOP] No mean_dice improvement for {patience} epochs. "
                      f"Stopping at epoch {epoch_num} (best mean_dice: {best_performance:.4f}).")

        if is_distributed:
            stop_tensor = torch.tensor([stop_flag], device=device, dtype=torch.int)
            dist.broadcast(stop_tensor, src=0)
            stop_flag = int(stop_tensor.item())

        if stop_flag:
            if is_main_process(rank) and hasattr(iterator, 'close'):
                iterator.close()
            break

        if epoch_num >= max_epoch - 1:
            if is_main_process(rank):
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(raw.model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                raw.loss_module.save_weights(
                    os.path.join(snapshot_path, 'combinatorial_loss_weights_' + 'epoch_' + str(epoch_num) + '.pth'))
                if hasattr(iterator, 'close'):
                    iterator.close()
            break

    # FINAL TEST: run once, after the training loop ends (whether it ended
    # via early stopping or reaching max_epochs), on the genuinely held-out
    # test split — only when the dataset actually has one distinct from
    # whatever was used for per-epoch validation (see final_test_split_name
    # set up near the top of this function). Runs only on rank 0; other
    # ranks wait at the barrier below so nothing tears down the process
    # group while rank 0 is still evaluating.
    if final_test_split_name is not None:
        if is_main_process(rank):
            run_final_test(args, raw, device, rank, snapshot_path, test_dataset_cls, test_kwargs,
                            final_test_split_name, eval_base_dir, writer, iter_num)
        if is_distributed:
            dist.barrier()
    else:
        if is_main_process(rank):
            logging.info(f"[TEST] No separate held-out test split for dataset '{dataset_name}' "
                          f"(its validation split already covers final evaluation); skipping.")

    if is_main_process(rank) and writer is not None:
        writer.close()

    if is_distributed:
        dist.destroy_process_group()

    return "Training Finished!"
