import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from itertools import combinations

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
from torch.cuda.amp import GradScaler, autocast

from utils.dataset_synapse import Synapse_preloaded_dataset, Synapse_dataset, RandomGenerator
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

        Previously each combo was produced by its own small Python-level
        sequence of ops (and, for 'concat'/'weighted_fusion', its own full
        module forward call) inside a doubly-nested Python loop — ~44 combos
        each dispatching several tiny CUDA kernels. Here, combos are grouped
        by (op, combo-size k) and processed as a SINGLE batched tensor
        operation per group, cutting the kernel-launch count from ~150+ down
        to roughly one per (op, k) group (a handful of groups total).
        """
        device = maps.device
        M, B, C, H, W = maps.shape
        elementwise_ops = {'add', 'avg', 'mul', 'sub', 'max'}
        fused_logits = []

        for op in self.operations:
            combos = self.combination_indices[op]
            if not combos:
                continue

            # group this op's combos by their size k, since gather/grouped-conv
            # shapes must be uniform within a batched call
            by_k = {}
            for local_idx, comb in enumerate(combos):
                by_k.setdefault(len(comb), []).append((local_idx, comb))

            # collect (local_idx -> tensor) so we can re-assemble in original order
            produced = {}

            if op in elementwise_ops:
                for k, items in by_k.items():
                    idx_tensor = torch.tensor([comb for _, comb in items], device=device, dtype=torch.long)  # [Ck, k]
                    gathered = maps[idx_tensor]  # [Ck, k, B, C, H, W]
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
                # Each combo has its OWN learned 1x1 conv (different weights),
                # so a plain shared conv can't be used. Instead we run every
                # same-size combo's conv as one GROUPED convolution: weights
                # and inputs from all combos in the group are concatenated
                # along the channel axis, and F.conv2d(..., groups=num_combos)
                # applies each combo's distinct weight to its own channel
                # slice in a single kernel call.
                for k, items in by_k.items():
                    cat_inputs, weights, biases = [], [], []
                    for local_idx, comb in items:
                        cat_inputs.append(torch.cat([maps[c] for c in comb], dim=1))  # [B, k*C, H, W]
                        conv = self.concat_convs[str(local_idx)]
                        weights.append(conv.weight)  # [C, k*C, 1, 1]
                        if conv.bias is not None:
                            biases.append(conv.bias)
                        else:
                            biases.append(torch.zeros(self.num_classes, device=device))
                    grouped_input = torch.cat(cat_inputs, dim=1)      # [B, Ck*k*C, H, W]
                    grouped_weight = torch.cat(weights, dim=0)        # [Ck*C, k*C, 1, 1]
                    grouped_bias = torch.cat(biases, dim=0)           # [Ck*C]
                    grouped_out = F.conv2d(grouped_input, grouped_weight, grouped_bias, groups=len(items))
                    outs = grouped_out.split(self.num_classes, dim=1)
                    for (local_idx, _), out in zip(items, outs):
                        produced[local_idx] = out

            elif op in ('weighted_fusion', 'wf'):
                # weight_convs[i] is SHARED across every same-size combo
                # already (one WeightedFusion module per k), so instead of
                # running the whole submodule once per combo, batch all
                # same-size combos into the conv's input batch dimension and
                # run each stage's conv exactly once.
                for k, items in by_k.items():
                    mod = self.weighted_fusion_modules[str(k)]
                    idx_tensor = torch.tensor([comb for _, comb in items], device=device, dtype=torch.long)  # [Ck, k]
                    gathered = maps[idx_tensor]  # [Ck, k, B, C, H, W]
                    Ck = gathered.shape[0]
                    weight_maps = []
                    for i in range(k):
                        stage_input = gathered[:, i].reshape(Ck * B, C, H, W)
                        wmap = mod.weight_convs[i](stage_input)          # [Ck*B, 1, H, W]
                        weight_maps.append(wmap.reshape(Ck, B, 1, H, W))
                    weights_stack = torch.stack(weight_maps, dim=1)      # [Ck, k, B, 1, H, W]
                    weights_stack = F.softmax(weights_stack, dim=1)
                    fused = (weights_stack * gathered).sum(dim=1)        # [Ck, B, C, H, W]
                    for j, (local_idx, _) in enumerate(items):
                        produced[local_idx] = fused[j]
            else:
                raise ValueError(f"Unsupported op: {op}")

            # re-assemble in original combo order for this op
            for local_idx in range(len(combos)):
                fused_logits.append(produced[local_idx])

        return fused_logits

    def forward(self, output_maps, label_batch=None, ce_loss=None, dice_loss=None):
        device = output_maps[0].device
        compute_loss = label_batch is not None and ce_loss is not None and dice_loss is not None

        maps = torch.stack(output_maps, dim=0)  # [M, B, C, H, W]
        M = maps.shape[0]

        fused_logits = self._generate_mutations(maps)

        if not compute_loss:
            # Preserve the original return contract: only the synthesized
            # combo maps (not the originals), as a plain python list.
            return fused_logits

        # Build the weight vector aligned 1:1 with [originals..., combos...]
        if self.use_learnable_weights:
            orig_vals, synth_vals = self._compute_all_weights()
            weight_list = list(orig_vals)
            for op in self.operations:
                w_list = synth_vals[op]
                weight_list.extend(w_list[idx] for idx in range(len(w_list)))
        else:
            n_total = M + len(fused_logits)
            weight_list = [torch.ones(1, device=device) for _ in range(n_total)]

        weights_tensor = torch.cat(weight_list)  # [N]

        all_logits = torch.stack(list(maps.unbind(0)) + fused_logits, dim=0)  # [N, B, C, H, W]

        # VECTORIZED loss aggregation: instead of calling ce_loss/dice_loss in
        # a Python loop once per map (N ~= 48 sequential small calls), use
        # torch.func.vmap to evaluate both loss functions across all N maps
        # as a single batched call. vmap works generically even though
        # dice_loss's internal reduction may be non-linear (it re-executes
        # the function's tensor ops with an added batch dimension, rather
        # than relying on any mathematical decomposability), so results are
        # numerically identical to calling each loss N separate times.
        def per_map_losses(logits, label):
            ce = ce_loss(logits, label.long())
            dc = dice_loss(logits, label, softmax=True)
            return ce, dc

        try:
            from torch.func import vmap
            ces, dices = vmap(per_map_losses, in_dims=(0, None))(all_logits, label_batch)
        except Exception:
            # Fallback for older torch versions or loss modules that use
            # ops incompatible with vmap's batching transform (e.g. Python
            # control flow that branches on tensor values). Still correct,
            # just not kernel-batched.
            ces_list, dices_list = [], []
            for i in range(all_logits.shape[0]):
                ce, dc = per_map_losses(all_logits[i], label_batch)
                ces_list.append(ce)
                dices_list.append(dc)
            ces = torch.stack(ces_list)
            dices = torch.stack(dices_list)

        combined_per_map = self.lc1 * ces + self.lc2 * dices  # [N]
        weighted_per_map = combined_per_map * weights_tensor  # [N]

        deep_supervision_loss = weighted_per_map[:M].sum()
        mutation_loss = weighted_per_map[M:].sum()

        if self.supervision in ['mutation', 'lomix']:
            final_loss = deep_supervision_loss + mutation_loss
        else:
            final_loss = deep_supervision_loss
        return final_loss, deep_supervision_loss, mutation_loss

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
    SINGLE DistributedDataParallel wrapper covers both sets of parameters
    (including the loss module's learnable combination weights and its
    concat_convs). DDP needs every parameter that receives gradients to be
    inside the wrapped module so it can register its gradient-sync hooks;
    keeping model and loss_module as two separately-optimized objects (as
    the DataParallel version did) would otherwise require two DDP wrappers
    or manual gradient handling.
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
    initializes the NCCL process group. Returns (rank, local_rank, world_size,
    is_distributed). Falls back to a single-process/single-GPU run if the
    script wasn't launched with torchrun (no distributed env vars present),
    so this file still works with `python train_synapse_lomix.py ...` on one
    GPU without any special casing elsewhere.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def is_main_process(rank):
    return rank == 0


@torch.no_grad()
def inference(args, model, best_performance, db_test, testloader, device, rank):
    """
    FIX: db_test / testloader are now built ONCE outside the epoch loop and
    passed in, instead of being rebuilt (and re-preloaded from disk into RAM)
    on every single call.

    DDP note: run only on rank 0. Each validation "batch" is one full 3D
    volume (batch_size=1) processed slice-by-slice inside val_single_volume,
    so splitting it across ranks would need extra gather/reduce logic for
    little benefit relative to training-time cost. Other ranks just wait at
    the barrier in trainer_synapse() while rank 0 evaluates.
    """
    if not is_main_process(rank):
        return best_performance

    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
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
    model.train()
    return performance


def trainer_synapse(args, model, snapshot_path, supervision='lomix', operations=['add', 'mul', 'wf', 'concat'],
                     n_outs=4, use_learnable_weights=True):
    rank, local_rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                             format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(f"Distributed: {is_distributed}, world_size: {world_size}")
    else:
        logging.basicConfig(level=logging.CRITICAL)  # silence non-main ranks

    base_lr = args.base_lr
    num_classes = args.num_classes
    # FIX: with DDP, args.batch_size is the PER-GPU batch size (each process
    # handles its own batch independently), unlike DataParallel where the
    # supplied batch was split across GPUs internally. So no more "* args.n_gpu"
    # here — the effective global batch size is batch_size * world_size.
    batch_size = args.batch_size

    db_train = Synapse_preloaded_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                          nclass=args.num_classes,
                                          transform=transforms.Compose(
                                              [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    if is_main_process(rank):
        print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id + rank * 1000)

    # FIX: DistributedSampler replaces shuffle=True — each rank sees a
    # disjoint shard of the dataset per epoch. sampler.set_epoch(...) below
    # is required so the shuffling differs across epochs.
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

    # Validation set / loader: only rank 0 actually needs it (see inference()),
    # but build it on all ranks cheaply guarded by rank to avoid every process
    # preloading the volumes into RAM redundantly.
    if is_main_process(rank):
        db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir,
                                             nclass=args.num_classes)
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
        # FIX: DistributedDataParallel instead of DataParallel. Each GPU runs
        # its own process (no shared-GIL orchestration bottleneck), gradients
        # are synced via an efficient NCCL all-reduce that overlaps with
        # backward(), and the model isn't re-replicated to every GPU on every
        # forward pass the way DataParallel does it. This is the main fix for
        # the near-0% GPU utilization observed with DataParallel.
        combined = DDP(combined, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False)
        raw = combined.module  # unwrapped access for checkpointing / eval
    else:
        raw = combined

    combined.train()
    optimizer = optim.AdamW(combined.parameters(), lr=base_lr, weight_decay=0.0001)

    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    writer = SummaryWriter(snapshot_path + '/log') if is_main_process(rank) else None
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    if is_main_process(rank):
        logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = range(max_epoch)

    for epoch_num in iterator:
        if is_distributed:
            train_sampler.set_epoch(epoch_num)



        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch = image_batch.to(device, non_blocking=True)
            label_batch = label_batch.squeeze(1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                loss, deep_supervision_loss, mutation_loss = combined(image_batch, label_batch, ce_loss, dice_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

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

        if is_main_process(rank):
            logging.info(
                'iteration %d, epoch %d : loss : %f, deep_supervision_loss : %f, mutation_loss : %f, lr: %f' % (
                    iter_num, epoch_num, loss.item(), deep_supervision_loss.item(), mutation_loss.item(), lr_))

            raw.loss_module.print_weights()

            save_mode_path = os.path.join(snapshot_path, 'last.pth')
            torch.save(raw.model.state_dict(), save_mode_path)

        # FIX: all ranks must reach this barrier together. Rank 0 runs
        # inference (see is_main_process check inside inference()); other
        # ranks would otherwise race ahead into the next epoch's forward
        # pass while rank 0 is still validating, which can desync NCCL
        # collectives if a later all-reduce assumes every rank is at the
        # same step.
        if is_distributed:
            dist.barrier()

        performance = inference(args, raw.model, best_performance, db_test, testloader, device, rank)

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

    if is_main_process(rank) and writer is not None:
        writer.close()

    if is_distributed:
        dist.destroy_process_group()

    return "Training Finished!"
