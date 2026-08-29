import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from lib.networks import EMCADNet
from trainer import trainer_synapse

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='./data/synapse/train_npz_new',
                    help='root dir for data. For ACDC, point this at the ACDC training slices dir '
                         '(e.g. ./data/acdc/train_npz), preprocessed into the same .npz format '
                         '(keys "image"/"label") as Synapse.')
parser.add_argument('--volume_path', type=str,
                    default='./data/synapse/test_vol_h5_new',
                    help='root dir for validation volume data. For ACDC, point this at the ACDC '
                         'validation volumes dir, preprocessed into the same .npy.h5 format as Synapse.')
parser.add_argument('--dataset', type=str,
                    default='Synapse', choices=['Synapse', 'ACDC'],
                    help='experiment_name / which dataset config to use')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse',
                    help='list dir. For ACDC, pass --list_dir ./lists/lists_ACDC (or wherever your '
                         'ACDC train/test_vol .txt split files live) — this default only applies to Synapse.')
parser.add_argument('--num_classes', type=int,
                    default=9,
                    help='output channel of network. NOTE: this is overridden below based on --dataset '
                         '(9 for Synapse, 4 for ACDC) to match each dataset\'s fixed label set.')
# network related parameters
parser.add_argument('--encoder', type=str,
                    default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
parser.add_argument('--expansion_factor', type=int,
                    default=2, help='expansion factor in MSCB block')
parser.add_argument('--kernel_sizes', type=int, nargs='+',
                    default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
parser.add_argument('--lgag_ks', type=int,
                    default=3, help='Kernel size in LGAG')
parser.add_argument('--activation_mscb', type=str,
                    default='relu6', help='activation used in MSCB: relu6 or relu')
parser.add_argument('--no_dw_parallel', action='store_true', 
                    default=False, help='use this flag to disable depth-wise parallel convolutions')
parser.add_argument('--concatenation', action='store_true', 
                    default=False, help='use this flag to concatenate feature maps in MSDC block')
parser.add_argument('--no_pretrain', action='store_true', 
                    default=False, help='use this flag to turn off loading pretrained enocder weights')
parser.add_argument('--supervision', type=str,
                    default='lomix', help='loss supervision: lomix, mutation, deep_supervision or last_layer')

parser.add_argument('--max_iterations', type=int,
                    default=50000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=6, help='batch_size PER GPU (effective global batch size = batch_size * world_size under DDP)')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--z_spacing', type=int, default=1,
                    help='slice spacing used when reconstructing volumes for validation metrics. '
                         'Defaults to 1 for both Synapse and ACDC; override if your preprocessing '
                         'pipeline uses a different physical slice spacing.')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu (kept for snapshot-path naming; actual '
                    'world size for DDP comes from torchrun, not this flag)')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')

args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    dataset_name = args.dataset
    # ACDC ADDED: dataset_config now covers both Synapse and ACDC. root_path,
    # volume_path, and list_dir all continue to come straight from the CLI
    # args (--root_path/--volume_path/--list_dir) exactly as before — nothing
    # here hardcodes dataset-specific paths, so running with --dataset ACDC
    # just requires also passing --list_dir (and --root_path/--volume_path if
    # your ACDC data lives somewhere other than the Synapse defaults above).
    # num_classes and z_spacing are the only genuinely dataset-specific
    # values: ACDC's standard label set is background + RV + myocardium + LV
    # = 4 classes (vs. Synapse's 9). z_spacing defaults to 1 for both here;
    # override via --z_spacing if your ACDC volumes were preprocessed with a
    # different physical slice spacing baked into the validation/metric code.
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': 9,
            'z_spacing': args.z_spacing,
        },
        'ACDC': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': 4,
            'z_spacing': args.z_spacing,
        },
    }
    if dataset_name not in dataset_config:
        raise ValueError(
            f"Unsupported --dataset '{dataset_name}'. Supported datasets: {list(dataset_config.keys())}. "
            f"To add another dataset, add an entry to dataset_config with the correct num_classes, and "
            f"make sure its data is preprocessed into the same .npz (train slices) / .npy.h5 (test "
            f"volumes) format the Synapse_preloaded_dataset/Synapse_dataset classes expect."
        )
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    print(args.no_pretrain)
    if args.concatenation:
        aggregation = 'concat'
    else: 
        aggregation = 'add'
    
    if args.no_dw_parallel:
        dw_mode = 'series'
    else: 
        dw_mode = 'parallel'
    
    print(aggregation)
    use_learnable_weights = False
    if args.supervision == 'lomix':
        operations = ['add', 'mul', 'wf', 'concat']
        use_learnable_weights = True
    elif args.supervision == 'mutation':
        operations =['add']
    else:
        operations = []

    if use_learnable_weights == True:
        learnable = 'learnable_'
    else:
        learnable = ''
    run = 1
    args.exp = args.encoder + '_EMCADNet_kernel_sizes_' + str(args.kernel_sizes) + '_dw_' + dw_mode + '_' + aggregation + '_lgag_ks_' + str(args.lgag_ks) + '_act_mscb_' + args.activation_mscb + '_loss_'+learnable+'relative_softp_' + args.supervision + '_'+str(operations)+'_output_last_layer_Run'+str(run)+'_' + dataset_name + str(args.img_size)#+'_nclass_'+str(args.num_classes) #add_sub_multiply_concat #_final_layer
    snapshot_path = "model_pth/{}/{}".format(args.exp, args.encoder + '_EMCADNet_kernel_sizes_' + str(args.kernel_sizes) + '_dw_' + dw_mode + '_' + aggregation + '_lgag_ks_' + str(args.lgag_ks) + '_act_mscb_' + args.activation_mscb + '_loss_'+learnable+'relative_softp_' + args.supervision + '_'+str(operations)+'_output_last_layer_Run'+str(run)) #add_sub_multiply_concat #_final_layer
    snapshot_path = snapshot_path.replace('[', '').replace(']', '').replace(', ', '_')
    
    snapshot_path = snapshot_path + '_pretrain' if not args.no_pretrain else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 50000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 300 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.0001 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    # FIX: only rank 0 needs to create the snapshot directory. Under torchrun,
    # every process runs this whole script, so guard directory creation (and
    # the "Model successfully created." print) by local rank to avoid a race
    # on os.makedirs and duplicated console spam from every GPU process.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0 and not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if local_rank == 0:
        print(snapshot_path)

    model = EMCADNet(num_classes=args.num_classes, kernel_sizes=args.kernel_sizes, expansion_factor=args.expansion_factor, dw_parallel=not args.no_dw_parallel, add=not args.concatenation, lgag_ks=args.lgag_ks, activation=args.activation_mscb, encoder=args.encoder, pretrain= not args.no_pretrain)

    # FIX: no model.cuda() here anymore. Under DDP each process needs the
    # model placed on ITS OWN local-rank GPU (cuda:LOCAL_RANK), which is only
    # known once trainer_synapse() calls setup_distributed(). Calling
    # model.cuda() here would put every process's model on cuda:0 before that
    # rank info even exists. trainer_synapse() now handles device placement
    # (and DDP wrapping) internally.

    if local_rank == 0:
        print('Model successfully created.')
    
    # ACDC ADDED: both datasets route through the same trainer_synapse
    # function. Nothing in trainer_synapse.py or the Synapse_preloaded_dataset
    # / Synapse_dataset classes is Synapse-specific beyond variable naming —
    # they're driven entirely by the root_path/volume_path/list_dir/num_classes
    # already resolved above, so ACDC works through the identical training
    # loop, DDP setup, AMP, LR scheduler, and early-stopping logic.
    trainer = {'Synapse': trainer_synapse, 'ACDC': trainer_synapse}
    trainer[dataset_name](args, model, snapshot_path, supervision=args.supervision, operations=operations, use_learnable_weights=use_learnable_weights)
