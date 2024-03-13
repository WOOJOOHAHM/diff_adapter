import sys
sys.path.append('./Temporal_difference/cloned_model/diff_adapter')
sys.path.append('./Temporal_difference/diff_adapter_script')

import os
import wandb
import argparse
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from util import *
from VIT import build_diff
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--model_name', type=str, default=True, help='Base Image model') # clip, st_adapter12, st_adapter24
    parser.add_argument('--dataset', type=str, help='Train dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per gpu')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs.')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--inter_type', type=str, help='# interpolation_type: 1-> just_before(JB)   2-> Differ Mean(DM)     3-> Differ Mean Interpolation(DMI)')
    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--layers', type=int, default='12')
    parser.add_argument('--substitute_frame', type=int, default=None)
    parser.add_argument('--fast_dev_run', action='store_true', default=False, help='True: Training without error')

    args = parser.parse_args()
    project_name = args.project_name
    model_name = args.model_name
    dataset = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs
    num_frames = args.num_frames
    fast_dev_run = args.fast_dev_run
    inter_type = args.inter_type
    layers = args.layers
    if args.substitute_frame == None:
        substitute_frame = num_frames
    else:
        substitute_frame = substitute_frame

    if project_name == 'compare_frames':
        save_dir = f"/home/lab/hahmwj/data/trained_model/differential_EVL/{project_name}/{dataset}/{model_name}_{inter_type}_{layers}l/{num_frames}f/"
    elif project_name == 'compare_inter_type':
        save_dir = f"/home/lab/hahmwj/data/trained_model/differential_EVL/{project_name}/{dataset}/{model_name}_{layers}l_{num_frames}f/{inter_type}/"
    elif project_name == 'compare_layers':
        save_dir = f"/home/lab/hahmwj/data/trained_model/differential_EVL/{project_name}/{dataset}/{model_name}_{inter_type}_{num_frames}f/{layers}l/"
    elif project_name == 'k400_train':
        save_dir = f"/home/lab/hahmwj/data/trained_model/differential_EVL/{project_name}/{dataset}/{model_name}_{inter_type}/{num_frames}f_{layers}l/"
    elif project_name == 'test':
        save_dir = '/home/lab/hahmwj/data/trained_model/differential_EVL/test/hmdb_sample/'

    logger = WandbLogger(project=f"{project_name}", name = f'{model_name}_{inter_type}_{num_frames}_{layers}_{dataset}', 
                         save_dir = save_dir, log_model = 'all', id = None)
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="epoch"), early_stop_callback]
    device = [0, 1, 2, 3]
    trainer = pl.Trainer(
        max_epochs=epochs,
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        devices = device,
        strategy=DDPStrategy(find_unused_parameters=True, static_graph=True),
        precision = 16)
    print(f'-----------------------------------------------------------------Creating dataset   GPU Rank: {trainer.local_rank}-----------------------------------------------------------------')
    dataloader_train, dataloader_val, dataloader_test, num_classes = load_data(dataset_name = dataset, 
            path = '/home/lab/hahmwj/data/csv_files/', 
            batch_size = batch_size,
            num_workers = 8,
            num_frames = num_frames,
            video_size=224)
    
    print(f'-----------------------------------------------------------------Creating model   GPU Rank: {trainer.local_rank}-----------------------------------------------------------------')
    differ_block = get_differ_block_layer(layers)
    model = build_diff(num_classes = num_classes, 
                       inter_type = inter_type, 
                       differ_layer = differ_block, 
                       num_frames = num_frames,
                       classification_rule = 'mean',
                       substitute_frame = 4)
    lr = args.lr * batch_size *  len(device) / 256
    model = lightening_module(model,
            num_classes = num_classes,
            warmup_steps = 3,
            weight_decay = 1e-2,
            lr = lr,
            optimizer = 'AdamW')
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
if __name__ == '__main__': main()