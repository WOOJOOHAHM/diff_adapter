# python /home/lab/hahmwj/differential_EVL_transformer/script/test.py --model_name st_24 --project compare_frames --dataset hmdb --num_frames 32 --layers 4
import sys
sys.path.append('/home/lab/hahmwj/data/cloned_model/CLIP/')
sys.path.append('/home/lab/hahmwj/data/cloned_model/st_adapter')
sys.path.append('/home/lab/hahmwj/data/cloned_model/')
sys.path.append('/home/lab/hahmwj/diff_adapter_script/')

import os
import wandb
import argparse
import seaborn as sns
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from torchmetrics.functional import accuracy, auroc, confusion_matrix, f1_score
from pytorch_lightning.callbacks import ModelCheckpoint

from util import *
import clip
from st_adapter import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Base Image model') # R50, R101, ViT-B/16, ViT-L/14
    parser.add_argument('--project', type=str, help='Base project name') # R50, R101, ViT-B/16, ViT-L/14
    parser.add_argument('--dataset', type=str, help='Train dataset')
    parser.add_argument('--num_frames', type=int, default=True, help='Number of frames')
    parser.add_argument('--layers', type=str, default=None, help='Aggregation classifiers')


    args = parser.parse_args()
    model_name = args.model_name
    dataset = args.dataset
    num_frames = args.num_frames
    layers = args.layers
    project = args.project
    labels = set(pd.read_csv(f'/home/lab/hahmwj/data/csv_files/{dataset}.csv')['label'])

    dataloader_train, dataloader_val, dataloader_test, num_classes = load_data(dataset_name = dataset, 
            path = '/home/lab/hahmwj/data/csv_files/', 
            num_workers = 16,
            num_frames = num_frames,
            video_size=224)

    model = lightening_module.load_from_checkpoint('/home/lab/hahmwj/data/trained_model/differential_EVL/trained_base_model/hmdb/st_adapter24/trained_base_model/qt9rux1p/checkpoints/epoch=19-step=680.ckpt')

    if 'differ' not in model_name:
        save_root_dir=f"/home/lab/hahmwj/data/trained_model/differential_EVL/test_result/{dataset}/{project}/{model_name}_{num_frames}/"
    else:
        save_root_dir=f"/home/lab/hahmwj/data/trained_model/differential_EVL/test_result/{dataset}/{project}/{model_name}_{num_frames}_{layers}/"

    trainer = pl.Trainer(accelerator="auto", 
                    default_root_dir=save_root_dir,
                    precision = 16,
                    devices = [0])
        
    predictions = trainer.predict(model, dataloaders=dataloader_test)

    y = torch.cat([item["y"] for item in predictions])
    y_pred = torch.cat([item["y_pred"] for item in predictions])
    y_prob = torch.cat([item["y_prob"] for item in predictions])
    
    accuracy1 = accuracy(y_prob, y, task="multiclass", num_classes=num_classes)
    accuracy5 = accuracy(y_prob, y, task="multiclass", num_classes=num_classes, top_k=5)
    _AUROC = auroc(y_prob, y, task="multiclass", num_classes=num_classes)
    _F1_SCORE = f1_score(y_prob, y, task="multiclass", num_classes=num_classes)
    print("accuracy:", accuracy1)
    print("accuracy_top5:", accuracy5)
    print("auroc:", _AUROC)
    print("f1_score:", _F1_SCORE)

    cm = confusion_matrix(y_pred, y, task="multiclass", num_classes=num_classes)

    plt.figure(figsize=(20, 20), dpi=100)
    ax = sns.heatmap(cm, annot=False, fmt="d", xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    numbers = [accuracy1.item(), accuracy5.item(), _AUROC.item(), _F1_SCORE.item()]

    # 리스트를 DataFrame으로 변환
    result = pd.DataFrame([numbers], columns=["accuracy", "accuracy_top5", "auroc", "f1_score"])

    # CSV 파일로 저장
    result.to_csv(f"{save_root_dir}results.csv", index=False)
    plt.savefig(f"{save_root_dir}output.png", dpi=300)
if __name__ == "__main__":
    main()