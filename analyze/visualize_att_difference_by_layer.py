# python /home/lab/hahmwj/differential_EVL_transformer/analyze/visualize_att_difference_by_layer.py --model_name clip --num_frames 8 --classifier mean --dataset hmdb --video_path Free_Hugs_-_Paris_www_calins-gratuits_com_hug_f_cm_np4_ba_med_2.avi
# python /home/lab/hahmwj/differential_EVL_transformer/analyze/visualize_att_difference_by_layer.py --model_name clip --num_frames 8 --classifier mean --dataset hmdb --save_type True --video_path Free_Hugs_-_Paris_www_calins-gratuits_com_hug_f_cm_np4_ba_med_2.avi
import sys
import os
sys.path.append('/home/lab/hahmwj/data/cloned_model/CLIP/')
sys.path.append('/home/lab/hahmwj/data/cloned_model/st_adapter')
sys.path.append('/home/lab/hahmwj/differential_EVL_transformer/')

import math
import argparse
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.functional import accuracy, auroc, confusion_matrix, f1_score
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.io import read_video

from util import *
from dataset import *
import clip
from st_adapter import *

def inference(video_path, model, label, dataframe, num_frames, video_size):
    labels = sorted(set(dataframe['label']))
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    test_transform = T.Compose(
        [
            ToTensorVideo(),
            T.Resize(size=(video_size, video_size)),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    video, audio, info = read_video(video_path, pts_unit="sec")
    video = extract_frames(video, num_frames)
    video = test_transform(video)
    video = video.unsqueeze(dim = 0).cuda()
    model.eval()
    with torch.no_grad():
        x, x1, x2, x3, x4 = model(video)
        video = video.cpu()
        x = x.cpu()
        x1 = x1.cpu()
        x2 = x2.cpu()
        x3 = x3.cpu()
        x4 = x4.cpu()
        probabilities = torch.softmax(x, dim=1)
        max_index = torch.argmax(probabilities, dim=1).item()
        print(f'Answer: {label}  Prob: {labels[max_index]}    Confidence score: {probabilities.squeeze()[max_index] * 100 : 5f}')
        return video.squeeze().permute(1, 0, 2, 3), x1, x2, x3, x4
    
def get_difference(tensor, type):
    if type == 'video':
        difference = tensor[1:, :, :] - tensor[:-1, :, :]
        difference = difference.pow(2)
        difference = difference.mean(dim = 1)
        difference = difference.view(difference.size(0), -1)
    else:
        difference = tensor[1:, :, :] - tensor[:-1, :, :]
        difference = difference.pow(2)
        difference = difference.mean(dim = 2)
    difference = torch.softmax(difference, dim=1)
    video_size = int(math.sqrt(difference.size(1)))
    difference = difference.view(difference.size(0), video_size, video_size)
    return difference

def draw_difference_attention_plot(video_diff, x1, x2, x3, x4, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + 'total.png'
    fig, axs = plt.subplots(5, 7, figsize=(63, 45))
    for i in tqdm(range(video_diff.size(0))):
        sns.heatmap(video_diff[i].detach().cpu().numpy(), cmap="coolwarm", annot=True, ax=axs[0, i])
        sns.heatmap(x1[i].detach().cpu().numpy(), cmap="coolwarm", annot=True, ax=axs[1, i])
        sns.heatmap(x2[i].detach().cpu().numpy(), cmap="coolwarm", annot=True, ax=axs[2, i])
        sns.heatmap(x3[i].detach().cpu().numpy(), cmap="coolwarm", annot=True, ax=axs[3, i])
        sns.heatmap(x4[i].detach().cpu().numpy(), cmap="coolwarm", annot=True, ax=axs[4, i])
    
    plt.tight_layout()
    plt.savefig(save_path)

def save_difference_seperate(video_diff, x1, x2, x3, x4, save_path):
    save_path = save_path + 'seperate/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_heads = video_diff.size(0)
    for head in tqdm(range(num_heads)):
        plt.figure(figsize=(24, 24))
        sns.heatmap(video_diff[head].detach().cpu().numpy(), cmap="coolwarm", annot=True)
        plt.title(f"Difference Heatmap {head + 1} - {head}")
        plt.savefig(save_path + f'video_diff_{head + 1}_{head}.png')
        plt.close()
        
    for head in tqdm(range(num_heads)):
        plt.figure(figsize=(24, 24))
        sns.heatmap(x1[head].detach().cpu().numpy(), cmap="coolwarm", annot=True)
        plt.title(f"Difference Heatmap {head + 1} - {head}")
        plt.savefig(save_path + f'x1_{head + 1}_{head}.png')
        plt.close()

    for head in tqdm(range(num_heads)):
        plt.figure(figsize=(24, 24))
        sns.heatmap(x2[head].detach().cpu().numpy(), cmap="coolwarm", annot=True)
        plt.title(f"Difference Heatmap {head + 1} - {head}")
        plt.savefig(save_path + f'x2_{head + 1}_{head}.png')
        plt.close()

    for head in tqdm(range(num_heads)):
        plt.figure(figsize=(24, 24))
        sns.heatmap(x3[head].detach().cpu().numpy(), cmap="coolwarm", annot=True)
        plt.title(f"Difference Heatmap {head + 1} - {head}")
        plt.savefig(save_path + f'x3_{head + 1}_{head}.png')
        plt.close()

    for head in tqdm(range(num_heads)):
        plt.figure(figsize=(24, 24))
        sns.heatmap(x4[head].detach().cpu().numpy(), cmap="coolwarm", annot=True)
        plt.title(f"Difference Heatmap {head + 1} - {head}")
        plt.savefig(save_path + f'x4_{head + 1}_{head}.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Base Image model') # R50, R101, ViT-B/16, ViT-L/14
    parser.add_argument('--num_frames', type=int, default=True, help='Number of frames')
    parser.add_argument('--classifier', type=str, default=True, help='Aggregation classifiers')
    parser.add_argument('--dataset', type=str, help='Video_path')
    parser.add_argument('--video_path', type=str, help='Video_path')
    args = parser.parse_args()

    model_name = args.model_name
    dataset = args.dataset
    num_frames = args.num_frames
    classifier = args.classifier
    video_path = args.video_path

    dataframe = pd.read_csv(f'/home/lab/hahmwj/data/csv_files/{dataset}.csv')
    video_path = dataframe.loc[dataframe['video_path'].str.contains(video_path), 'video_path'].values[0]
    label = dataframe.loc[dataframe['video_path'].str.contains(video_path), 'label'].values[0]
    model = lightening_module.load_from_checkpoint('/home/lab/hahmwj/data/trained_model/differential_EVL/clip_mean/hmdb/Dif_Base/z30c8nxs/checkpoints/epoch=99-step=900.ckpt', visualize = True)
    video, x1, x2, x3, x4 = inference(video_path, model, label, dataframe, num_frames, 224)
    print(x1.size())
    x1_attn = x1_attn[:, 1:, :].size()
    x2_attn = x2_attn[:, 1:, :].size()
    x3_attn = x3_attn[:, 1:, :].size()
    x4_attn = x4_attn[:, 1:, :].size()

    x1_cls = x1_cls[:, 1:, :].size()
    x2_cls = x2_cls[:, 1:, :].size()
    x3_cls = x3_cls[:, 1:, :].size()
    x4_cls = x4_cls[:, 1:, :].size()
    print(x1[:, 1, :].size())
    print(asd)
    video_difference = get_difference(video, 'video')
    x1_difference = get_difference(x1, 'tensor')
    x2_difference = get_difference(x2, 'tensor')
    x3_difference = get_difference(x3, 'tensor')
    x4_difference = get_difference(x4, 'tensor')

    save_path = f'/home/lab/hahmwj/data/analyze/{model_name}_{classifier}/'
    
    save_difference_seperate(video_difference, x1, x2, x3, x4, save_path)
    draw_difference_attention_plot(video_difference, x1_difference, x2_difference, x3_difference, x4_difference, save_path)
if __name__ == '__main__': main()