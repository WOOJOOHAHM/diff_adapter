# python /home/lab/hahmwj/differential_EVL_transformer/analyze/test_save_last_layer.py --num_frames 8 --dataset hmdb
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
    model.cuda()
    model.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            x, x1, x2, x3, x4 = model(video)
            video = video.cpu()
            x = x.cpu().to(torch.float32)
            x1 = x1.cpu().to(torch.float32)
            x2 = x2.cpu().to(torch.float32)
            x3 = x3.cpu().to(torch.float32)
            x4 = x4.cpu().to(torch.float32)
            probabilities = torch.softmax(x, dim=1)
            max_index = torch.argmax(probabilities, dim=1).item()
        # print(f'Answer: {label}  Prob: {labels[max_index]}    Confidence score: {probabilities.squeeze()[max_index] * 100 : 5f}')
        return label, labels[max_index], x4, probabilities.squeeze()[max_index] * 100
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', type=int, default=True, help='Number of frames')
    parser.add_argument('--dataset', type=str, help='Video_path')
    args = parser.parse_args()

    dataframe = pd.read_csv(f'/home/lab/hahmwj/data/csv_files/{args.dataset}.csv')
    dataframe = dataframe[dataframe['type']=='test'].reset_index()
    wrong = []
    answers = []
    preds = []
    last_layers = []
    confs = []
    ckt_pth = '/home/lab/hahmwj/data/trained_model/differential_EVL/st_adapter24_mean/hmdb/Dif Base/r551hruy/checkpoints/epoch=27-step=252.ckpt'
    model = lightening_module.load_from_checkpoint(ckt_pth)
    for i in tqdm(range(len(dataframe))):
        video_path, label = dataframe['video_path'][i], dataframe['label'][i]
        answer, pred, x4, conf = inference(video_path, model, label, dataframe, args.num_frames, 224)
        if answer != pred:
            wrong.append('Wrong')
        else:
            wrong.append('Correct')
        answers.append(answer)
        preds.append(pred)
        last_layers.append(x4)
        confs.append(conf)
    df = pd.DataFrame({
        'video_path':dataframe['video_path'],
    'wrong': wrong,
    'answers': answers,
    'preds': preds,
    'last_layers': last_layers,
    'confs': confs
})
    df.to_pickle(os.path.dirname(ckt_pth) + '/test_result.pkl')
if __name__ == '__main__': main()