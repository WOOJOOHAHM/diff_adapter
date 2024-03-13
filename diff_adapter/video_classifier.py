import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_model import *

class mean_classifier(nn.Module):
    def __init__(self, width, num_classes):
        super(mean_classifier, self).__init__()
        self.dropout = nn.Dropout(0.5)

        
        self.ln_post = LayerNorm(width)
        self.fc = nn.Linear(width, num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x):
        # 주어진 입력 x를 그대로 반환합니다.
        x = x[:, :, :, 0, :]
        x = x.mean(dim=(2))
        
        x = self.ln_post(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class concat_cls_classifeir(nn.Module):
    def __init__(self, width, num_classes, clip):
        super(concat_cls_classifeir, self).__init__()
        self.dropout = nn.Dropout(0.5)

        
        self.ln_post = LayerNorm(width*clip)
        self.fc = nn.Linear(width*clip, num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x):
        # 주어진 입력 x를 그대로 반환합니다.
        x = x[:, :, :, 0, :]
        x = x.mean(dim=(2)).view(1, -1)

        x = self.ln_post(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class concat_fcls_mean_diff_classifeir(nn.Module):
    def __init__(self, width, num_classes):
        super(concat_fcls_mean_diff_classifeir, self).__init__()
        self.dropout = nn.Dropout(0.5)

        
        self.ln_post = LayerNorm(width*2)
        self.fc = nn.Linear(width*2, num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x):
        # 주어진 입력 x를 그대로 반환합니다.
        x = x.flatten(1, 2)[:, :, 0, :]
        mean_diff = torch.mean(x[:, 1:, :] - x[:, :-1, :], dim = 1)
        x = x[:, 0, :]
        x = torch.cat((x, mean_diff), dim=1)
        x = self.ln_post(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
# RNN모델은, CLip과 안에 있는 frame을 다시 곱해서 원래의 frame으로 변경한 다음에 진행한다.
class rnn_classifier(nn.Module):
    def __init__(self, width, num_classes, rnn_model):
        super(rnn_classifier, self).__init__()
        if rnn_model == 'RNN':
            self.RNN = nn.RNN(input_size = width,
                            hidden_size = 512,
                            num_layers = 2,
                            nonlinearity = 'tanh',
                            bias = True,
                            bidirectional = False,
                            dropout = 0.3,
                            batch_first = True)
        elif rnn_model == 'LSTM':
            self.RNN = nn.LSTM(input_size = width,
                            hidden_size = 512,
                            num_layers = 2,
                            bias = True,
                            bidirectional = False,
                            dropout = 0.3,
                            batch_first = True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # 주어진 입력 x를 그대로 반환합니다.
        x = x[:, :, :, 0, :].view(x.size(0), -1, x.size(-1))
        self.RNN.flatten_parameters()
        _, (hidden, _) = self.RNN(x)
        x = hidden[-1]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# class multi_layer_rnn_blocks(nn.Module):
#     def __init__(self, width, num_classes, rnn_model):
#         super(rnn_classifier, self).__init__()

class classifier(nn.Module):
    def __init__(self, classification_rule, width, num_classes, clip):
        super(classifier, self).__init__()
        if classification_rule == 'mean':
            self.classifier = mean_classifier(width, num_classes)
        elif classification_rule == 'concat_cls':
            self.classifier = concat_cls_classifeir(width, num_classes, clip)
        elif classification_rule == 'concat_fcls_mean_diff':
            self.classifier = concat_fcls_mean_diff_classifeir(width, num_classes)
        elif classification_rule == 'RNN':
            self.classifier = rnn_classifier(width, num_classes, classification_rule)
        elif classification_rule == 'LSTM':
            self.classifier = rnn_classifier(width, num_classes, classification_rule)

    def forward(self, x, cls_tokens):
        # 주어진 입력 x를 그대로 반환합니다.
        print('cls_tokens:  ', cls_tokens.size())
        x = self.classifier(x)
        return x