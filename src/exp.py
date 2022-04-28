import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import matthews_corrcoef
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging
import os
import sys
import random
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import accuracy_score
from config import get_args
from dataset import get_dataloader
from model import Model
from inference import inference

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

args= get_args()
SEED = args.seed

model_path = os.path.join(args.output_path, 'model')
pred_path = os.path.join(args.output_path, 'result')
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(pred_path):
    os.makedirs(pred_path)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if 'roberta' in args.model and 'base' in args.model:
    backbond = RobertaModel.from_pretrained("roberta-base")

elif 'roberta' in args.model and 'large' in args.model:
    backbond = RobertaModel.from_pretrained("roberta-large")

elif 'bert' in args.model and 'base' in args.model:
    backbond = BertModel.from_pretrained("bert-base-uncased")

elif 'bert' in args.model and 'large' in args.model:
    backbond = BertModel.from_pretrained("bert-large-uncased")

model = Model(args, backbond).to(device)

if args.task == 'sts':
    loss_funtion = nn.MSELoss()
else:
    loss_funtion = nn.CrossEntropyLoss()

optimizer_weight = optim.AdamW(model.weight_lst, lr = args.lr)
optimizer_bias = optim.AdamW(model.param_lst, lr = args.lr, weight_decay=0)

if args.task == 'mnli':

    train_dataloader, val_m_dataloader, val_mm_dataloader, test_m_dataloader, test_mm_dataloader = get_dataloader(args)
    best_acc_m = 0
    best_epoch_m =0
    best_acc_mm = 0
    best_epoch_mm =0
    for epoch in range(args.mnli_epoch):

        model.train()

        for batch_id, data in enumerate(tqdm(train_dataloader)):
        
            tokens, mask, type_id, label = data
            tokens, mask, type_id, label = tokens.to(device),mask.to(device), type_id.to(device), label.to(device)
            output = model(tokens = tokens, mask = mask, type_id = type_id)

            loss = loss_funtion(output, label)
            optimizer_weight.zero_grad()
            optimizer_bias.zero_grad()
            loss.backward()
            optimizer_weight.step()
            optimizer_bias.step()

        model.eval()

        with torch.no_grad():

            correct_m = 0
            count_m = 0 
            correct_mm = 0
            count_mm = 0 

            for batch_id, data in enumerate(tqdm(val_m_dataloader)):
                tokens, mask, type_id, label = data
                tokens, mask, type_id, label = tokens.to(device),mask.to(device), type_id.to(device), label.to(device)
                output = model(tokens = tokens, mask = mask, type_id = type_id)
                output = output.view(-1,3)
                pred = torch.max(output, 1)[1]
                for j in range(len(pred)):
                    if pred[j] == label[j]:
                        correct_m += 1
                    count_m += 1

            for batch_id, data in enumerate(tqdm(val_mm_dataloader)):
                tokens, mask, type_id, label = data
                tokens, mask, type_id, label = tokens.to(device),mask.to(device), type_id.to(device), label.to(device)
                output = model(tokens = tokens, mask = mask, type_id = type_id)
                output = output.view(-1,3)
                pred = torch.max(output, 1)[1]
                for j in range(len(pred)):
                    if pred[j] == label[j]:
                        correct_mm += 1
                    count_mm += 1
                
        score_m = correct_m/count_m
        score_mm = correct_mm/count_mm
        print('Epoch = ', epoch + 1)
        print('Val_m = ', score_m)
        print('Val_mm = ', score_mm)
        print('-----------------------------------------')

        if score_m >= best_acc_m:
            best_acc_m = score_m
            best_epoch_m = epoch + 1
            torch.save(model.state_dict(), os.path.join(model_path, 'mnli_m.ckpt'))
        if score_mm >= best_acc_mm:
            best_acc_mm = score_mm
            best_epoch_mm = epoch + 1
            torch.save(model.state_dict(), os.path.join(model_path, 'mnli_mm.ckpt'))
        
        print('Best Epoch_m = ', best_epoch_m)
        print('Best Performance_m = ', best_acc_m)
        print('Best Epoch_mm = ', best_epoch_mm)
        print('Best Performance_mm = ', best_acc_mm)
        print('=========================================')
    
    print('Start Inference!!!')
    ckpt_m = torch.load(os.path.join(model_path, 'mnli_m.ckpt'))
    ckpt_mm = torch.load(os.path.join(model_path, 'mnli_mm.ckpt'))
    model_m.load_state_dict(ckpt_m)
    model_mm.load_state_dict(ckpt_mm)
    model_m.eval()
    model_mm.eval()

    inference(args, model_m, test_m_dataloader, 'mnli_m')
    inference(args, model_mm, test_mm_dataloader, 'mnli_mm')
        


else:

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(args)
    best_acc = 0
    best_epoch=0
    for epoch in range(args.epoch): 

        model.train()

        for batch_id, data in enumerate(tqdm(train_dataloader)):

            tokens, mask, type_id, label = data
            tokens, mask, type_id, label = tokens.to(device),mask.to(device), type_id.to(device), label.to(device)
            output = model(tokens = tokens, mask = mask, type_id = type_id)

            loss = loss_funtion(output, label)
            optimizer_weight.zero_grad()
            optimizer_bias.zero_grad()
            loss.backward()
            optimizer_weight.step()
            optimizer_bias.step()
        
        model.eval()
        with torch.no_grad():
            correct = 0
            count = 0
            my_ans = []
            real_ans = []
            for batch_id, data in enumerate(tqdm(val_dataloader)):
                tokens, mask, type_id, label = data
                tokens, mask, type_id, label = tokens.to(device),mask.to(device), type_id.to(device), label.to(device)
                output = model(tokens = tokens, mask = mask, type_id = type_id)

                if args.task == 'sts':
                    output = output.view(-1)
                    for j in range(len(output)):
                        my_ans.append(int(output[j]))
                        real_ans.append(int(label[j]))

                else:
                    output = output.view(-1,2)
                    pred = torch.max(output, 1)[1]

                    for j in range(len(pred)):
                        if args.task == 'cola':
                            if label[j] == 0:
                                label[j] = -1
                            if pred[j] == 0:
                                pred[j] = -1
                        my_ans.append(int(pred[j]))
                        real_ans.append(int(label[j]))

        print('Epoch = ', epoch + 1)
        if args.task == 'cola': #report correlation
            score = matthews_corrcoef(real_ans, my_ans)
            print('Val Score = ', score)
        
        elif args.task == 'sts': #report correlation
            score_pear = stats.pearsonr(real_ans, my_ans)
            score = stats.spearmanr(real_ans, my_ans)
            print('Val spear = ', score)
            print('Val pear = ', score_pear)
        else: #report acc
            score = accuracy_score(real_ans, my_ans)
            print('Val acc = ', score)
        print('-----------------------------------------')
        
        if score >= best_acc:
            best_acc = score
            best_epoch = epoch + 1
            model_name = str(args.task) + '.ckpt'
            save_path = os.path.join(model_path, model_name)
            torch.save(model.state_dict(), save_path)

        print('Best Epoch = ', best_epoch)
        print('Best Performance = ', best_acc)
        print('=========================================')

    print('Start Inference!!!')
    ckpt = torch.load(save_path)
    model.load_state_dict(ckpt)
    model.eval()

    inference(args, model, test_dataloader)
