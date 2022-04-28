from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args, backbond):
        super(Model, self).__init__()
        self.backbond = backbond
        self.args = args
        self.weight_lst= []
        self.param_lst = []

        for name,param in self.backbond.named_parameters(): 
            if 'LayerNorm' in name and 'attention' not in name:
                self.param_lst.append(param)
                continue
            elif 'adapter' in name:
                if 'bias' in name:
                    self.param_lst.append(param)                
                else:
                    self.weight_lst.append(param)
                continue
            else:
                param.requires_grad = False

        if 'base' in args.model:
            self.hidden_size = 768
        elif 'large' in args.model:
            self.hidden_size = 1024

        if args.task == 'mnli':
            self.outputclass = 3
        elif args.task == 'sts':
            self.outputclass = 1
        else:
            self.outputclass = 2

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.hidden_size, self.outputclass),
        )

        for name,param in self.fc.named_parameters(): 
            self.weight_lst.append(param)

    def forward(self, tokens, mask, type_id):
        embedding = self.backbond(input_ids = tokens, attention_mask = mask, token_type_ids = type_id)[1]
        answer = self.fc(embedding)
        return answer
