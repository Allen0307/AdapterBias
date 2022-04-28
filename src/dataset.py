from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel


#==============================CoLA==============================

class cola_dataset(Dataset): 
    def __init__(self, mode, args, tokenizer):

        self.mode = mode
        self.args = args
        self.tokenizer = tokenizer
        if self.mode == 'train':
            data_path = os.path.join(args.GLUE_path, 'CoLA/train.tsv')
            self.df = pd.read_csv(data_path, sep='\t')
            self.df.columns = [1,'label',3,'sen']

        elif self.mode == 'val':
            data_path = os.path.join(args.GLUE_path, 'CoLA/dev.tsv')
            self.df = pd.read_csv(data_path, sep='\t')
            self.df.columns = [1,'label',3,'sen']

        else:
            data_path = os.path.join(args.GLUE_path, 'CoLA/test.tsv')
            self.df = pd.read_csv(data_path, sep='\t')
            self.df.columns = ['id', 'sen']

        self.len = len(self.df)
        
    def __getitem__(self, index):

        encoded = self.tokenizer.encode_plus(
            text = self.df['sen'][index],  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.args.max_len,  # maximum length of a sentence
            padding = 'max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_token_type_ids = True,
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )

        if self.mode == 'test':
            label = 0
        else:
            label = self.df['label'][index]

        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        
        return input_ids.view(self.args.max_len), attn_mask.view(self.args.max_len), token_type_ids.view(self.args.max_len), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)

#==============================MNLI==============================

class mnli_dataset(Dataset): 
    def __init__(self, mode, args, tokenizer):

        self.mode = mode
        self.args = args
        self.tokenizer = tokenizer
        if self.mode == 'train':
            train_path = os.path.join(args.GLUE_path, 'MNLI/train.tsv')
            self.df = pd.read_csv(train_path, sep='\t',error_bad_lines=False, keep_default_na = False)

        elif self.mode == 'val_m':
            val_match_path = os.path.join(args.GLUE_path, 'MNLI/dev_matched.tsv')
            self.df = pd.read_csv(val_match_path, sep='\t',error_bad_lines=False, keep_default_na = False)

        elif self.mode == 'val_mm':
            val_mismatch_path =os.path.join(args.GLUE_path, 'MNLI/dev_mismatched.tsv')
            self.df = pd.read_csv(val_mismatch_path, sep='\t',error_bad_lines=False, keep_default_na = False)

        elif self.mode == 'test_m':
            test_match_path = os.path.join(args.GLUE_path, 'MNLI/test_matched.tsv')
            self.df = pd.read_csv(test_match_path, sep='\t',error_bad_lines=False, keep_default_na = False)

        else:
            test_mismatch_path = os.path.join(args.GLUE_path,'MNLI/test_mismatched.tsv')
            self.df = pd.read_csv(test_mismatch_path, sep='\t',error_bad_lines=False, keep_default_na = False)

        self.len = len(self.df)

    def __getitem__(self, index):

        encoded = self.tokenizer.encode_plus(
            self.df['sentence1'][index],  # the sentence to be encoded
            self.df['sentence2'][index],
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.args.max_len,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_token_type_ids = True,
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )
        if self.mode == 'test-m' or self.mode == 'test.mm' or 'test' in self.mode:
            label = 0
        
        else:

            if self.df['gold_label'][index] == 'entailment':
                label = 2
            elif self.df['gold_label'][index] == 'neutral':
                label = 1
            else:
                label = 0
        
        input_ids = encoded['input_ids'][0][:self.args.max_len]
        attn_mask = encoded['attention_mask'][0][:self.args.max_len]
        token_type_ids = encoded['token_type_ids'][0][:self.args.max_len]
        return input_ids.view(self.args.max_len), attn_mask.view(self.args.max_len), token_type_ids.view(self.args.max_len), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)

#==============================MRPC==============================

class mrpc_dataset(Dataset): 
    def __init__(self, mode, args, tokenizer):

        self.mode = mode
        self.args = args
        self.tokenizer = tokenizer
        if self.mode == 'train':
            train_path = os.path.join(args.GLUE_path,'MRPC/train.tsv')
            self.df = pd.read_csv(train_path, sep='\t',error_bad_lines=False)
            self.df.columns = ['label','ID1','ID2','sen1','sen2']
            self.df.dropna()

        elif self.mode == 'val':
            val_path = os.path.join(args.GLUE_path,'MRPC/dev.tsv')
            self.df = pd.read_csv(val_path, sep='\t',error_bad_lines=False)
            self.df.columns = ['label','ID1','ID2','sen1','sen2']
            self.df.dropna()

        else:
            test_path = os.path.join(args.GLUE_path,'MRPC/test.tsv')
            self.df = pd.read_csv(test_path, sep='\t',error_bad_lines=False)
            self.df.columns = ['id', 'ID1', 'ID2', 'sen1', 'sen2']
            self.df.dropna()

        self.len = len(self.df)

    def __getitem__(self, index):

        encoded = self.tokenizer.encode_plus(
            self.df['sen1'][index],  # the sentence to be encoded
            self.df['sen2'][index],
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.args.max_len,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_token_type_ids = True,
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )

        if self.mode == 'test':
            label = 0
        else:
            label = self.df['label'][index]

              
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        return input_ids.view(self.args.max_len), attn_mask.view(self.args.max_len), token_type_ids.view(self.args.max_len), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)

#==============================QNLI==============================

class qnli_dataset(Dataset): 
    def __init__(self, mode, args, tokenizer):

        self.mode = mode
        self.args = args
        self.tokenizer = tokenizer
        if self.mode == 'train':
            train_path = os.path.join(args.GLUE_path,'QNLI/train.tsv')
            self.df = pd.read_csv(train_path, sep='\t',error_bad_lines=False)
            self.df.columns = [1,'sen1','sen2','label']
            self.df.dropna()

        elif self.mode == 'val':
            val_path = os.path.join(args.GLUE_path,'QNLI/dev.tsv')
            self.df = pd.read_csv(val_path, sep='\t',error_bad_lines=False)
            self.df.columns = [1,'sen1','sen2','label']
            self.df.dropna()

        else:
            test_path = os.path.join(args.GLUE_path,'QNLI/test.tsv')
            self.df = pd.read_csv(test_path, sep='\t',error_bad_lines=False)
            self.df.columns = ['id', 'sen1', 'sen2']
            self.df.dropna()

        self.len = len(self.df)

    def __getitem__(self, index):

        encoded = self.tokenizer.encode_plus(
            self.df['sen1'][index],  # the sentence to be encoded
            self.df['sen2'][index],
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.args.max_len,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_token_type_ids = True,
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )

        if self.mode == 'test':
            label = 0
        else:
            if self.df['label'][index] == 'entailment':
                label = 1
            else:
                label = 0

              
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        return input_ids.view(self.args.max_len), attn_mask.view(self.args.max_len), token_type_ids.view(self.args.max_len), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)

#==============================QQP==============================

class qqp_dataset(Dataset): 
    def __init__(self, mode, args, tokenizer):

        self.mode = mode
        self.args = args
        self.tokenizer = tokenizer
        if self.mode == 'train':
            train_path = os.path.join(args.GLUE_path,'QQP/train.tsv')
            self.df = pd.read_csv(train_path, sep='\t')
            self.df.columns = [1,2,3,'sen1','sen2','label']

        elif self.mode == 'val':
            val_path = os.path.join(args.GLUE_path,'QQP/dev.tsv')
            self.df = pd.read_csv(val_path, sep='\t')
            self.df.columns = [1,2,3,'sen1','sen2','label']

        else:
            test_path = os.path.join(args.GLUE_path,'QQP/test.tsv')
            self.df = pd.read_csv(test_path, sep='\t')
            self.df.columns = ['id', 'sen1', 'sen2']

        self.len = len(self.df)

    def __getitem__(self, index):

        encoded = self.tokenizer.encode_plus(
            self.df['sen1'][index],  # the sentence to be encoded
            self.df['sen2'][index],
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.args.max_len,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_token_type_ids = True,
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )

        if self.mode == 'test':
            label = 0
        else:
            label = self.df['label'][index]  

              
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        return input_ids.view(self.args.max_len), attn_mask.view(self.args.max_len), token_type_ids.view(self.args.max_len), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)

#==============================RTE==============================

class rte_dataset(Dataset): 
    def __init__(self, mode, args, tokenizer):

        self.mode = mode
        self.args = args
        self.tokenizer = tokenizer
        if self.mode == 'train':
            train_path = os.path.join(args.GLUE_path,'RTE/train.tsv')
            self.df = pd.read_csv(train_path, sep='\t',error_bad_lines=False)
            self.df.columns = [1,'sen1','sen2','label']

        elif self.mode == 'val':
            val_path = os.path.join(args.GLUE_path,'RTE/dev.tsv')
            self.df = pd.read_csv(val_path, sep='\t',error_bad_lines=False)
            self.df.columns = [1,'sen1','sen2','label']

        else:
            test_path = os.path.join(args.GLUE_path,'RTE/test.tsv')
            self.df = pd.read_csv(test_path, sep='\t')
            self.df.columns = ['id', 'sen1', 'sen2']

        self.len = len(self.df)

    def __getitem__(self, index):

        encoded = self.tokenizer.encode_plus(
            self.df['sen1'][index],  # the sentence to be encoded
            self.df['sen2'][index],
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.args.max_len,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_token_type_ids = True,
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )

        if self.mode == 'test':
            label = 0
        else:
            if self.df['label'][index] == 'entailment':
                label = 1
            else:
                label = 0
              
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        return input_ids.view(self.args.max_len), attn_mask.view(self.args.max_len), token_type_ids.view(self.args.max_len), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)

#==============================SST-2==============================

class sst_dataset(Dataset): 
    def __init__(self, mode, args, tokenizer):

        self.mode = mode
        self.args = args
        self.tokenizer = tokenizer
        if self.mode == 'train':
            data_path = os.path.join(args.GLUE_path, 'SST-2/train.tsv')
            self.df = pd.read_csv(data_path, sep='\t')
            self.df.columns = ['sen','label']

        elif self.mode == 'val':
            data_path = os.path.join(args.GLUE_path, 'SST-2/dev.tsv')
            self.df = pd.read_csv(data_path, sep='\t')
            self.df.columns = ['sen','label']

        else:
            data_path = os.path.join(args.GLUE_path, 'SST-2/test.tsv')
            self.df = pd.read_csv(data_path, sep='\t')
            self.df.columns = ['id', 'sen']

        self.len = len(self.df)
        
    def __getitem__(self, index):

        encoded = self.tokenizer.encode_plus(
            text = self.df['sen'][index],  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.args.max_len,  # maximum length of a sentence
            padding = 'max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_token_type_ids = True,
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )

        if self.mode == 'test':
            label = 0
        else:
            label = self.df['label'][index]

        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        
        return input_ids.view(self.args.max_len), attn_mask.view(self.args.max_len), token_type_ids.view(self.args.max_len), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)

#==============================STS-B==============================

class sts_dataset(Dataset): 
    def __init__(self, mode, args, tokenizer):

        self.mode = mode
        self.args = args
        self.tokenizer = tokenizer
        if self.mode == 'train':
            train_path = os.path.join(args.GLUE_path,'STS-B/train.tsv')
            self.df = pd.read_csv(train_path, sep='\t',error_bad_lines=False)
            self.df.dropna()

        elif self.mode == 'val':
            val_path = os.path.join(args.GLUE_path,'STS-B/dev.tsv')
            self.df = pd.read_csv(val_path, sep='\t',error_bad_lines=False)
            self.df.dropna()

        else:
            test_path = os.path.join(args.GLUE_path,'STS-B/test.tsv')
            self.df = pd.read_csv(test_path, sep='\t',error_bad_lines=False)
            self.df.dropna()

        self.len = len(self.df)

    def __getitem__(self, index):

        encoded = self.tokenizer.encode_plus(
            self.df['sentence1'][index],  # the sentence to be encoded
            self.df['sentence2'][index],
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.args.max_len,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_token_type_ids = True,
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )

        if self.mode == 'test':
            label = 0
        else:
            if str(self.df['score'][index]) == 'nan':
                label = 0
            else:
                label = int(self.df['score'][index])
              
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        return input_ids.view(self.args.max_len), attn_mask.view(self.args.max_len), token_type_ids.view(self.args.max_len), torch.tensor(label, dtype=torch.long)

    def __len__(self):

        return(self.len)


def get_dataloader(args):
    
    if 'roberta' in args.model and 'base' in args.model:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    elif 'roberta' in args.model and 'large' in args.model:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    elif 'bert' in args.model and 'base' in args.model:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    elif 'bert' in args.model and 'large' in args.model:
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

    if args.task == 'cola':

        train_dataset = cola_dataset('train', args, tokenizer)
        val_dataset = cola_dataset('val', args, tokenizer)
        test_dataset = cola_dataset('test', args, tokenizer)

    elif args.task == 'mnli':

        train_dataset = mnli_dataset('train', args, tokenizer)
        val_m_dataset = mnli_dataset('val_m', args, tokenizer)
        val_mm_dataset = mnli_dataset('val_mm', args, tokenizer)
        test_m_dataset = mnli_dataset('test_m', args, tokenizer)
        test_mm_dataset = mnli_dataset('test_mm', args, tokenizer)
    
    elif args.task == 'mrpc':

        train_dataset = mrpc_dataset('train', args, tokenizer)
        val_dataset = mrpc_dataset('val', args, tokenizer)
        test_dataset = mrpc_dataset('test', args, tokenizer)

    elif args.task == 'qnli':

        train_dataset = qnli_dataset('train', args, tokenizer)
        val_dataset = qnli_dataset('val', args, tokenizer)
        test_dataset = qnli_dataset('test', args, tokenizer)
    
    elif args.task == 'qqp':

        train_dataset = qqp_dataset('train', args, tokenizer)
        val_dataset = qqp_dataset('val', args, tokenizer)
        test_dataset = qqp_dataset('test', args, tokenizer)

    elif args.task == 'rte':

        train_dataset = rte_dataset('train', args, tokenizer)
        val_dataset = rte_dataset('val', args, tokenizer)
        test_dataset = rte_dataset('test', args, tokenizer)

    elif args.task == 'sst':

        train_dataset = sst_dataset('train', args, tokenizer)
        val_dataset = sst_dataset('val', args, tokenizer)
        test_dataset = sst_dataset('test', args, tokenizer)

    elif args.task == 'sts':

        train_dataset = sts_dataset('train', args, tokenizer)
        val_dataset = sts_dataset('val', args, tokenizer)
        test_dataset = sts_dataset('test', args, tokenizer)
    
    #create dataloader
    
    if args.task == 'mnli':
        train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
        val_m_dataloader = DataLoader(val_m_dataset, batch_size = args.batch_size)
        val_mm_dataloader = DataLoader(val_mm_dataset, batch_size = args.batch_size)
        test_m_dataloader = DataLoader(test_m_dataset, batch_size = args.batch_size)
        test_mm_dataloader = DataLoader(test_mm_dataset, batch_size = args.batch_size)

        return train_dataloader, val_m_dataloader, val_mm_dataloader, test_m_dataloader, test_mm_dataloader
        
    else:

        train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size)

        return train_dataloader, val_dataloader, test_dataloader