import os
import csv
import torch
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def inference(args, model, dataloader, type_for_mnli = None):
    ans = []

    if args.task == 'mnli':
        task_name = str(type_for_mnli) + '.tsv'
    else:
        task_name = str(args.task) + '.tsv'
    pred_path = os.path.join(os.path.join(args.output_path, 'result'), task_name)

    if args.task != 'mnli':

        with torch.no_grad():
            for batch_id, data in enumerate(tqdm(dataloader)):
                tokens, mask, type_id, _ = data
                tokens, mask, type_id = tokens.to(device),mask.to(device), type_id.to(device)
                output = model(tokens = tokens, mask = mask, type_id = type_id)

                if args.task == 'sts':
                    output = output.view(-1)
                    pred = output
                else:
                    output = output.view(-1,2)
                    pred = torch.max(output, 1)[1]
                for i in range(len(pred)):
                    ans.append(int(pred[i]))
                    
        with open(pred_path, 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['Id', 'Label'])

            if args.task == 'qnli' or args.task == 'rte':
                for idx, label in enumerate(ans):
                    if label == 1:
                        tsv_writer.writerow([idx, 'entailment'])
                    else:
                        tsv_writer.writerow([idx, 'not_entailment'])
            else:
                for idx, label in enumerate(ans):
                    tsv_writer.writerow([idx, label])
    elif args.task == 'mnli':

        with torch.no_grad():
            for batch_id, data in enumerate(tqdm(dataloader)):
                tokens, mask, type_id, _ = data
                tokens, mask, type_id = tokens.to(device),mask.to(device), type_id.to(device)
                output = model(tokens = tokens, mask = mask, type_id = type_id)
                output = output.view(-1,3)
                pred = torch.max(output, 1)[1]
                for i in range(len(pred)):
                    ans.append(int(pred[i]))
                    
        with open(pred_path, 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['Id', 'Label'])
            for idx, label in enumerate(ans):
                if label == 0:
                    tsv_writer.writerow([idx, 'contradiction'])
                if label == 1:
                    tsv_writer.writerow([idx, 'neutral'])
                else:
                    tsv_writer.writerow([idx, 'entailment'])