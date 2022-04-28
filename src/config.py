import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # options
    parser.add_argument('-s', '--seed', default=307, type=int)
    parser.add_argument('-g', '--GLUE_path', default='../GLUE', type=str)
    parser.add_argument('-o', '--output_path', default='./results', type=str)
    '''
    output_path就是你要跑的模型位置，資料夾名稱要有模型名稱
    ex: adapter_alpha --> 'gdrive/My Drive/colab/307/alpha
        adapter_xi --> 'gdrive/My Drive/colab/307/xi
        adapter_bias_learn --> 'gdrive/My Drive/colab/307/vector
        adapter_bias_fix --> 'gdrive/My Drive/colab/307/vector_fix

        如果要讓bias的xi每層都一樣記得要讓task_specific = 1
    '''
    
    parser.add_argument('--share_alpha', default=0, type=int) #0 is for different xi in each layer, 1 is for same xi in each layer
    parser.add_argument('-m', '--model', default='bert-base', type=str)
    parser.add_argument('-t', '--task', default='cola', type=str)
    parser.add_argument('-a', '--adapter', default=True, type=bool)

    #universial
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--lr', default=0.0001)
    parser.add_argument('--max_len', default=128, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    #COLA
    parser.add_argument('--cola_epoch', default=120, type=int)
    parser.add_argument('--cola_lr', default=0.0001)
    parser.add_argument('--cola_len', default=128, type=int)
    parser.add_argument('--cola_batch', default=32, type=int)

    #MNLI
    parser.add_argument('--mnli_epoch', default=20, type=int)
    parser.add_argument('--mnli_lr', default=0.0001)
    parser.add_argument('--mnli_len', default=128, type=int)
    parser.add_argument('--mnli_batch', default=32, type=int)

    #MRPC
    parser.add_argument('--mrpc_epoch', default=100, type=int)
    parser.add_argument('--mrpc_lr', default=0.0001)
    parser.add_argument('--mrpc_len', default=128, type=int)
    parser.add_argument('--mrpc_batch', default=32, type=int)

    #QNLI
    parser.add_argument('--qnli_epoch', default=25, type=int)
    parser.add_argument('--qnli_lr', default=0.0001)
    parser.add_argument('--qnli_len', default=512, type=int)
    parser.add_argument('--qnli_batch', default=4, type=int)

    #QQP
    parser.add_argument('--qqp_epoch', default=20, type=int)
    parser.add_argument('--qqp_lr', default=0.0001)
    parser.add_argument('--qqp_len', default=330, type=int)
    parser.add_argument('--qqp_batch', default=8, type=int)

    #RTE
    parser.add_argument('--rte_epoch', default=120, type=int)
    parser.add_argument('--rte_lr', default=0.0001)
    parser.add_argument('--rte_len', default=350, type=int)
    parser.add_argument('--rte_batch', default=1, type=int)

    #SST-2
    parser.add_argument('--sst_epoch', default=80, type=int)
    parser.add_argument('--sst_lr', default=0.0001)
    parser.add_argument('--sst_len', default=128, type=int)
    parser.add_argument('--sst_batch', default=16, type=int)

    #STS-B
    parser.add_argument('--sts_epoch', default=30, type=int)
    parser.add_argument('--sts_lr', default=0.0001)
    parser.add_argument('--sts_len', default=512, type=int)
    parser.add_argument('--sts_batch', default=8, type=int)

    #Squard
    parser.add_argument('--squard_epoch', default=2, type=int)
    parser.add_argument('--squard_lr', default=0.00003, type = float)
    parser.add_argument('--squard_max_len', default=384, type=int)
    parser.add_argument('--squard_doc_stride', default=128, type=int)
    parser.add_argument('--squard_batch', default=3, type=int)
    
    args = parser.parse_args()
    
    return args