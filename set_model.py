import redis ai
from transformers import BertTokenizer, BertForPreTraining
import torch
import torch.nn as nn


#initializing bert model
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForPreTraining.from_pretrained('bert-base-uncased',return_dict=False)
#model.cls.seq_relationship = nn.Identity()

con = redisai.Client()
model_file = 'bert_pretrained.pt'
bert_model  =  open(model_file, 'rb').read()


con.modelset('bert-qa', 'TORCH', 'CPU', bert_model)
bert_languages = ['python','ruby','java','javascript','php','go']

for lang in bert_languages:
    con.tensorset(lang+'_embeddings',np.load('language/'+lang+'_bert.npy'))

