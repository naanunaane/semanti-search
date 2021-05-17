from flask import Flask, render_template, request, url_for

app = Flask(__name__)

import tensorflow as tf
import pandas as pd
import re
from transformers import BertTokenizer
import redisai
import redis
from nearpy import Engine
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#model = BertForPreTraining()
#model.return_dict = False
#model.cls.seq_relationship = nn.Identity()
import numpy as np
from tqdm import tqdm

#df = pd.read_csv('final_search.csv')

#search_index = nmslib.init(method='hnsw', space='cosinesimil')


#search_index.loadIndex('./final.nmslib')


# Initialising the redisai client
con = redisai.Client()
dbcon = redis.StrictRedis(host='localhost', port=12000, db=0,password="codesearch")
model_file = 'bert_pretrained.pt'
bert_model  =  open(model_file, 'rb').read()


con.modelset('bert-qa', 'TORCH', 'CPU', bert_model)
bert_languages = ['python','ruby','java','javascript','php','go']

for lang in bert_languages:
    con.tensorset(lang+'_embeddings',np.load('language/'+lang+'_bert.npy'))



def search_bert(query,language):
    # load the corresponding embeddings dataset
 
    dimension = 768
    embeddings = con.tensorget(language+'_embeddings')
    engine = Engine(dimension)
    for index in tqdm(range(5000)):
        v = embeddings[index]
        engine.store_vector(v,index)
    inputs  = tokenizer(query,max_length=512,return_tensors='pt')
    con.tensorset('input_ids',np.array(inputs['input_ids']))
    con.tensorset('token_type_ids',np.array(inputs['token_type_ids']))
    con.modelrun('bert-qa', ['input_ids',  'token_type_ids'],['prediction_logits', 'seq_relationship_logits'])
    output =  con.tensorget('seq_relationship_logits').squeeze()
    neighbors=engine.neighbours(output)
    count = 0
    functions = []
    docs = []
    dists = []
    for _,idx,dist in tqdm(neighbors):
        count = count+1
        codestring = dbcon.get(language+'_'+str(idx)).decode('utf-8')
        docstring,code = codestring.split('<CODESTRING>')
        print(f'distance:{dist:.4f} \n {docstring}  \n {code} \n------------\n')
        functions.append(code)
        dists.append(dist)
        docs.append(docstring)
    #should add list of git for this
    return functions,dists,docs

print(search_bert('testing this','python'))


@app.route('/')
def main_page():
    return render_template("main_page.html")

@app.route('/results', methods=['GET'])
def results_page():
	query = request.args.get('query')
	funcs,dists,docs = search_bert(query,'python')
	values = len(funcs)
	return render_template("results_page.html",data=query, result = values, codes=funcs,dist=dists,docstrings=docs)

if __name__ == "__main__":
    app.run()
