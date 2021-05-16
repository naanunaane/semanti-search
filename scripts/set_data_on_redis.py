from redisearch import Client, IndexDefinition, TextField
from redis import ResponseError
import redisai
import pandas as pd
import numpy as np
from onnx import load as onnx_load
import ml2rt

def create_index_in_redisearch(lang):
    """
    Creates a redisearch client object and the index if it doesn't already exist.
    Returns redisearch client object.
    ### Parameters
    - lang: The programming language for which we need the client and index to be created
    """
    SCHEMA = (
        TextField("funcName"),
        TextField("url"),
        TextField("docString", weight=5.0),
        TextField("codeString", weight=5.0)
    )
    definition = IndexDefinition(prefix=['doc:'])
    client = Client("{}_docs".format(lang))
    try:
        client.info()
    except ResponseError:
        # Index does not exist. We need to create it!
        client.create_index(SCHEMA, definition=definition)

    return client


def write_df_to_redis(corpus_df, num, client):
    """
    Writes top num values of the dataframe to redisearch for the passed client.
    Returns nothing.
    ### Parameters
    - corpus_df: Pandas dataframe with column name containing the corpus
    - client: The RediSearch client object for that specific index
    - num: Maximum number of records to be written to Redisearch with the index in client
    """
    for i in range(1, num + 1):
        doc = {
            'funcName': corpus_df['funcName'][i],
            'docString': corpus_df['docstring'][i],
            'codeString': corpus_df['codestring'][i],
            'url': corpus_df['code_link'][i]
        }
        client.redis.hset('doc:{index}'.format(index=i), mapping=doc)


# Creating a client object for redisai
con = redisai.Client("localhost", 6379)

# You shouldn't run this if the backend has already been loaded
# Throws up an error if reloaded
#con.loadbackend("TORCH", "/usr/lib/redis/modules/backends/redisai_torch/redisai_torch.so")

# Listing down the languages we're supporting.
# If you want to have your models running only for some languages,
# remove that language from this list
lang_list = ["python", "javascript", "go", "java"]

# Next, we load the csvs to get the corpus for each programming language
column_names = ["bool", "code_link", "funcName", "docstring", "codestring"]

# For the purposes of running things without restrictions on memory,
# even on resource constrained systems, we write only 5000 functions per language
num_rows = 5000

# For every language we support, set the SVD model, the reduced TFIDF vectors for LSI
# on redis for faster inference. We also set the actual function data on RediSearch to allow for the
# basic TFIDF scored RediSearch on the function data used in the Flask app
for lang in lang_list:
    # Using pandas to read csv
    corpus_df = pd.read_csv("../data_&_models/{}/train_{}_small.csv".format(lang, lang), names=column_names)
    # Getting the client object for the index created for the function
    client = create_index_in_redisearch(lang)
    # Writes the num_rows of the dataframe to redisearch
    write_df_to_redis(corpus_df=corpus_df, num=num_rows, client=client)
    # Reading the reduced vectors for each language
    vectors_df = pd.read_csv("../data_&_models/{}/train_{}_matrix_{}.csv".format(lang, lang, num_rows), index_col=False)
    # Writing the tensor to redisai
    con.tensorset('{}_vec'.format(lang), np.delete(vectors_df.to_numpy(dtype='float32'), 0, 1), dtype='float32')
    # Reading the svd model fit to the language
    svd_onnx = onnx_load('../data_&_models/{}/svd.onnx'.format(lang))
    con.modelset(key="{}_svd".format(lang), backend="ONNX", device="CPU", data=svd_onnx.SerializeToString(),
                 tag="v1.00", inputs=['query_tfidf'], outputs=['mul'])

# # Now we set the torchscript for acting as the scorer
# scorer_script = ml2rt.load_script("calculate_scores.txt")
# con.scriptset("lsi_scorer", "CPU", scorer_script)
#
# # Now we set the BERT model
model_file = '../data_&_models/bert_pretrained.pt'
with open(model_file, 'rb') as f:
    model = f.read()

con.modelset('bert-qa', 'TORCH', 'CPU', model)
