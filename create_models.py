from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as rt
from skl2onnx import to_onnx
import pandas as pd
import numpy as np

# sudo docker run \
# > -p 6379:6379 \
# > -v /home/user/data-redis:/data \
# > -v /home/user/redis.conf:/usr/local/etc/redis/redis.conf \
# > redislabs/redismod \
# > /usr/local/etc/redis/redis.conf
# To run redis locally

# first we load the csv's docstring column to get the corpus
column_names = ["bool", "code_link", "funcName", "docstring", "codestring"]
corpus_python_df = pd.read_csv("../../data/codesearch/train_valid/python/train_python.csv", names=column_names)
corpus_go_df = pd.read_csv("../../data/codesearch/train_valid/go/train_go.csv", names=column_names)
corpus_js_df = pd.read_csv("../../data/codesearch/train_valid/javascript/train_javascript.csv", names=column_names)
corpus_java_df = pd.read_csv("../../data/codesearch/train_valid/java/train_java.csv", names=column_names)
# initialising the vectorizer object
vectorizer = TfidfVectorizer()
tfidf_python = vectorizer.fit_transform(corpus_python_df['docstring'][1:5001].apply(lambda x: np.str_(x)))

# initialising the reducer with truncated svd
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
lsi_python = svd.fit_transform(tfidf_python)

lsi_vector_df = pd.DataFrame(lsi_python)
lsi_vector_df.to_csv("../../data/codesearch/train_valid/python/train_python_matrix_5000.csv")

queryString = ["write metadata to file"]
query_python = vectorizer.transform(queryString)
query_lsi = svd.transform(query_python)
cosine_similarity(lsi_python, Y=query_lsi, dense_output=True)

try:

    svd_onnx = to_onnx(svd, query_python.toarray().astype(np.float32), target_opset=12)
except RuntimeError as e:
    print(e)
# writing model to redisai

with open("../../data/codesearch/train_valid/python/svd_python.onnx", "wb") as f:
    f.write(svd_onnx.SerializeToString())

sess = rt.InferenceSession("../../data/codesearch/train_valid/python/svd_python.onnx")
results = sess.run(None, {'X': query_python.todense().astype(np.float32)})
# results[0] gives us the ndarray we expect
cosine_similarity(lsi_python, Y=results[0], dense_output=True)

from redisearch import Client, IndexDefinition, TextField, Query
from redis import ResponseError

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


def create_response_redisearch(query, client, num):
    """
    Returns list of dictionary containing the results for the specific query passed to the function
    with TFIDF.DOCNORM scorer in RediSearch
    ### Parameters
    - query: Query string passed by user
    - client: The RediSearch client object for that specific index
    - num: Maximum number of records to be sent back
    """

    # Using TFIDF.DOCNORM scorer for the documents. For more info see: https://oss.redislabs.com/redisearch/Scoring/
    q = Query(query).scorer('TFIDF.DOCNORM').with_scores()
    response = client.search(q)
    # Initialising the output list
    output = []
    # Getting the min between the number of documents received and the limit
    length = min(num, len(response.docs))
    for i in range(length):
        output.append(
            {"id": int(response.docs[i].id.replace('doc:', '')),
             "codelink": response.docs[i].url,
             "funcName": response.docs[i].funcName,
             "score": response.docs[i].score,
             "docString": response.docs[i].docString,
             "codeString": response.docs[i].codeString}
        )

    return output

def create_response_model(ids, client, scores):
    """
    Returns list of dictionary containing the records for the specific function ids passed to the function
    using the specific client for the index
    ### Parameters
    - ids: List of ids (int) of the required documents.
    - client: The RediSearch client object for that specific index
    - scores: List of scores of the corresponding id in `ids` list
    """
    # The prefix used in IndexDefinition while creating the index
    indexing = "doc"
    indexing += '{0}'
    # Adding the prefix to all the ids. Ex: [1,2] becomes ["doc:1", "doc:2"]
    ids_pf = [indexing.format(i) for i in ids]

    # Initialising output array
    output = []
    # As I cannot pass a list/collection to the get function,
    # I'll have to specifically get each of the variables in list and pass
    # There should be a better way to do this
    for i in range(len(ids)):
        response = client.get(ids_pf[i])[0]
        output.append(
            {"id": response.docs[i].id,
             "codelink": response.docs[i].url,
             "funcName": response.docs[i].funcName,
             "score": scores[i],
             "docString": response.docs[i].docString,
             "codeString": response.docs[i].codeString}
        )

    return output

import tensorflow as tf
import redisai
con = redisai.Client("localhost", 6379)
con.loadbackend("ONNX", "/usr/lib/redis/modules/backends/redisai_onnxruntime/redisai_onnxruntime.so")
con.loadbackend("TF", "/usr/lib/redis/modules/backends/redisai_tensorflow/redisai_tensorflow.so")
con.tensorset('query_vec', query_python.todense().astype(np.float32), dtype='float32')
con.modelset(key = "python_svd", backend = "ONNX", device = "CPU", data = svd_onnx.SerializeToString(), tag = "v1.00", inputs=['X'], outputs=['mul'])
con.modelrun(key = "python_svd", inputs = ["query_vec"], outputs = ["query_svd"])

# @tf.function
# def compute_cosine_similarities(a, b):
#     # x shape is n_a * dim
#     # y shape is n_b * dim
#     # results shape is n_a * n_b
#
#     normalize_a = tf.nn.l2_normalize(a,1)
#     normalize_b = tf.nn.l2_normalize(b,1)
#     similarity = tf.matmul(normalize_a, normalize_b, transpose_b=False)
#     return similarity

# class Scorer(tf.Module):
#   @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
#   def compute_cosine_similarities(self, corpus, query):
#       # x shape is n_a * dim
#       # y shape is n_b * dim
#       # results shape is n_a * n_b
#       normalize_a = tf.nn.l2_normalize(corpus, 1)
#       normalize_b = tf.nn.l2_normalize(query, 1)
#       similarity = tf.matmul(normalize_a, normalize_b, transpose_b=True)
#       return similarity
#
# scorer = Scorer()
# tf.saved_model.save(scorer, '../../data/codesearch/train_valid/python/scorer.h5')

from sklearn.base import TransformerMixin
import skl2onnx.common.data_types as sklearn_types
from joblib import load, dump
# NOTE: Due to issues with conversion of TFIDF tokeniser to ONNX, I am forced to pickle the fit vectoriser
# so that it can be utilised to vectorise the query string. See: http://onnx.ai/sklearn-onnx/parameterized.html

# Another approach for this is to utilise another vectoriser like HashingVectorizer etc, but that could reduce
# the performance of this method

# Initialising the vectorizer object
vectoriser = TfidfVectorizer()
tfidf_python = vectoriser.fit_transform(corpus_python_df['docstring'][1:5001].apply(lambda x: np.str_(x)))

# Saving the vectoriser in a pickle file
dump(vectoriser, "../../data/codesearch/train_valid/python/tfidf_python.joblib")

vec = load("../../data/codesearch/train_valid/python/tfidf_python.joblib")
queryString = ["write metadata to file"]
query_python = vec.transform(queryString)

class SVD_plus_scorer(TransformerMixin):
    def __init__(self, n_components, n_iter, random_state):
        """
        Initialises the TruncatedSVD that is used for dimensionality reduction of the TFIDF vector
        """
        self.reducer_ = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=random_state)

    def fit_transform(self, corpus, y=None):
        """
        Fits the TruncatedSVD to the corpus document and returns the reduced dimensions (to n_components seen above)
        ----
        Returns:
            ndarray(nrows(corpus), n_components): This is the reduced dimensions of the
                                                  vectors for the corpus used for training
        """
        self.reducer_.fit(corpus)
        return self.reducer_.transform(corpus)

    def transform(self, corpus_svd, query):
        """
         Transforms the query string's tokenised vector to lower dimension by using the reducer
         Also computes the cosine similarity between all documents of the corpus and the query string
        ----
        Returns:
        ndarray(nrows(corpus), 1): These are the similarity scores between the documents and the query string
        """
        query_lsi = self.reducer_.transform(query)
        return cosine_similarity(corpus_svd, query_lsi, dense_output=True)

lsa = SVD_plus_scorer(5,7,42)
lsi_python3 = lsa.fit_transform(tfidf_python)

svd_mixin = skl2onnx.wrap_as_onnx_mixin(lsa)

query_python3 = ["Write metadata to file"]
scores3 = lsa.transform(lsi_python3, vec.transform(query_python3))

initial_type = [('query', sklearn_types.FloatTensorType([None, None]))]

try:
    lsa_onnx = to_onnx(lsa, initial_types=initial_type,target_opset=12)
except RuntimeError as e:
    print(e)
# writing model to redisai

with open("../../data/codesearch/train_valid/python/lsa_python.onnx", "wb") as f:
    f.write(lsa_onnx.SerializeToString())


import tensorflow as tf

input_corpus = tf.keras.Input(shape=(5000,5))
input_query = tf.keras.Input(shape=(1,5))
outputs = tf.keras.layers.Dot(axes=(2), normalize=True)([input_corpus, input_query])
model1 = tf.keras.Model(inputs=[input_corpus, input_query], outputs=outputs)

