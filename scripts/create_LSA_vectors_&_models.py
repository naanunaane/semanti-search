from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import skl2onnx.common.data_types as sklearn_types
from skl2onnx import convert_sklearn
import pandas as pd
import numpy as np
from joblib import load, dump
from skl2onnx import to_onnx

# Listing down the languages we'll be supporting
lang_list = ["python", "javascript", "go", "java"]

# Next, we load the csvs to get the corpus for each programming language
column_names = ["bool", "code_link", "funcName", "docstring", "codestring"]

# For the purposes of running things without restrictions on memory, we will create our models with
# a very small sample dataset for each programming language
num_rows = 5000

# For every language we support, we create and store the respective fit models in required formats
for lang in lang_list:
    # Using pandas to read csv
    corpus_df = pd.read_csv("../data_&_models/{}/train_{}_small.csv".format(lang, lang), names=column_names)

    # Initialising the vectorizer object with the docstring of the functions
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus_df['docstring'][1:num_rows+1].apply(lambda x: np.str_(x)))

    # Due to issues with onnx packages, we are unable to add the tfidf vectoriser onto onnx
    # See http://onnx.ai/sklearn-onnx/parameterized.html#tfidfvectorizer-countvectorizer
    # We use .pkl format to store the vectoriser. As the vectorisation doesn't really require data locality,
    # we are not losing a lot by not hosting this on redis-ai as well
    # Saving the vectoriser in a pickle file
    dump(vectorizer, "../data_&_models/{}/tfidf.joblib".format(lang))

    # Initialising the reducer with truncated svd to reduce the tokenisation to
    # 5 dimensions for each document
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    lsi = svd.fit_transform(tfidf)

    # Comment the below lines if don't want to actually want to re-write the csvs
    lsi_vector_df = pd.DataFrame(lsi)
    lsi_vector_df.to_csv("../data_&_models/{}/train_{}_matrix_{}.csv".format(lang, lang, num_rows))

    # Converts the fit svd model for the language to onnx file format for further use
    # See http://onnx.ai/sklearn-onnx/introduction.html, for more details when you
    # experiment on your own
    query_string = ["Write metadata to file"]
    query_vectorised = vectorizer.transform(query_string)
    try:
        svd_onnx = to_onnx(svd, query_vectorised.toarray().astype(np.float32), target_opset=12)
    except RuntimeError as e:
        print(e)
    # write the svd to onnx file
    with open("../data_&_models/{}/svd.onnx".format(lang), "wb") as f:
        f.write(svd_onnx.SerializeToString())