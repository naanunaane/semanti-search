# Semanti-search
A Flask web application built to utilise the power of Semantic Search to help developers get to the solution as quickly as possible by leveraging the legacy of bug-fixing and features built by people around the world.

Engineers work best when they can easily find code examples to guide them on particular coding tasks. For some questions — for example, “How to programmatically close or hide the Android soft keyboard?” — information is readily available from popular resources like Stack Overflow. 

It also becomes imperative to consider the fact that different people do things differently when given an objective and just considering the code string was not going to be enough. We had to consider the documentation added for the function modules to better grasp the semantic meaning of that piece of code.

However, things become difficult when the developers have to find ways to access/search for design patterns or code modules already implemented in the company's proprietary code [[1]](https://ai.facebook.com/blog/neural-code-search-ml-based-code-search-using-natural-language-queries/). 

For this reason, we decided to develop a way for people to access solutions by harnessing the power of the web as well as the proprietary code base using their own natural queries. To implement this, we had to go beyond basic document searches that the code bases like Github and Gitlab provide to us with regex search and keyword search. The search with the natural langauge had to encode the semantic meaning of the query and try to find code documents that were semantically similar to what the user wanted. 

## Architecture of Semanti-Search

![Architecture Diagram](https://github.com/naanunaane/semanti-search/raw/4b3abfe2bc6749cdda4bb2a7e38b154045e8deca/resources/Arch_Diagram_v1.1.png "Architecture Diagram")


## Methods:

### Latent Semantic Analysis:
The cornerstone method of semantic web search, Latent Semantic Analysis (LSA), or Latent Semantic Indexing (LSI) is a very convenient and generalisable method to find similarity between documents for the semantic meaning embedded in them. 

The steps involved in LSA include:

### Build a vocabulary and a tokeniser
Using TFIDF vectorisers, we first build the vocabulary of the text corpus. Due to the word being encoded with certain meaning, it was imperative for us to consider the fact that different words mean different things in different languages. 

For this reason, we build the vocabulary and the tokeniser for each language. The vocabulary and the tokeniser is built for the docstring of each function in our corpus. 

### Reduce dimensions of the vectors
To make the semantic search scalable, memory efficient and have better performance, we need to reduce the dimensions of the vectors by using Truncated Singular Value Decomposition. 

This method lets us also standardise the number of dimensions corpus and the query are restricted to. 

For the purposes of keeping the data minimal in size to be able to run in local setup, the current code in the repository truncates the vectors to 5 dimensions. People experimenting on the repository can tune this to their data. 

### Calculate similarity scores between the vectors
Cosine Similarity is a common method to calculate similarity between documents and other vectors due to it being significant as well as performance friendly. 

The cosine similarity is defined as the dot product of the vectors divided by the magnitude of the vectors. 

### BERT
BERT (Bidirectional Encoder Representations) is a Transformer based language model pretrained on english lancuage corpus with around 3.3B word-count.This model reported state-of-the-art performance on a wide range of natural language understanding tasks. 

BERT model consists of an encoder and decoder, the query sentance is encoded by the model into a high dimentional vector which is decoded by the decoder. We use the encoder of a pretrained-BERT model to encode queries into vectors which are fed to the nearest-neighbor search.

### Nearest Neighbor Search
Docstrings of the functions are converted into vectors and used ffor approximate nearest neighbor matching based on the query. For each query vector obtained from the BERT-encoder 10 nearest docstring vectors are found and returned based on their distance from the query.


## Dev Setup

### Requirements:
1. Ubuntu 18.04 system
2. Docker
3. Python 3.x
4. Redismod docker container

### Local setup

### Docker image setup:
the redislabs/redismod built from source resulted in errors while loading pytorch model, so the shared-object files from redislabs/redisai image are copied to the redismod container in the corresponding library location (/usr/lib/redis/modules/backends/redisai_torch/lib)
`
#### Setup redis docker container locally:
1. After making sure that you have docker container running

`docker run \
 -p 6379:6379 \
 -v /home/user/data-redis:/data \
 -v /home/user/redis.conf:/usr/local/etc/redis/redis.conf \
 redislabs/redismod`

#### Writing data to RediSearch and RedisAI:
1. Go inside the scripts directory from the current directory of this project

`cd scripts`
   
2. Run python script to install required dependencies

`python3 set_data_on_redis.py`

#### [Optional] If you want to add your own data, and models
1. Create a new directory for your language

`mkdir data_&_models/new_lang`

2. Add your data as a csv in the directory. To be able to run the script smoothly, follow this naming convention. 

2.1. CSV name of corpus: train_{}_small.csv       (replace {} with language)

2.2. CSV name of vectors: train_{}_matrix_{}.csv  ({} with language, number of rows)

3. Run script to get vectors and models

`cd scripts`

`python3 create_LSA_vectors_&_models.py`
