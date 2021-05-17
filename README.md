# Semanti-search
A Flask web application built to utilise the power of Semantic Search to help developers get to the solution as quickly as possible by leveraging the legacy of bug-fixing and features built by people around the world.

Engineers work best when they can easily find code examples to guide them on particular coding tasks. For some questions — for example, “How to programmatically close or hide the Android soft keyboard?” — information is readily available from popular resources like Stack Overflow. 

It also becomes imperative to consider the fact that different people do things differently when given an objective and just considering the code string was not going to be enough. We had to consider the documentation added for the function modules to better grasp the semantic meaning of that piece of code.

However, things become difficult when the developers have to find ways to access/search for design patterns or code modules already implemented in the company's proprietary code [[1]](https://ai.facebook.com/blog/neural-code-search-ml-based-code-search-using-natural-language-queries/). 

For this reason, we decided to develop a way for people to access solutions by harnessing the power of the web as well as the proprietary code base using their own natural queries. To implement this, we had to go beyond basic document searches that the code bases like Github and Gitlab provide to us with regex search and keyword search. The search with the natural langauge had to encode the semantic meaning of the query and try to find code documents that were semantically similar to what the user wanted. 

## Architecture of Semanti-Search

![Architecture Diagram](https://github.com/naanunaane/semanti-search/blob/main/code_search_architecture.png "Architecture Diagram")


## Methods:

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
4. Redisai docker container
5. Redis cloud database
### setup

### Docker image setup:
Run the docker-image for redisai exposing the 6379 port 

#### Database setup:
Code-docstring pairs are saved in a csv formate as shown in example port, these are added to redis-cloud database as shown in the set_model.py script
