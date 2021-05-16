from flask import Flask, request, Response, render_template
import requests
import json
from joblib import load, dump
from redisearch import Client, Query
import redisai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# Initialising the redisai client
con = redisai.Client("localhost", 6379)

# Initialising the redisearch clients objects inside dictionary
clients = {}

# Initialising the vectorisers inside dictionary
vectorisers = {}

lang_list = ["python", "go", "javascript", "java"]
for lang in lang_list:
    clients["{}_client".format(lang)] = Client("{}_docs".format(lang))
    old_vectoriser = load("data_&_models/{}/tfidf.pkl".format(lang))
    new_vectoriser = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words = "english", lowercase = True,
                          max_features = 500000, vocabulary=old_vectoriser.vocabulary_)
    vectorisers["{}_vec".format(lang)] = new_vectoriser

def create_response_model(ids, client, scores):
    """
    Returns list of dictionary containing the records for the specific function ids passed to the function
    using the specific client for the index
    ### Parameters
    - ids: List of ids (int) of the required documents.
    - client: The RediSearch client object for that specific index
    - scores: List of scores of the corresponding id in `ids` list
    """

    # Initialising output array
    output = []
    # As I cannot pass a list/collection to the get function,
    # I'll have to specifically get each of the variables in list and pass
    # There should be a better way to do this
    for i in range(len(ids)):
        response = client.get("doc:{}".format(ids[i]))[0]
        print("The response for id {} is {}".format(i, response))
        output.append(
            {"id": response.docs[i].id,
             "codelink": response.docs[i].url,
             "funcName": response.docs[i].funcName,
             "score": scores[i],
             "docString": response.docs[i].docString,
             "codeString": response.docs[i].codeString}
        )

    return output

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

def run_lsi(query, lang, con):
    """
    Returns the response object to be sent back for the model
    ### Parameters
    - query: Query string passed by user
    - lang: The programming language for which query was made
    - con: The RedisAI client
    """
    # First, we vectorise the query string and then set the tensor in redisai
    query_vec = vectorisers["{}_vec".format(lang)].fit_transform(list(query))
    print(query_vec.getrow(0).todense())
    con.tensorset('query_vec', query_vec.getrow(0).todense().astype(np.float32), dtype='float32')
    con.modelrun(key="{}_svd".format(lang), inputs=["query_vec"], outputs=["query_svd"])
    query_svd = con.tensorget(key="query_svd", as_numpy=True)
    corpus_vec = con.tensorget(key="{}_vec".format(lang), as_numpy=True)
    print(corpus_vec[1:10])
    print(query_svd)
    scores = cosine_similarity(corpus_vec, Y=query_svd, dense_output=True)
    print("printing scores", scores)
    ids = np.argsort(scores).tolist()
    # print("From run_lsi, going to query for these ids", ids)
    response = create_response_model(ids[0:2], clients["{}_client".format(lang)], scores.tolist()[0:2])

    return response

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/hello/<name>')
# ‘/hello’ URL is bound with name_of_app() function. name is name of the user
def name_of_app(name):
    return 'Hello %s, Welcome to SemantiSearch! \n' \
           'With our service, you can search for relevant answers in stackoverflow ' \
           'and github code modules just by using natural language' % name

@app.route('/')
# '/search' URL is bound with search_only_bar() function
def search_only_bar():
    return render_template(
        '/searchLandingView/landingPageOnlyBar.html',
    )

@app.route('/search/')
# '/search' URL is bound with search_landing_page() function
def search_landing_page():
    queryString = request.args.get('querystring')
    chosenLanguage = request.args.get('lang')
    semanticsNeeded = request.args.get('semantics')
    print(chosenLanguage)
    print(semanticsNeeded)
    return render_template(
        '/searchLandingView/landingPage2Cards.html',
        queryValue=queryString,
        chosenLanguage=chosenLanguage,
        semanticsNeeded=semanticsNeeded,
    )


@app.route('/search/model/<string:query>/')
@app.route('/search/model/<string:query>/<string:lang>/')
# '/search' URL is bound with search_landing_page() function
def search_model(query,lang="python"):
    """
    Returns the response object to be sent back from redisearch and model
    ### Parameters
    - query: Query string passed by user
    - lang: The programming language for which query was made. Default is "python"
    """
    # uncomment to read from static file
    # with open("data/sample_model_output.json", "r") as read_file:
    #     model_data = json.load(read_file)
    model_data = {"queryID": 11110000,
                  "model1": create_response_redisearch(query, clients["{}_client".format(lang)], 3),
                  "model2": run_lsi(query, lang, con),
                  "model3": []}

    print("The model data response is ", model_data)
    return model_data
    # return render_template(
    #     '/searchLandingView/landingPage2Cards.html',
    #     title="Jinja Demo Site",
    #     description="Smarter page templates with Flask & Jinja."
    # )



@app.route('/search/<string:query>/')
@app.route('/search/<string:query>/<string:lang>')
# '/search' URL is bound with search_stack_overflow() function
# function accepts the query string and returns top num questions
# by relevance using the stackoverflow API
def search_stack_overflow(query,lang='python'):
    # first getting the argument 'query' from the GET request
    # which is the natural language query string
    query = request.args.get('query')
    # getting the response from the stackoverflow API
    num = 5
    ques_data = requests.get('https://api.stackexchange.com/2.2/search/advanced?pagesize={}&order=desc&sort=relevance&q={}{}{}&site=stackoverflow'.format(num, query, " ",lang))

    # uncomment this to run in local without hitting API all the time
    # with open("data/sample-stackoverflow-query-response.json", "r") as read_file:
    #     ques_data = json.load(read_file)

    # preparing the output json
    response = {'has_more': ques_data.json().get('has_more'), 'items': {}}  # added key telling whether there are more questions for query or not
    question_ids = []  # Empty list of question ids

    print("printing items of stackoverflow query", ques_data.json().get('items'))

    # looping through each question if response object actually has output
    if len(ques_data.json().get('items')) != 0:
        for question in ques_data.json().get('items'):
            # getting required values related to the question
            question_details = {'creation_date': question.get("creation_date"),
                                'answer_count': question.get("answer_count"),
                                'body': question.get("body"),
                                'is_answered': question.get("is_answered"),
                                'last_activity_date': question.get("last_activity_date"),
                                'link': question.get("link"),
                                'tags': question.get("tags"),
                                'title': question.get("title"),
                                'view_count': question.get("view_count"),
                                'is_accepted': not (question.get("accepted_answer_id") == None),
                                'accepted_id': question.get("accepted_answer_id"),
                                'answers': list()}
            question_ids.append(question.get("question_id"))  # appending the question id to the list of question ids
            response['items'][question.get("question_id")] = question_details

        # getting answers for the questions
        if len(question_ids) != 0:
            ans_data = requests.get(
            'https://api.stackexchange.com/2.2/questions/{}/answers?pagesize={}&order=desc&sort=votes&site=stackoverflow'.format(
                ';'.join(map(str, question_ids)), num * 5))

            # uncomment the below part to run in local without hitting API all the time
            # with open("data/sample-stackoverflow-answer-response.json", "r") as read_file:
            #     ans_data = json.load(read_file)

            for answer in ans_data.json().get('items'):
                # getting the details about the answer
                answer_details = {'body': answer.get("body"),
                                  'is_accepted': answer.get("is_accepted"),
                                  'score': answer.get("score"),
                                  'answer_id': answer.get("answer_id")}
                response['items'][answer.get("question_id")]['answers'].append(answer_details)
    return response


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.

    # parameters to run method include:
    # 1. host: Hostname to listen on. Defaults to 127.0.0.1 (localhost).
    #          Set to ‘0.0.0.0’ to have server available externally
    # 2. port: Defaults to 5000
    # 3. debug: Defaults to false. If set to true, provides a debug information. Set to false while running in Prod
    app.run('0.0.0.0', 5000, True)
