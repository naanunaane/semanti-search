from flask import Flask, request
import requests
import json

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/hello/<name>')
# ‘/hello’ URL is bound with name_of_app() function.
def name_of_app(name):
    return 'Hello %s, Welcome to SemantiSearch! \n' \
           'With our service, you can search for relevant answers in stackoverflow ' \
           'and github code modules just by using natural language' % name

@app.route('/ques/<int:num>')
# '/ques' URL is bound with search_stack_overflow() function
# function accepts the query string and returns top num questions
# by relevance using the stackoverflow API
def search_stack_overflow(num):
    # first getting the argument 'query' from the GET request
    # which is the natural language query string
    query = request.args.get('query')
    # getting the response from the stackoverflow API
    # ques_data = requests.get('https://api.stackexchange.com/2.2/search/advanced?pagesize={}&order=desc&sort=relevance&q={}&site=stackoverflow&filter=!9Qz3Xf4W6'.format(num, query))

    with open("data/sample-stackoverflow-query-response.json", "r") as read_file:
        ques_data = json.load(read_file)

    # preparing the output json
    response = {'has_more': ques_data.get('has_more'), 'items': {}}  # added key telling whether there are more questions for query or not
    question_ids = []  # Empty list of question ids

    # looping through each question now
    for question in ques_data.get('items'):
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
                            'is_accepted': not(question.get("accepted_answer_id") == None),
                            'accepted_id': question.get("accepted_answer_id"),
                            'answers': list()}
        question_ids.append(question.get("question_id"))  # appending the question id to the list of question ids
        response['items'][question.get("question_id")] = question_details

    # getting answers for the questions
    # ans_data = requests.get('https://api.stackexchange.com/2.2/questions/{}/answers?pagesize={}&order=desc&sort=votes&site=stackoverflow&filter=!9Qz3Xr)ML'.format(';'.join(map(str, question_ids)), num*5))

    with open("data/sample-stackoverflow-answer-response.json", "r") as read_file:
        ans_data = json.load(read_file)

    for answer in ans_data.get('items'):
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
