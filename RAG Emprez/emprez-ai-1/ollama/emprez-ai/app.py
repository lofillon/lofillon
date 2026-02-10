from flask import Flask, request
from ai_process import EmprezAIProcessor

app = Flask(__name__)

processor = EmprezAIProcessor('/tmp/lanceDB')

def create_app():
    return app

@app.route("/question")
def question():
    question = request.args.get('query')
    if question is None:
       return "No Question"
    else:
        response,links = processor.process(question)
        return {'response':response,'links':links}    


@app.route("/")
def hello():
    return "Emprez AI - 0.0.1"
