from ai_process import EmprezAIProcessor
import sys


def main():
    if len(sys.argv) != 2 :
        print('Usage python manual.py <Question>')
        exit()
    question = sys.argv[1]
#    processor = EmprezAIProcessor('/tmp/lanceDB')
    processor = EmprezAIProcessor('../../lanceDB.2025.01.29')
    response,links = processor.process(question)
    print(f'response: {response} , links {links}')    


'''

#'''

if __name__ == "__main__":
    main()