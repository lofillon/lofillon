import json
from pathlib import Path
import pandas as pd
import nltk
nltk.download("punkt")
import re
import ollama

# lancedb imports for embedding api
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector


model = get_registry().get("colbert").create(name="colbert-ir/colbertv2.0")

class ArticleModel(LanceModel):
    title: str = model.SourceField()
    content: str = model.SourceField()
    url: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

def createTable():
    db = lancedb.connect("/tmp/lancedb")
    table = db.create_table(
        "acticles",
        schema=ArticleModel,
        mode="overwrite",
    )
    return table


# add in vector db
def lanceDBConnection(df):
    """
    LanceDB insertion
    """
    db = lancedb.connect("/tmp/lancedb")
    table = db.create_table(
        "scratch",
        schema=ArticleModel,
        mode="overwrite",
    )
    table.add(df)
    return table


# Recursive Text Splitter
def recursive_text_splitter(text, max_chunk_length=1000, overlap=100):
    """
    Helper function for chunking text recursively
    """
    # Initialize result
    result = []

    current_chunk_count = 0
    separator = ["\n", " "]
    _splits = re.split(f"({separator})", text)
    splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]

    for i in range(len(splits)):
        if current_chunk_count != 0:
            chunk = "".join(
                splits[
                    current_chunk_count
                    - overlap : current_chunk_count
                    + max_chunk_length
                ]
            )
        else:
            chunk = "".join(splits[0:max_chunk_length])

        if len(chunk) > 0:
            result.append("".join(chunk))
            print('*** Chunk *** -->',chunk)
        current_chunk_count += max_chunk_length

    return result


def readJsonAsDict(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def readJsonFileAndSaveInLance(filename, table):
    print('Reading ',filename)
    data = readJsonAsDict(test)
    print('data.title',data['title'])
    print('data.url',data['url'])
    chunks = recursive_text_splitter(data['content'], max_chunk_length=100, overlap=10)
    #print("Chunks read: ",chunks)
    df = pd.DataFrame({"content": chunks,"title":data['title'],"url":data['url']})
    #print("Pandas DataFrame ",df)
    table.add(df)



test = 'data/fr/emprez_business_intelligence/Module Emprez BI - Fins de quarts retardées.json'

data = readJsonAsDict(test)
print('data.title',data['title'])
print('data.url',data['url'])
#print('data',data)


# Split the text using the recursive character text splitter
chunks = recursive_text_splitter(data['content'], max_chunk_length=100, overlap=10)
print("Chunks read: ",chunks)
#chunks = recursive_text_splitter(data['title'], max_chunk_length=100, overlap=10)
#print("Chunks read: ",chunks)

df = pd.DataFrame({"content": chunks,"title":data['title'],"url":data['url']})

#print("Pandas DataFrame ",df)


table = createTable()
table.add(df)

question = "Quels sont les quarts en retard?"

result = table.search(question).limit(5).to_list()
print('Result title',result.title)

#context = [r["content"] for r in result]
#print('Context',context)

'''
directory = 'data'   

pathlist = Path(directory).rglob('*.json')
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    print(path_in_str)
'''     