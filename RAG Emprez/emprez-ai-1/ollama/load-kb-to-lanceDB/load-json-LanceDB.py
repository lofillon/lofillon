import json
from pathlib import Path
import pandas as pd
# import nltk
# nltk.download("punkt")
import re
import ollama

# lancedb imports for embedding api
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

LANCE_DB = '/tmp/lanceDB'

model = get_registry().get("colbert").create(name="colbert-ir/colbertv2.0")

class ArticleModel(LanceModel):
    title: str = model.SourceField()
    content: str = model.SourceField()
    url: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

def createTable():
    db = lancedb.connect(LANCE_DB)
    table = db.create_table(
        "articles",
        schema=ArticleModel,
        mode="overwrite",
    )
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
           # print('*** Chunk *** -->',chunk)
        current_chunk_count += max_chunk_length

    return result


def readJsonAsDict(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def readJsonFileAndSaveInLance(filename, table):
    print('Reading ',filename)
    data = readJsonAsDict(filename)
    print('data.title',data['title'])
    print('data.url',data['url'])
    chunks = recursive_text_splitter(data['content'], max_chunk_length=1000, overlap=100)
    #print("Chunks read: ",chunks)
    df = pd.DataFrame({"content": chunks,"title":data['title'],"url":data['url']})
    #print("Pandas DataFrame ",df)
    table.add(df)



#test = 'data/fr/emprez_business_intelligence/Module Emprez BI - Fins de quarts retardées.json'


table = createTable()

directory = 'data'   

pathlist = Path(directory).rglob('*.json')
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    readJsonFileAndSaveInLance(path,table)
    #print(path_in_str)


print(f'Load in LanceDB {LANCE_DB} done')