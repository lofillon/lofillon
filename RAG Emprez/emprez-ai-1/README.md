# emprez-ai

Ollama can be downloaded here https://ollama.com/download
ollama pull zephyr



Create Environnements
python3.12 -m venv XXXXX.env  # Do NOT use python 3.13+ as long as it is not compatible with torch
source XXXXX.env/bin/activate
python3 -m pip install -r requirements.txt  # When available.
deactivate # to quit env


lancedb.xx contains a copy of the loaded db

ollama 
 - emprez-model with a very basic model for Emprez based on Zephyr
 - load-kb-to-lanceDB loads data contains in data (retrieved and copied from scraping!) into LanceDB
 - emprez-ai contains a web very basic web-service to receive queries, submit them to ollama and send back the answer. 


scraping 
   contains scripts to retrieve documents from Emprez Documentation.

    To activate the python environment:
        source ./scraping.env/bin/activate

 - All documents can be found in scraping/en or scraping/fr

