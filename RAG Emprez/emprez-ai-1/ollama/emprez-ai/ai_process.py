import ollama
import lancedb



class EmprezAIProcessor:
    def __init__(self, lanceDBPath):
        self.database = lancedb.connect(lanceDBPath)
        self.articleTable = self.database.open_table("articles")
        self.limitResultNumber = 5

    def process(self,question):
        # Semantic Search
        result = self.articleTable.search(question).limit(self.limitResultNumber).to_list()
        context = [r["content"] for r in result]   #content is the field in the articles
        links = []
        uniqueTitle = []
        for r in result:
            if (r["title"] not in uniqueTitle):
                links.append({'title': r["title"],'url': r["url"]})
                uniqueTitle.append(r["title"])
        # Context Prompt
        base_prompt = """You are an AI assistant. Your task is to understand the user question, and provide an answer using the provided contexts. Every answer you generate should have citations in this pattern  "Answer [position].", for example: "Earth is round [1][2].," if it's relevant.
        Your answers are correct, high-quality, and written by an domain expert. 
        If the provided context does not contain the answer, simply state, "The provided context does not have the answer."
        The answer should be less than 300 words.

        User question: {}

        Contexts:
        {}
        """

        # llm
        prompt = f"{base_prompt.format(question, context)}"
        #ollama.generate
        response = ollama.chat(
            model="emprez",
            #stream=True
            #tools=[add_two_numbers]
            messages=[
                {
                    "role": "system",
                    "content": prompt,            
                },
            ],
        )
        print('**** Response ****')
        return response["message"]["content"], links
'''
print('**** Response ****')
print(response["message"]["content"])
print('**** Links ****')
for link in links:
    print(f'{link['title']} : {link['url']}')
'''
