import ollama
import lancedb



class EmprezAIProcessor:
    def __init__(self, lanceDBPath):
        self.database = lancedb.connect(lanceDBPath)
        self.articleTable = self.database.open_table("articles")
        self.limitResultNumber = 5

    def readMenus(self):
        with open('menu.v2.yaml') as f_in:
            return f_in.read()


    def process(self,question):
        # Semantic Search
        result = self.articleTable.search(question).limit(self.limitResultNumber).to_list()
        context = [r["content"] for r in result]   #content is the field in the articles
        context = '' # dont use the db for now.
        links = []
        uniqueTitle = []
        menus = self.readMenus()
        for r in result:
            if (r["title"] not in uniqueTitle):
                links.append({'title': r["title"],'url': r["url"]})
                uniqueTitle.append(r["title"])
        # Context Prompt
        base_prompt = """You are an AI assistant. Your task is to understand the user question, and provide an answer using the provided contexts. Every answer you generate should have citations in this pattern  "Answer [position].", for example: "Earth is round [1][2].," if it's relevant.
        Your answers are correct, high-quality, and written by an domain expert. 
        If the provided context does not contain the answer, simply state, "The provided context does not have the answer."
        Show all the steps that are leading to your answer.

        The answer should be as concise as possible.

        User question: {0}

        Contexts:
        {1}


        Use this menu "{2}" to guide the user.
        The menu is a text document formatted as followed:
        - Menu1 
          name: Optional Name
          description: description of the content of the menu
          - Sub Menu1
            description: description of the content of Sub Menu1
          - Sub Menu2
            description: description of the content of Sub Menu2

        Use the following format to describe what menu to use:
        Menu1 -> Sub Menu2

        """

        # llm
        prompt = f"{base_prompt.format(question, context,menus)}"
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
