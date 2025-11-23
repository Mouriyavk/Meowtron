import requests

class LlmAsk:
    def __init__(self,ollamaurl="http://localhost:11434/api/generate"):
        self.ollamaurl = ollamaurl

    def ask(self,promt,model ="llama3.2:1b"):
        data={
            "model":model,
            "prompt":promt,
            "stream":False
            }
        res=requests.post(self.ollamaurl, json=data)
        res.raise_for_status()
        response_data = res.json()

        return response_data.get('response', 'No response field found')

