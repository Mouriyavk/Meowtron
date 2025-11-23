from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch

class Emotion:
    def __init__(self, model_path="models/distilbert_emotion_model", tokenizer_path="models/distilbert_emotion_tokenizer"):
           
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.labels = self.model.config.id2label
    def dectectEmo(self,text):
        input = self.tokenizer(text,return_tensors="pt")
        with torch.no_grad():
            output=self.model(**input)
            probs=F.softmax(output.logits,dim=-1)
            
        emotions = {}
        for i in range(len(self.labels)):
            label = self.labels[i]
            prob = probs[0][i].item()
            emotions[label]=prob

        return emotions

    def emoInStr(self,emotion: dict,top: int = None) -> str:

        sortedEmo = dict(sorted(emotion.items(), key=lambda x: x[1], reverse=True))

        if top is not None:
            sortedEmo = dict(list(sortedEmo.items())[:top])

        emotion_str = ", ".join([f"{label} ({prob:.2f})" for label, prob in sortedEmo.items()])
        return emotion_str
