import json


class DataProcess:
    def __init__(self, llm):
        self.llm = llm



    def MsgProcess(self, usermsg, topEmo):
        prompt = f'''
    You are a semantic-memory query normalizer.

    Your job is to convert a raw user message into a list of short, generalizable semantic search cues.

    — RULES —
    1. DO NOT answer the question.
    2. DO NOT output full sentences.
    3. Output MUST be a pure JSON array of strings.
    4. Generate 3–8 cues.
    5. Keep each cue short (1–3 words).
    6. Include:
       - high-level abstract concepts
       - category-level concepts
       - specific keywords/entities from the user's message
    7. If the user asks what they like → include cues like:
       "preference", "food preference", "game preference"
    8. Ignore the emotion parameter except for context.

    NOW PROCESS:
    User message: "{usermsg}"
    Top emotion: "{topEmo}"
    Return only a JSON array.
    '''

        # --- Ask LLM ---
        raw = self.llm.ask(prompt)

        # --- Safety: extract JSON array from messy LLM output ---
        try:
            # Try to load directly
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x) for x in data]
        except:
            pass

        # Try to locate JSON array inside the string
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            extracted = raw[start:end]
            data = json.loads(extracted)
            if isinstance(data, list):
                return [str(x) for x in data]
        except:
            pass

        # --- Fallback (never break your system) ---
        # create minimal cues from the message itself
        tokens = [w for w in usermsg.lower().split() if w.isalpha()]
        fallback = list(set(tokens))[:5]
        if not fallback:
            fallback = ["general query"]

        return fallback


    def ExtractFacts(self,userMsg,topEmo):
        prompt=f'''Extract factual information from the user message.

Return ONLY a JSON array of objects like:
[
  {{
    "content": "... factual statement ...",
    "category": "habit | preference | personality | action ......",
    "polarity": "positive | negative | neutral"
  }}
]

Rules:
- DO NOT add emotion here. (It will be added by a classifier)
- Each fact must be short.
- Only include facts, no opinions.
- choose one category based on the content
- choose only one polarity based on the emotion and the content
- If multiple facts exist, create multiple objects.
- No explanation, no text outside JSON.

User message:{userMsg} 
topEmotion:{topEmo}
'''
        raw = self.llm.ask(prompt)
        try:
            fact_list = json.loads(raw)
        except:
            return []
    
        final_facts = []

        for f in fact_list:
            content = f.get("content", "").strip()
            category = f.get("category", "other")
            polarity = f.get("polarity", "neutral")

            if not content:
                continue 


            final_facts.append({
                "content": content,
                "category": category,
                "emotion": topEmo,
                "polarity": polarity
            })

        return final_facts
