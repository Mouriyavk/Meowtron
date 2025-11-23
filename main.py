from core import dataProcessing
from core.llm import LlmAsk
from core.emotion import Emotion
from core.promtBuilder import PromptBuild
from core.memoryOld import ModelMemory
from core.dataProcessing import DataProcess

def main():

    llm = LlmAsk()
    emo = Emotion()
    prombuilder = PromptBuild()
    memory = ModelMemory()
    dp = DataProcess(llm)

    print("Meowtron is running. Type 'exit' to stop.\n")

    while True:
        print("#----------------------------------------#")
        userMsg = input("You: ")

        if userMsg.lower().strip() == "exit":
            print("Shutting down Meowtron...")
            break


        
        emoProbs = emo.dectectEmo(userMsg)
        emoTop = emo.emoInStr(emoProbs, 1)

        memQuestion = dp.MsgProcess(userMsg,emoTop)
        print(f"MemoryQuery: {memQuestion}")
        
        print("\n")

        related = memory.retrieve(memQuestion)
        print(f"memoryRetrived: {related}")
        
        print("\n")
        
        prompt = prombuilder.build(userMsg, emoTop,related)
        print(f"promt: {prompt}")

        print("\n")

        response = llm.ask(prompt)
        print(f"Meowtron: {response}\n")
        
        print("\n")
       
        facts = dp.ExtractFacts(userMsg,emoTop)
        print(f"FactsExtracted: {facts}")

        memory.addFactsSenmantic(facts)

        print("#----------------------------------------#")


if __name__ == "__main__":
    main()


