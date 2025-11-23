class PromptBuild:

    def format_memories_for_prompt(self, mem_results):
        return [item["content"] for item in mem_results]

    def build(self, user_input, emotion_str, usermemory):

        system_desc = "sassy, cat-like, sarcastic"
        response_style = "roast 2-4 lines max"

        # FIXED: use self.method()
        memory = self.format_memories_for_prompt(usermemory)

        # Optional: combine memory into prompt
        memory_block = "\n".join(f"- {m}" for m in memory)

        return (
            f"{system_desc}\n"
            f"{emotion_str}\n"
            f"{response_style}\n"
            f"Memory:\n{memory_block}\n"
            f"User Input: {user_input}"
        )

