import os
from typing import Optional
from groq_chat_cache import GroqChatCache
from groq import AsyncGroq


class GroqChatbotCompletions:


    def __init__(self, input_user_id):
        self.user_id: str = input_user_id
        self._groq_cache: Optional[GroqChatCache] = None


    @property
    def memory_cache(self) -> GroqChatCache:
        if self._groq_cache is None:
            self._groq_cache = GroqChatCache(input_user_id=self.user_id)
        return self._groq_cache

    async def qwen_3_32b_chatbot_with_memory(self, input_message: str):
        self.memory_cache.add_user_to_memory_cache(user_input_message=input_message)
        messages = self.memory_cache.get_chat_history_from_memory_cache()
        client = AsyncGroq(api_key=os.getenv('CLIENT_GROQ_API_1'))
        comp = await client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages,
            temperature=0.7,
            max_completion_tokens=10351,
            top_p=0.95,
            reasoning_effort="default",
            reasoning_format="parsed",
            stream=False,
            stop=None,
        )
        msg=await comp.choices[0].message
        m = await comp.choices[0]

        self.memory_cache.add_assistant_to_memory_cache(assistant_input_message=msg.content)
        return msg.content
