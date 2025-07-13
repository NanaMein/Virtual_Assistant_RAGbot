import asyncio
from typing import Optional

from cachetools import TTLCache
from groq.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam
)
from datetime import datetime, timezone, timedelta
import pytz

class GroqChatCache:

    def __init__(self, input_user_id:str):
        self.user_id:str = input_user_id



    def _core_memory_cache(self, memory_cache_id: str):

        print(f"The current user id: {memory_cache_id}")
        try:
            new_mem = chat_groq_cache[memory_cache_id]
            print("Old Cache")
        except KeyError:
            print("New Cache")
            chat_groq_cache[memory_cache_id] = []
            new_mem = chat_groq_cache[memory_cache_id]
        return new_mem

    def add_user_to_memory_cache(self, user_input_message: str = ""):
        if not user_input_message:
            memory = self._core_memory_cache(self.user_id)

            user_msg_ready = ChatCompletionUserMessageParam(
                content="*speechless*", role="user"
            )
            memory.append(user_msg_ready)
            return False

        memory1 = self._core_memory_cache(self.user_id)

        user_msg_ready = ChatCompletionUserMessageParam(
            content=user_input_message, role="user"
        )
        memory1.append(user_msg_ready)
        return True

    def add_assistant_to_memory_cache(self, assistant_input_message: str = None):
        if assistant_input_message is None:
            return False
        memory = self._core_memory_cache(self.user_id)

        assist_msg_ready = ChatCompletionAssistantMessageParam(
            content=assistant_input_message, role="assistant"
        )
        memory.append(assist_msg_ready)
        return True

    def get_chat_history_from_memory_cache(self):
        orig_list = self._core_memory_cache(self.user_id)
        new_list = [] + orig_list
        return new_list

    def get_chat_with_system_prompt(self, system_prompt_template: str = ""):

        system_prompt = ChatCompletionSystemMessageParam(
            content=system_prompt_template, role="system"
        )

        orig_chat = self._core_memory_cache(self.user_id)

        system_and_chat = [system_prompt] + orig_chat
        return system_and_chat

    def delete_old_message_per_call(self):
        try:
            memory = self._core_memory_cache(self.user_id)
            memory.pop(0)
            return True
        except Exception as e:
            print(type(e))
            return False

    def auto_delete_with_limiter(self, limit_to_hold: int):
        memory = self._core_memory_cache(self.user_id)
        while len(memory) > limit_to_hold:
            memory.pop(0)
        return len(memory)

    @staticmethod
    def real_time_indicator():
        return datetime.now(pytz.timezone('Asia/Manila')).strftime('%Y-%m-%d %H:%M:%S')



def ttl_in_hours(how_many_hour: float):
    ttl = 3600 * how_many_hour
    return ttl

def testing_groq_cache():
    test_groq_cache = TTLCache(maxsize=100, ttl=ttl_in_hours(.5))
    cache_lock = asyncio.Lock()

    return test_groq_cache, cache_lock

chat_groq_cache, cache_lock = testing_groq_cache()