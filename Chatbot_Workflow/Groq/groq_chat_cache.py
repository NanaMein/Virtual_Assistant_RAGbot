from cachetools import TTLCache
from groq.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam
)
from datetime import datetime, timezone, timedelta
import pytz

def ttl_in_hours(how_many_hour: float):
    ttl = 3600 * how_many_hour
    return ttl

class GroqChatCache:

    groq_cache = TTLCache(maxsize=100, ttl=ttl_in_hours(.5))

    def __init__(self, input_user_id:str):
        self.user_id:str = input_user_id


    def _core_memory_cache(self, memory_cache_id):
        try:
            memory = self.groq_cache[memory_cache_id]
        except KeyError:
            memory = []
            self.groq_cache[memory_cache_id] = memory
        return memory

    def add_user_to_memory_cache(self, user_input_message: str = None):
        if user_input_message is None:
            return False

        memory = self._core_memory_cache(self.user_id)
        ph_time = datetime.now(pytz.timezone('Asia/Manila')).strftime('%Y-%m-%d %H:%M:%S')

        msg_with_time = f"""<TimeStamp({ph_time})>
        [content={user_input_message}]
        </TimeStamp({ph_time})>"""

        user_msg_ready = ChatCompletionUserMessageParam(
            content=msg_with_time, role="user"
        )

        memory.append(user_msg_ready)
        return True

    def add_assistant_to_memory_cache(self, assistant_input_message: str = None):
        if assistant_input_message is None:
            return False
        memory = self._core_memory_cache(self.user_id)

        ph_time = datetime.now(pytz.timezone('Asia/Manila')).strftime('%Y-%m-%d %H:%M:%S')

        msg_with_time = f"""<TimeStamp({ph_time})>
        [content={assistant_input_message}]
        </TimeStamp({ph_time})>"""

        assist_msg_ready = ChatCompletionAssistantMessageParam(
            content=msg_with_time, role="assistant"
        )

        memory.append(assist_msg_ready)
        return True

    def get_chat_history_from_memory_cache(self):
        orig_list = self._core_memory_cache(self.user_id)
        new_list = [] + orig_list
        return new_list

    def get_chat_with_system_prompt(self, system_prompt: str):
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




