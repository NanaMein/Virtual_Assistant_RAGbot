import asyncio
from typing import TypeVar, Generic, Optional

from cachetools import TTLCache
from groq.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam
)
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import pytz
from dotenv import load_dotenv


load_dotenv()



def ttl_in_hours(how_many_hour: float):
    ttl = 3600 * how_many_hour
    return ttl

T = TypeVar("T")


@dataclass(frozen=True)
class MessageResult(Generic[T]):
    ok: bool
    data: Optional[T] = None
    error_description: str | Exception = None

    @property
    def error_result(self):
        return f"""
        Common Error Appeared: An Error occurred in Groq Layer(groq-chat-cache) \n
        Error Description or Traceback: {self.error_description}
        """

class GroqChatCache:

    _cache = TTLCache(maxsize=100, ttl=ttl_in_hours(.5))

    def __init__(self, input_user_id:str):
        self.user_id:str = input_user_id
        self._lock = asyncio.Lock()



    async def _core_memory_cache(self, memory_cache_id: str) -> list:
        async with self._lock:
            cache_obj = self._cache.get(memory_cache_id)
            if cache_obj is not None:
                return cache_obj

            else:
                self._cache.pop(memory_cache_id,None)
                self._cache.expire()
                self._cache[memory_cache_id] = []
                return self._cache[memory_cache_id]


    async def add_user_message_to_chat(self, user_input_message: str = "") -> MessageResult:
        try:
            memory = await self._core_memory_cache(self.user_id)

            user_msg_ready = ChatCompletionUserMessageParam(
                content=user_input_message, role="user"
            )
            memory.append(user_msg_ready)
            return MessageResult(ok=True)
        except (KeyError, ValueError, AttributeError, TypeError) as norm_err:
            return MessageResult(ok=False, error_description=norm_err)
        except Exception as ex:
            return MessageResult(ok=False, error_description=ex)

    async def add_assistant_message_to_chat(self, assistant_input_message: str = "") -> MessageResult:
        try:
            memory = await self._core_memory_cache(self.user_id)

            assist_msg_ready = ChatCompletionAssistantMessageParam(
                content=assistant_input_message, role="assistant"
            )
            memory.append(assist_msg_ready)
            return MessageResult(ok=True)
        except (KeyError, ValueError, AttributeError, TypeError) as norm_err:
            return MessageResult(ok=False, error_description=norm_err)
        except Exception as ex:
            return MessageResult(ok=False, error_description=ex)

    async def get_all_messages(self):
        try:
            orig_list = await self._core_memory_cache(self.user_id)
            new_list = [] + orig_list
            # return new_list
            return MessageResult(ok=True, data=new_list)
        except Exception as ex:
            return MessageResult(ok=False, error_description=ex)

    async def get_all_with_system_prompt(self, system_prompt_template: str)-> MessageResult:
        try:
            sys_prompt = system_prompt_template + f"""\n\n
            ###CurrentDateTime: {self.real_time_indicator()}"""
            system_prompt = ChatCompletionSystemMessageParam(
                content=sys_prompt, role="system"
            )

            orig_chat = await self._core_memory_cache(self.user_id)

            system_and_chat = [system_prompt] + orig_chat
            return MessageResult(ok=True, data=system_and_chat)
        except Exception as ex:
            return MessageResult(ok=False, error_description=ex)

    async def setting_limit_to_messages(self, conversation_turn: int) -> MessageResult:
        try:
            memory = await self._core_memory_cache(self.user_id)
            while len(memory) > conversation_turn:
                memory.pop(0)
            # return len(memory)
            return MessageResult(ok=True, data=len(memory))
        except Exception as ex:
            return MessageResult(ok=False, error_description=ex)

    @staticmethod
    def real_time_indicator() -> str:
        return datetime.now(pytz.timezone('Asia/Manila')).strftime('%Y-%m-%d %H:%M:%S')
