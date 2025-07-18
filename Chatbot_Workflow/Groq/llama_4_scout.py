import asyncio
import os
from typing import Optional, Union

from groq.types.chat import ChatCompletionMessage
from langchain.chains.question_answering.map_reduce_prompt import messages

from Chatbot_Workflow.Groq.groq_chat_cache import GroqChatCache
from groq import AsyncGroq, GroqError
import groq
from dotenv import load_dotenv

load_dotenv()

class LlamaScoutGroqChatCompletions:


    # def __init__(self, input_user_id: str = ""):
    #     self.user_id: str = input_user_id
    #     self._groq_cache: Optional[GroqChatCache] = None
    #
    # @property
    # def memory_cache(self) -> GroqChatCache:
    #     if self._groq_cache is None:
    #         self._groq_cache = GroqChatCache(input_user_id=self.user_id)
    #     return self._groq_cache


    async def groq_llama_4_scout_all_features(
            self, content: str,
            role: str ,
            mixed_messages: list[dict[str,str]] | None = None,
            temperature: float = 0.7,
            max_completion_tokens: int = 8000,
    )-> ChatCompletionMessage:
        """model is: meta-llama/llama-4-scout-17b-16e-instruct

        role: user, assistant, system, hybrid

        """
        if role in( "user" , "assistant" , "system"):
            messages = [{"role":role, "content":content}]
        elif role == "hybrid" and mixed_messages:

            messages = mixed_messages
        else:
            raise ValueError("When role is 'hybrid', mixed_messages must be provided.")

        client = AsyncGroq(api_key=os.getenv('CLIENT_GROQ_API_1'))

        model = "meta-llama/llama-4-scout-17b-16e-instruct"

        comp = await client.chat.completions.create(
            # model="qwen/qwen3-32b",
            messages=messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        output = comp.choices[0].message
        print("Output_model_dump_is: ")
        print(output.model_dump())
        return output

    async def groq_llama_4_scout_user_only(
            self, content: Union[list[dict[str,str]] , str ],
    )-> ChatCompletionMessage:

        if isinstance(content, str):
            user_messages = [{"role": "user", "content": content}]
        elif isinstance(content, list):
            user_messages = content
        else:
            raise ValueError("Missing required content: pass a str or list of message dicts.")

        client = AsyncGroq(api_key=os.getenv('CLIENT_GROQ_API_1'))

        model = "meta-llama/llama-4-scout-17b-16e-instruct"
        try:
            comp = await client.chat.completions.create(
                # model="qwen/qwen3-32b",
                messages=user_messages,
                model=model,
                temperature=.7,
                max_completion_tokens=8000,
                top_p=0.95,
                stream=False,
                stop=None,
            )
        except GroqError:
            raise GroqError("error po putangina")
        output = comp.choices[0].message
        print("Output_model_dump_is: ")
        print(">>", output.model_dump(), "<<")
        return output

    async def groq_llama_4_scout_with_system(
            self, user_with_system_content: Union[list[dict[str,str]] , str ],
    )-> ChatCompletionMessage:

        if isinstance(user_with_system_content, list):
            user_messages = user_with_system_content
        else:
            raise ValueError("Missing required content: pass a list of message dicts.")

        client = AsyncGroq(api_key=os.getenv('CLIENT_GROQ_API_1'))

        model = "meta-llama/llama-4-scout-17b-16e-instruct"
        try:
            comp = await client.chat.completions.create(
                # model="qwen/qwen3-32b",
                messages=user_messages,
                model=model,
                temperature=.7,
                max_completion_tokens=8000,
                top_p=0.95,
                stream=False,
                stop=None,
            )
        except GroqError:
            raise GroqError("error po putangina")
        output = comp.choices[0].message
        print("Output_model_dump_is: ")
        print(">>", output.model_dump(), "<<")
        return output
