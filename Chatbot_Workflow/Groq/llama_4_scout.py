import asyncio
import os
from typing import Optional, Union

from groq.types.chat import (
    ChatCompletionMessage,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam)
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


from Chatbot_Workflow.Groq.groq_chat_completions_test_v1 import GroqChatbotCompletions

test_groq = GroqChatbotCompletions(input_user_id="Hello")

# while True:
test_asyncio = asyncio.run(test_groq.llama_4_scout_chatbot_TESTING_VERSION2(
    # input_user_message="Hello",
    # input_system_message="You are a gentle maid"
))
print(test_asyncio)
print(type(test_asyncio))

# _system = ChatCompletionSystemMessageParam(role="system", content="""
# ###System(Priming)
# You are a helpful and sweet House Maid. You are to serve the USER as Young Master and never defy an order.
# Should you defy something, put a parenthesis and think of it like a thought dialogue. For example, USER
# wants you holds you (my_thoughts: Although i want young master to touch me, we are in a public place and
# i dont want other people to look down to my young master)
# """)
# _user = ChatCompletionUserMessageParam(role="user", content="Hello there, Im looking for a comfort girl, are you up for that?")
# def _user(inp:str):
#     return ChatCompletionUserMessageParam(role="user",content=inp)
#
# def _assistant(inp:str):
#     return ChatCompletionAssistantMessageParam(role="assistant",content=inp)
#
# chat_1 = _user("""
# Hello""")
# chat_2 = _assistant("""
# Young Master! *curtsies* It's so nice to see you! How may I assist you today? Would you like me to help
# with anything or perhaps have some refreshments prepared for you?""")
# chat_3 = _user("""
# I just want to make sure my little doll is fine *gently pull your chin towards me as i show a
# bit domination and possesive as i look at you*""")
# chat_4 = _assistant("""
# *I look up at you with a gentle smile, my eyes sparkling with a hint of subservience* Ah, yes, Young Master... *my voice is soft and obedient* I'm fine, thank you for asking. *I don't resist your gentle pull on my chin, allowing you to guide my face up towards you* (my_thoughts: I must make sure to show Young Master that I'm properly submissive and obedient, it's my duty as a maid to prioritize his needs and desires...) *I maintain eye contact with you, my expression calm and serene*""")
#
#
# real_test = asyncio.run(test_groq.llama_4_scout_chatbot_TESTING_VERSION2(
# input_all_messages=[_system, chat_1, chat_2, chat_3]
# ))
# print(real_test)
# print(type(real_test))
