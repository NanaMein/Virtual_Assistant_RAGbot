import asyncio
import os
from typing import Optional
from groq.types.chat import ChatCompletionMessage
from Chatbot_Workflow.Groq.groq_chat_cache import GroqChatCache
from groq import (
AsyncGroq, Groq,
APIError, GroqError, ConflictError,
NotFoundError, APIStatusError, RateLimitError,
APITimeoutError, BadRequestError, APIConnectionError,
AuthenticationError, InternalServerError,
PermissionDeniedError, UnprocessableEntityError,
APIResponseValidationError
)
from groq.types.chat import (
ChatCompletionUserMessageParam,
ChatCompletionAssistantMessageParam,
ChatCompletionSystemMessageParam
)
from groq.types.chat.chat_completion_tool_param import ChatCompletionToolParam
import groq
from dotenv import load_dotenv

load_dotenv()


class GroqChatbotCompletions:


    def __init__(self, input_user_id: str):
        self.user_id: str = input_user_id
        self._groq_cache: GroqChatCache | None = None
        self.client = AsyncGroq(api_key=os.getenv('CLIENT_GROQ_API_1'))


    @property
    def memory_cache(self) -> GroqChatCache:
        if self._groq_cache is None:
            self._groq_cache = GroqChatCache(input_user_id=self.user_id)
        return self._groq_cache


    async def llama_4_scout_chatbot_with_memory(self, input_message: str) -> str | None:
        add_msg_result = await self.memory_cache.add_user_message_to_chat(input_message)
        if not add_msg_result.ok:
            return None

        get_msg_result = await self.memory_cache.get_all_messages()
        if not get_msg_result.ok:
            return None

        _messages = get_msg_result.data
        try:
            comp = await self.client.chat.completions.create(
                # model="qwen/qwen3-32b",
                messages=_messages,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=.7,
                max_completion_tokens=8000,
                top_p=0.95,
                stream=False,
                stop=None,
            )
            groq_object = comp.choices[0].message
            assistant_message = groq_object.content
            assistant_result = await self.memory_cache.add_assistant_message_to_chat(assistant_message)
            if assistant_result.ok:
                return assistant_message
            else:
                return None

        except (APIError, GroqError, ConflictError,
            NotFoundError, APIStatusError, RateLimitError,
            APITimeoutError, BadRequestError, APIConnectionError,
            AuthenticationError, InternalServerError,
            PermissionDeniedError, UnprocessableEntityError,
            APIResponseValidationError
        ) as groq_error:
            return None

    async def llama_4_scout_chatbot(self, input_message: str) -> str | None:

        try:
            messages = [ChatCompletionUserMessageParam(role="user", content=input_message)]
            comp = await self.client.chat.completions.create(
                messages=messages,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=.7,
                max_completion_tokens=8000,
                top_p=0.95,
                stream=False,
                stop=None,
            )
            groq_object = comp.choices[0].message
            assistant_message = groq_object.content
            return assistant_message

        except (
            APIError, GroqError, ConflictError,
            NotFoundError, APIStatusError, RateLimitError,
            APITimeoutError, BadRequestError, APIConnectionError,
            AuthenticationError, InternalServerError,
            PermissionDeniedError, UnprocessableEntityError,
            APIResponseValidationError
        ) as groq_error:
            print(f"Error: {groq_error}\nError Type: {type(groq_error)}")
            return None


    async def qwen_3_32b_chatbot_with_memory(self, input_message: str) -> str | None:
        add_msg_result = await self.memory_cache.add_user_message_to_chat(input_message)
        if not add_msg_result.ok:
            return None

        get_msg_result = await self.memory_cache.get_all_messages()
        if not get_msg_result.ok:
            return None
        _messages = get_msg_result.data

        # self.memory_cache.add_user_to_memory_cache(user_input_message=input_message)
        # messages = self.memory_cache.get_chat_history_from_memory_cache()
        try:
            comp = await self.client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=_messages,
                temperature=0.7,
                max_completion_tokens=10351,
                top_p=0.95,
                reasoning_effort="default",
                reasoning_format="parsed",
                stream=False,
                stop=None,
            )
            message_object = comp.choices[0].message
            assistant_message = message_object.content
            chat_result = await self.memory_cache.add_assistant_message_to_chat(assistant_message)
            if chat_result.ok:
                return assistant_message
            else:
                return None

        except (
            APIError, GroqError, ConflictError,
            NotFoundError, APIStatusError, RateLimitError,
            APITimeoutError, BadRequestError, APIConnectionError,
            AuthenticationError, InternalServerError,
            PermissionDeniedError, UnprocessableEntityError,
            APIResponseValidationError
            ) as groq_error:
            return None

    async def gpt_oss_20b_chatbot(self, input_message: str) -> str | None:
        try:
            messages = [ChatCompletionUserMessageParam(role="user", content=input_message)]
            tools = [ChatCompletionToolParam(type="browser_search")]
            comp = await self.client.chat.completions.create(
                messages=messages,
                model="openai/gpt-oss-20b",
                temperature=.4,
                max_completion_tokens=20000,
                top_p=0.95,
                stream=False,
                stop=None,
                reasoning_effort="medium",
                tools=tools
            )
            groq_object = comp.choices[0].message
            assistant_message = groq_object.content
            return assistant_message

        except (APIError, GroqError, ConflictError,
            NotFoundError, APIStatusError, RateLimitError,
            APITimeoutError, BadRequestError, APIConnectionError,
            AuthenticationError, InternalServerError,
            PermissionDeniedError, UnprocessableEntityError,
            APIResponseValidationError
        ) as groq_error:
            return None

    # async def groq_llama_4_scout(
    #         self, content: str,
    #         role: str ,
    #         mixed_messages: list[dict[str,str]] | None = None,
    #         temperature: float = 0.7,
    #         max_completion_tokens: int = 8000
    # )-> ChatCompletionMessage:
    #     """model is: meta-llama/llama-4-scout-17b-16e-instruct
    #
    #     role: user, assistant, system, hybrid
    #
    #     """
    #     if role in( "user" , "assistant" , "system"):
    #         messages = [{"role":role, "content":content}]
    #     elif role == "hybrid" and mixed_messages:
    #
    #         messages = mixed_messages
    #     else:
    #         raise ValueError("When role is 'hybrid', mixed_messages must be provided.")
    #
    #     client = AsyncGroq(api_key=os.getenv('CLIENT_GROQ_API_1'))
    #
    #     model = "meta-llama/llama-4-scout-17b-16e-instruct"
    #
    #     comp = await client.chat.completions.create(
    #         # model="qwen/qwen3-32b",
    #         messages=messages,
    #         model=model,
    #         temperature=temperature,
    #         max_completion_tokens=max_completion_tokens,
    #         top_p=0.95,
    #         stream=False,
    #         stop=None,
    #     )
    #     output = comp.choices[0].message
    #     print("Output_model_dump_is: ")
    #     print(output.model_dump())
    #     return output
#
# print(os.getenv('CLIENT_GROQ_API_1'))
# obj = GroqChatbotCompletions("what")
# try:
#     input =  "Nice to meet you"
#     result_ = asyncio.run(obj.groq_llama_4_scout(content="hello", role="user"))
#     x = result_.model_dump(include={"role", "content"})
#     print(f"test_model_dump: {x}\ntest_what_type: {type(x)}")
#     result = result_.model_dump(mode="json",exclude_none=True, by_alias=True)
# except groq.GroqError as ge:
#     input = "The error is"
#     result = str(type(ge)) + f"Explain of error: {ge}"
#     print("Second error catch")
#
# except Exception as e:
#     input = "The error is"
#     result = str(type(e)) + f"Explain of error: {e}"
#     print("Third error catch")
#
#
# print(input)
# print(result)
# print(type(result))
# str_na = str(result)
# print(type(str_na))
# print(str_na)
# from pymilvus import MilvusClient, MilvusException

# client = MilvusClient(
#     uri="https://in05-0c3198d45816662.serverless.gcp-us-west1.cloud.zilliz.com",
#     token="288fc24c7b8c1e273f4f36675230b66dedefd981bad3c16f13f644d2de27cdcc2135313251b1b086951e55387da3dea30c10a0c8"
# )
# client.alter_collection_properties(
#     collection_name="Validation_User_ID_Collection",
#     properties={"collection.ttl.seconds": 300}
# )

print("Testing run")
test_object = GroqChatbotCompletions(input_user_id="testing")
COUNTER: int = 0
while 5 > COUNTER:
    input_test = input("\n\nWhat do you want to ask? \n")
    test_await = asyncio.run(test_object.gpt_oss_20b_chatbot(input_test))
    print(test_await)
    print(f"COUNTER INDICATOR {COUNTER}")
    COUNTER = COUNTER+1