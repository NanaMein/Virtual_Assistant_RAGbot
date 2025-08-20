import asyncio
import os
from typing import Optional, Any
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
ChatCompletionSystemMessageParam,
ChatCompletionMessageParam
)
from groq.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
import groq
from dotenv import load_dotenv

load_dotenv()

class InputMessageValidator(BaseModel):
    message: str

class InputParamsValidation(BaseModel):
    user: Optional[ChatCompletionUserMessageParam] = None
    system: Optional[ChatCompletionSystemMessageParam] = None

class InputAllMessageParamsValidation(BaseModel):
    messages: list[ChatCompletionMessageParam]



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

    async def llama_4_scout_chatbot(
            self,
            input_user_message: str,
            input_system_message: Any = None,
            input_all_messages: Any = None
    ) -> str | None:

        try:
            _user_message = InputMessageValidator(input_message=input_user_message)
            _system_message = InputMessageValidator(input_message=input_system_message)

            user_msg = ChatCompletionUserMessageParam(role="user", content=input_user_message)
            sys_msg = ChatCompletionSystemMessageParam(role="system", content=input_system_message)
            validated_input = InputParamsValidation(
                user=user_msg, system=sys_msg
            )
            if not input_system_message:
                input_messages = [ sys_msg ] + [ user_msg ]

            elif not input_system_message:
                input_messages = input_all_messages


            comp = await self.client.chat.completions.create(
                messages=input_messages,
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
            print(f"Groq Error: {groq_error}\nError Type: {type(groq_error)}")
            return None

        except ValidationError as ve:
            print(f"Pydantic Error: {ve}\nError Type: {type(ve)}")
            return None


    async def qwen_3_32b_chatbot_with_memory(self, input_message: str, system_message: str = None) -> str | None:
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


    async def llama_4_scout_chatbot_TESTING_VERSION1(
            self,
            input_user_message: str = None,
            input_system_message: str = None,
            input_all_messages: Any = None
    ) -> str | None:

        try:

            _user_message = ChatCompletionUserMessageParam(role="user", content=input_user_message)

            if input_all_messages:
                _messages = input_all_messages

            else:
               if input_system_message:
                   _system_message = ChatCompletionSystemMessageParam(role="system", content=input_system_message)
                   _messages = [_system_message] + [_user_message]

               else:
                   _messages = [_user_message]

            _all = InputAllMessageParamsValidation(messages=_messages)

            comp = await self.client.chat.completions.create(
                messages=_all.messages,
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
            print(f"Groq Error: {groq_error}\nError Type: {type(groq_error)}")
            return None

        except ValidationError as ve:
            print(f"Pydantic Error: {ve}\nError Type: {type(ve)}")
            return None


    def input_messages(
            self,
            input_user_message: str = None,
            input_system_message: str = None,
            input_all_messages: Any = None
    ):

        #
        # class InputMessageValidator(BaseModel):
        #     input_message: str
        #
        # class InputParamsValidation(BaseModel):
        #     user: Optional[ChatCompletionUserMessageParam] = None
        #     system: Optional[ChatCompletionSystemMessageParam] = None
        #
        # class InputAllMessageParamsValidation(BaseModel):
        #     messages: list[ChatCompletionMessageParam]


        if input_all_messages:
            try:
                checked= InputAllMessageParamsValidation(messages=input_all_messages)
                return checked.messages
            except ValidationError:
                return None

        if input_user_message:
            _user = ChatCompletionUserMessageParam(role="user", content=input_user_message)
            _system = ChatCompletionSystemMessageParam(role='system', content=input_system_message)
            if input_system_message:
                valid_list = [_system] + [_user]
                try:
                    with_system_prompt = InputAllMessageParamsValidation(messages=valid_list)
                    return with_system_prompt.messages
                except ValidationError:
                    return None

            else:
                valid_list = [_user]
                try:
                    without_system_prompt = InputAllMessageParamsValidation(messages=valid_list)
                    return without_system_prompt.messages
                except ValidationError:
                    return None
        return None

    def message_validator(
            self,
            input_user_message: Any = None,
            input_system_message: Any = None,
            input_all_messages: Any = None
    ):

        #
        # class InputMessageValidator(BaseModel):
        #     input_message: str
        #
        # class InputParamsValidation(BaseModel):
        #     user: Optional[ChatCompletionUserMessageParam] = None
        #     system: Optional[ChatCompletionSystemMessageParam] = None
        #
        # class InputAllMessageParamsValidation(BaseModel):
        #     messages: list[ChatCompletionMessageParam]

        try:
            input_user = InputMessageValidator(message=input_user_message)
            input_system = InputMessageValidator(message=input_system_message)

        except ValidationError:
            return  None


        if input_all_messages:
            try:
                checked = InputAllMessageParamsValidation(messages=input_all_messages)
                return checked.messages
            except ValidationError:
                return None
        try:
            _user = ChatCompletionUserMessageParam(role="user", content=input_user.message)
            _system = ChatCompletionSystemMessageParam(role='system', content=input_system.message)
            validated_ = InputParamsValidation(user=_user, system=_system)

            if input_user_message:

                if input_system_message:
                    message_list = [validated_.system, validated_.user]
                else:
                    message_list = [validated_.user]

        except ValidationError:
            return None



        if input_user_message:
            _user = ChatCompletionUserMessageParam(role="user", content=input_user_message)
            _system = ChatCompletionSystemMessageParam(role='system', content=input_system_message)
            if input_system_message:
                valid_list = [_system] + [_user]
                try:
                    with_system_prompt = InputAllMessageParamsValidation(messages=valid_list)
                    return with_system_prompt.messages
                except ValidationError:
                    return None

            else:
                valid_list = [_user]
                try:
                    without_system_prompt = InputAllMessageParamsValidation(messages=valid_list)
                    return without_system_prompt.messages
                except ValidationError:
                    return None
        return None

    async def llama_4_scout_chatbot_TESTING_VERSION2(
            self,
            input_user_message: str = None,
            input_system_message: str = None,
            input_all_messages: Any = None
    ) -> str | None:

        try:
            validated_messages = self.input_messages(
                input_user_message=input_user_message,
                input_system_message=input_system_message,
                input_all_messages=input_all_messages
            )
            if not validated_messages:
                return None

            comp = await self.client.chat.completions.create(
                messages=validated_messages,
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
            print(f"Groq Error: {groq_error}\nError Type: {type(groq_error)}")
            return None

        except ValidationError as ve:
            print(f"Pydantic Error: {ve}\nError Type: {type(ve)}")
            return None
