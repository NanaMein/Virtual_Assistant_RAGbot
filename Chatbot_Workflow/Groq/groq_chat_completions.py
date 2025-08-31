import os
from typing import Optional, Any
from groq import (
AsyncGroq,
APIError, GroqError, ConflictError,
NotFoundError, APIStatusError, RateLimitError,
APITimeoutError, BadRequestError, APIConnectionError,
AuthenticationError, InternalServerError,
PermissionDeniedError, UnprocessableEntityError,
APIResponseValidationError
)
from groq.types.chat import (
ChatCompletionUserMessageParam,
ChatCompletionSystemMessageParam,
ChatCompletionMessageParam
)
from groq.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()

class InputParamsValidation(BaseModel):
    user: Optional[ChatCompletionUserMessageParam] = None
    system: Optional[ChatCompletionSystemMessageParam] = None

class InputAllMessageParamsValidation(BaseModel):
    messages: list[ChatCompletionMessageParam]



class GroqChatbotCompletions:


    def __init__(self, input_user_id: str):
        self.user_id: str = input_user_id
        self.client = AsyncGroq(api_key=os.getenv('CLIENT_GROQ_API_1'))

    def input_messages(
            self,
            input_user_message: Any = None,
            input_system_message: Any = None,
            input_all_messages: Any = None
    ) -> Optional[list[ChatCompletionMessageParam]]:

        try:
            if input_all_messages:
                try:
                    checked= InputAllMessageParamsValidation(messages=input_all_messages)
                    return checked.messages
                except ValidationError:
                    return None

            if input_user_message:
                # _object = ParamsObject(user=input_user_message, system=input_system_message)
                _user = ChatCompletionUserMessageParam(role="user", content=input_user_message)

                if input_system_message:
                    _system = ChatCompletionSystemMessageParam(role='system', content=input_system_message)

                    # valid_list = [_system] + [_user]
                    valid_list = [_system, _user]
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


        except Exception as ex:
            print(f"Unexpected Error occurred: {ex}")
            return None

    async def qwen_3_32b_chatbot(
            self,
            input_user_message: Any = None,
            input_system_message: Any = None,
            input_all_messages: Any = None
    ) -> str | None:

        try:
            validated_messages = self.input_messages(
                input_user_message=input_user_message,
                input_system_message=input_system_message,
                input_all_messages=input_all_messages
            )

            if validated_messages is None:
                return None

            comp = await self.client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=validated_messages,
                temperature=0.7,
                max_completion_tokens=10351,
                top_p=0.95,
                reasoning_effort="default",
                reasoning_format="parsed",
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

        except Exception as ex:
            print(f"Unexpected Error: {ex}")
            return None

    async def gpt_oss_20b_chatbot(
            self,
            input_user_message: Any = None,
            input_system_message: Any = None,
            input_all_messages: Any = None
    ) -> str | None:

        try:
            validated_message = self.input_messages(
                input_user_message=input_user_message,
                input_system_message=input_system_message,
                input_all_messages=input_all_messages
            )
            if validated_message is None:
                return None

            comp = await self.client.chat.completions.create(
                messages=validated_message,
                model="openai/gpt-oss-20b",
                temperature=.4,
                max_completion_tokens=20000,
                top_p=0.95,
                stream=False,
                stop=None,
                reasoning_effort="medium",
                tools=[ChatCompletionToolParam(type="browser_search")]
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
            print(f"Groq Error: {groq_error}\nError Type: {type(groq_error)}")
            return None

        except Exception as ex:
            print(f"Unexpected Error: {ex}")
            return None

    async def llama_4_scout_chatbot(
            self,
            input_user_message: Any = None,
            input_system_message: Any = None,
            input_all_messages: Any = None
    ) -> str | None:

        try:
            validated_messages = self.input_messages(
                input_user_message=input_user_message,
                input_system_message=input_system_message,
                input_all_messages=input_all_messages
            )
            if validated_messages is None:
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

        except Exception as ex:
            print(f"Unexpected Error: {ex}")
            return None
