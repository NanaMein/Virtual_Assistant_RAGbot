import asyncio
import os
from typing import Optional

from groq.types.chat import ChatCompletionMessage
from langchain.chains.question_answering.map_reduce_prompt import messages

from Chatbot_Workflow.Groq.groq_chat_cache import GroqChatCache
from groq import AsyncGroq, GroqError
import groq
from dotenv import load_dotenv

load_dotenv()

class GroqChatbotCompletions:


    def __init__(self, input_user_id: str = ""):
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

    async def reasoning_llm_qwen3_32b(self,
            user_input: str,
            sys_prompt_tmpl: str = "",
            reasoning: bool = False
        ) -> str:

        memory = self.memory_cache

        memory.add_user_to_memory_cache(user_input)
        if not sys_prompt_tmpl:
            system_prompt = ""
        else:
            system_prompt = sys_prompt_tmpl

        messages = memory.get_chat_with_system_prompt(system_prompt_template=system_prompt)

        if reasoning:
            reasoning_effort = "default"
            temperature = 0.6
            top_p = 0.95
        else:
            reasoning_effort = "none"
            temperature = 0.7
            top_p = 0.8

        client = AsyncGroq(api_key=os.getenv('CLIENT_GROQ_API_1'))
        comp = await client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages,
            temperature=temperature,
            max_completion_tokens=10351,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            reasoning_format="hidden",
            stream=False,
            stop=None,
        )
        msg = comp.choices[0].message
        msg_content = msg.content
        memory.add_assistant_to_memory_cache(msg_content)
        memory.auto_delete_with_limiter(10)

        return msg_content


    async def groq_llama_4_scout(
            self, content: str,
            role: str ,
            mixed_messages: list[dict[str,str]] | None = None,
            temperature: float = 0.7,
            max_completion_tokens: int = 8000
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

print(os.getenv('CLIENT_GROQ_API_1'))
obj = GroqChatbotCompletions("what")
try:
    input =  "Nice to meet you"
    result_ = asyncio.run(obj.groq_llama_4_scout(content="hello", role="user"))
    x = result_.model_dump(include={"role", "content"})
    print(f"test_model_dump: {x}\ntest_what_type: {type(x)}")
    result = result_.model_dump(mode="json",exclude_none=True, by_alias=True)
except groq.GroqError as ge:
    input = "The error is"
    result = str(type(ge)) + f"Explain of error: {ge}"
    print("Second error catch")

except Exception as e:
    input = "The error is"
    result = str(type(e)) + f"Explain of error: {e}"
    print("Third error catch")


print(input)
print(result)
print(type(result))
str_na = str(result)
print(type(str_na))
print(str_na)
from pymilvus import MilvusClient

# client = MilvusClient(
#     uri="https://in05-0c3198d45816662.serverless.gcp-us-west1.cloud.zilliz.com",
#     token="288fc24c7b8c1e273f4f36675230b66dedefd981bad3c16f13f644d2de27cdcc2135313251b1b086951e55387da3dea30c10a0c8"
# )
# client.alter_collection_properties(
#     collection_name="Validation_User_ID_Collection",
#     properties={"collection.ttl.seconds": 300}
# )