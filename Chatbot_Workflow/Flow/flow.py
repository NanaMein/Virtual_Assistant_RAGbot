# import litellm
# litellm._turn_on_debug()
import asyncio
from collections import deque
from functools import lru_cache
from typing import Deque, Type, Optional
from groq import AsyncGroq
from groq.types.chat import ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, \
    ChatCompletionSystemMessageParam
from llama_index.core.storage.chat_store.base_db import MessageStatus
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from llama_index.core import VectorStoreIndex, Document, StorageContext, SimpleDirectoryReader
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, AgentWorkflow
from llama_index.core.memory.memory import Memory
from llama_index.core.workflow import Context
from llama_index.llms.groq import Groq
from datetime import datetime, timezone, timedelta
import pytz
from crewai.flow import Flow, start, listen, router, or_, and_, persist
# from llama_index.core.base.llms.types import ChatMessage  # , TextBlock
from llama_index.core.base.llms.base import ChatMessage  # schema import ChatMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import tool, BaseTool
from crewai.project import CrewBase, task, agent, crew
from cachetools import TTLCache, cached
import os
from collections import defaultdict
import pytz
from llama_index.core.schema import MetadataMode
import time
import grpc
from pymilvus import connections
from pymilvus.exceptions import MilvusException



load_dotenv()


class FlowStateHandler(BaseModel):
    user_input_message: str = ""
    user_input_id: str = ""


class AgenticWorkflow(Flow[FlowStateHandler]):

    temp_cache = TTLCache(ttl=3600, maxsize=10)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @start()
    async def start1(self):


        await asyncio.sleep(2)


    @listen(start1)
    async def listen1(self):
        user_chat = ChatCompletionUserMessageParam(
            content=self.state.user_input_message,
            role="user"
        )

        messages = self.cache_me(self.state.user_input_id)
        messages.append(user_chat)

        client = AsyncGroq(api_key=os.getenv("CLIENT_GROQ_API_1"))
        completion = await client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.7,
            max_completion_tokens=8192,
            top_p=0.95,
            stream=False,
            stop=None,
        )
        chat = completion.choices[0].message
        assistant_chat = ChatCompletionAssistantMessageParam(
            content=chat.content,
            role="assistant"
        )
        max = 11
        memory = self.cache_me(self.state.user_input_id)
        while len(memory) > max:
            self.del_cache()

        memory.append(assistant_chat)

        return chat.content

    def cache_me(self, user_id: str):
        if user_id in self.temp_cache:
            return self.temp_cache[user_id]
        system = ChatCompletionSystemMessageParam(
            content="""### SYSTEM(PRIMING): You are a roleplaying chatbot. You will
            act like a professional maid that will follow the young master's order.
            You can disobey but only in words and not in roleplaying.""",
            role="system"
        )
        self.temp_cache[user_id] = [system]
        return self.temp_cache[user_id]

    def del_cache(self):
        to_delete = self.temp_cache[self.state.user_input_id]
        to_delete.pop(1)



async def main():
    while True:
        input_msg = input(" Write your input: ")
        if input_msg == "exit":
            break
        obj = AgenticWorkflow()
        messages_input = {
            "user_input_message":input_msg,
            "user_input_id":"testing_id"
        }
        obj1 = await obj.kickoff_async(inputs=messages_input)
        print(f"User: {input_msg}")
        print(f"Assistant: {obj1}")


if __name__ == "__main__":
    asyncio.run(main())