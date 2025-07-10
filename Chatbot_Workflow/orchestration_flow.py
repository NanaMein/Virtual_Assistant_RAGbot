# import litellm
# litellm._turn_on_debug()
import asyncio
from collections import deque
from functools import lru_cache
from typing import Deque, Type, Optional
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


class OrchestrationState(BaseModel):
    user_input_message: str = ""
    user_input_id: str = ""


class OrchestrationFlow(Flow[OrchestrationState]):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @start()
    async def initial_assessment(self):
        
        await asyncio.sleep(2)


    @listen(initial_assessment)
    async def listen1(self):
        await asyncio.sleep(2)
        return "Ok"