# import litellm
# litellm._turn_on_debug()
import asyncio
from collections import deque, defaultdict
from functools import lru_cache
from typing import Deque, Type, Optional, Tuple
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
from crewai.flow import Flow, start, listen, router, or_, and_, persist
from llama_index.core.base.llms.base import ChatMessage  # schema import ChatMessage
from pydantic import BaseModel, Field, PrivateAttr
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import tool, BaseTool
from crewai.project import CrewBase, task, agent, crew
from cachetools import TTLCache, cached
from llama_index.core.schema import MetadataMode
from pymilvus import MilvusException
from pymilvus import MilvusClient
import grpc
import os
import pytz
import time
from grpc.aio import AioRpcError
from grpc import RpcError
load_dotenv()




class ChatConversationVectorCache:

    _cache = TTLCache(maxsize=100, ttl=3600)
    _lock = asyncio.Lock()

    def __init__(self, user_id: str, **kwargs):
        self.user_id:str = user_id

    async def _internal_cache(self):
        async with self._lock:
            if self.user_id in self._cache:
                self._cache.expire()
                return self._cache[self.user_id]

            self._cache[self.user_id] = []

            return self._cache[self.user_id]

    def utc_to_ph():
        # Set UTC timezone
        utc = pytz.UTC

        # Get current UTC time
        utc_now = datetime.now(utc)

        # Set PH timezone
        set_timezone = pytz.timezone(os.getenv('TIMEZONE'))

        # Convert UTC time to PH time
        timezone_ready = utc_now.astimezone(set_timezone)

        # Separate date and time
        date_now = timezone_ready.strftime('%Y-%m-%d')
        time_now = timezone_ready.strftime('%H:%M:%S')

        return date_now, time_now
    # Example usage:
    print("PH Date:", ph_date)
    print("PH Time:", ph_time)

    def text_to_vector(self, input_user_message: str = None):
        if not input_user_message:
            return False
        ph_date, ph_time = self.utc_to_ph()

        document_input = f"**role=>user * content=>{input_user_message}**"
        docs = [Document(text=document_input, metadata={"timestamp":})]

        return True