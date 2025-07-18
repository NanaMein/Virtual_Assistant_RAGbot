# import litellm
# litellm._turn_on_debug()
import asyncio
from collections import deque, defaultdict
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
from crewai.flow import Flow, start, listen, router, or_, and_, persist
from llama_index.core.base.llms.base import ChatMessage  # schema import ChatMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import tool, BaseTool
from crewai.project import CrewBase, task, agent, crew
from cachetools import TTLCache, cached
from llama_index.core.schema import MetadataMode
import os
import pytz
import time



load_dotenv()


class PersonalityVectorClass:

    vector_cache = TTLCache(maxsize=100, ttl=290)

    def __init__(self, user_input_id: str):
        self.user_id: str = user_input_id

    def activate_vector(self):
        max_retries = 3
        attempt = 0

        while attempt < max_retries:
            vector_store = self._vector_store(user_id=self.user_id)
            if vector_store:
                return vector_store

            attempt += 1
            print(f"[Retry] Attempt {attempt} failed for user_id: {self.user_id}")

        print(f"[Error] All {max_retries} attempts failed for user_id: {self.user_id}")
        return None

    def _vector_store(self, user_id:str):
        if user_id in self.vector_cache:
            return self.vector_cache[user_id]
        try:
            vector_store = MilvusVectorStore(
                uri=os.getenv('CLIENT_URI'),
                token=os.getenv('CLIENT_TOKEN'),
                collection_name=os.getenv("PERSONALITY_COLLECTION"),
                dim=1536,
                embedding_field='embeddings',
                enable_sparse=True,
                enable_dense=True,
                overwrite=False,  # CHANGE IT FOR DEVELOPMENT STAGE ONLY
                sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
                search_config={"nprobe": 20},
                similarity_metric="IP",
                consistency_level="Session",
                hybrid_ranker="WeightedRanker",
                hybrid_ranker_params={"weights": [0.65, .9]},
            )

            self.vector_cache[user_id] = vector_store
            return self.vector_cache[user_id]
        except Exception as e:
            return None


