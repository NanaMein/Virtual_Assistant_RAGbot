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
from pymilvus import MilvusException
import grpc
import os
import pytz
import time

load_dotenv()

_vector_store_cache_testing = TTLCache(maxsize=100, ttl=300)

class MessageConversationVectorClass:
    # vector_cache = TTLCache(maxsize=100, ttl=1000)

    def __init__(self, user_input_id: str):
        self.user_id: str = user_input_id


    # def activate_vector(self):
    #     max_retries = 3
    #     attempt = 0
    #
    #     while attempt < max_retries:
    #         vector_store = self._vector_store(user_id=self.user_id)
    #         if vector_store:
    #             return vector_store
    #
    #         attempt += 1
    #         print(f"[Retry] Attempt {attempt} failed for user_id: {self.user_id}")
    #
    #     print(f"[Error] All {max_retries} attempts failed for user_id: {self.user_id}")
    #     return None

    # def activate_vector_with_error_handling(self):
    #     try:
    #         print("Trying to connect the vector")
    #         vector_store = self._vector_store(self.user_id)
    #     except (grpc.aio._call.AioRpcError, UnboundLocalError) as e:
    #         print(f"ERROR HANDLING SUCCESS: {e}\n\nERROR TYPE: {type(e)}")
    #         self.vector_cache.pop(self.user_id, None)
    #         print("POP NOW")
    #         vector_store = self._vector_store(self.user_id)
    #         print("RE INITIALIZING THE VECTOR ")
    #     except Exception as e2:
    #         print(f"NEW ERROR OCCURED: {e2}\n\nERROR TYPE: {type(e2)}")
    #         self.vector_cache.pop(self.user_id, None)
    #         print("POP NOW")
    #         vector_store = self._vector_store(self.user_id)
    #         print("RE INITIALIZING THE VECTOR ")
    #     return vector_store

    def get_vector_store_by_id(self, user_id: str):
        """
        Lazily initializes and caches the entire MilvusVectorStore instance.
        This is the expensive operation we want to avoid repeating.
        """
        valid_name = user_id.strip()
        collection_name = f"Collection_Name_{valid_name}_2025"

        global _vector_store_cache_testing
        if _vector_store_cache is None:
            print("Lazy-loading the main TTLCache for the Vector Store...")
            # This cache will hold ONE item: the vector store instance.
            # The TTL is a proactive "health check". After 4.5 minutes, we assume the
            # internal connection might be stale and force a full re-initialization.
            _vector_store_cache_testing = TTLCache(maxsize=5, ttl=300)

        try:
            # Get the cached VectorStore instance.
            return _vector_store_cache_testing[user_id]
        except KeyError:
            # Runs if the cache is empty OR the TTL expired.
            print("Creating new MilvusVectorStore instance (cache miss or TTL expired)...")
            # The "create" step is now instantiating the entire class.
            vector_store = MilvusVectorStore(
                uri=os.getenv('CLIENT_URI'),
                token=os.getenv('CLIENT_TOKEN'),
                collection_name=collection_name,
                dim=1536,
                embedding_field='embeddings',
                enable_sparse=True,
                enable_dense=True,
                overwrite=False,  # CHANGE IT FOR DEVELOPMENT STAGE ONLY
                sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
                search_config={"nprobe": 40},
                similarity_metric="IP",
                consistency_level="Session",
                hybrid_ranker="WeightedRanker",
                hybrid_ranker_params={"weights": [0.3, 0.7]},
            )

            _vector_store_cache[user_id] = vector_store
            return _vector_store_cache[user_id]

    async def query_vector_store_robustly(self, user_id: str,user_message: str, query_engine):

        try:
            vector_store = get_vector_store_by_id(user_id)
            print("Attempting vector store query...")
            query_result = await query_engine.aquery(user_message)
            print("=> Query successful on first attempt.")
            return query_result

        except ImportError as e:  # Critical setup error
            print(f"FATAL: pymilvus not installed! {e}")
            raise

        except MilvusException as e:  # Transient error
            print(f"Caught MilvusException: {e}")
            print("Invalidating cached vector store...")

            # SAFE cache invalidation that handles None case
            if _vector_store_cache is not None:
                # Use either method - both are safe for missing keys
                try:
                    del _vector_store_cache[user_id]  # Option 1
                except KeyError:
                    pass
                # OR
                _vector_store_cache.pop(user_id, None)  # Option 2 (cleaner)

            # Rebuild store and retry
            vector_store = get_vector_store_by_id(user_id)
            print("Retrying query with refreshed vector store...")

    #
    # def _vector_store(self, user_id: str):
    #     if user_id in self.vector_cache:
    #         return self.vector_cache[user_id]
    #     collection_name = f"CLIENT_{user_id}_2025_TESTING"
    #     try:
    #         vector_store = MilvusVectorStore(
    #             uri=os.getenv('CLIENT_URI'),
    #             token=os.getenv('CLIENT_TOKEN'),
    #             collection_name=collection_name,
    #             dim=1536,
    #             embedding_field='embeddings',
    #             enable_sparse=True,
    #             enable_dense=True,
    #             overwrite=False,  # CHANGE IT FOR DEVELOPMENT STAGE ONLY
    #             sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
    #             search_config={"nprobe": 40},
    #             similarity_metric="IP",
    #             consistency_level="Session",
    #             hybrid_ranker="WeightedRanker",
    #             hybrid_ranker_params={"weights": [0.3, 0.7]},
    #         )
    #
    #         self.vector_cache[user_id] = vector_store
    #         return self.vector_cache[user_id]
    #     except Exception as e:
    #         return None

from llama_index.vector_stores.milvus import MilvusVectorStore

# from llama_index.core.vector_stores import VectorStoreQuery # For typing
# from pymilvus import MilvusException # The error you will catch

# 1. LAZY-LOADING SETUP for the Vector Store instance.


_vector_store_cache = None


def get_vector_store_by_id(user_id: str):
    """
    Lazily initializes and caches the entire MilvusVectorStore instance.
    This is the expensive operation we want to avoid repeating.
    """
    valid_name = user_id.strip()
    collection_name = f"Collection_Name_{valid_name}_2025"

    global _vector_store_cache
    if _vector_store_cache is None:
        print("Lazy-loading the main TTLCache for the Vector Store...")
        # This cache will hold ONE item: the vector store instance.
        # The TTL is a proactive "health check". After 4.5 minutes, we assume the
        # internal connection might be stale and force a full re-initialization.
        _vector_store_cache = TTLCache(maxsize=5, ttl=300)

    try:
        # Get the cached VectorStore instance.
        return _vector_store_cache[user_id]
    except KeyError:
        # Runs if the cache is empty OR the TTL expired.
        print("Creating new MilvusVectorStore instance (cache miss or TTL expired)...")
        # The "create" step is now instantiating the entire class.
        vector_store = MilvusVectorStore(
            uri=os.getenv('CLIENT_URI'),
            token=os.getenv('CLIENT_TOKEN'),
            collection_name=collection_name,
            dim=1536,
            embedding_field='embeddings',
            enable_sparse=True,
            enable_dense=True,
            overwrite=False,  # CHANGE IT FOR DEVELOPMENT STAGE ONLY
            sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
            search_config={"nprobe": 40},
            similarity_metric="IP",
            consistency_level="Session",
            hybrid_ranker="WeightedRanker",
            hybrid_ranker_params={"weights": [0.3, 0.7]},
        )

        _vector_store_cache[user_id] = vector_store
        return _vector_store_cache[user_id]



