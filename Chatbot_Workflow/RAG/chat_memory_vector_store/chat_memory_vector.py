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
from pymilvus import MilvusException, MilvusClient
import grpc
import os
import pytz
import time
from grpc.aio import AioRpcError
from grpc import RpcError
load_dotenv()



class GetMilvusVectorStore:

    vector_by_id = TTLCache(maxsize=100, ttl=300)
    vector_by_collection = TTLCache(maxsize=100, ttl=3600)

    def __init__(self, input_user_id: str):

        self.user_id: str = input_user_id
        self._resources: Optional[MilvusVectorStore] = None
        self._client: Optional[MilvusClient] = None
        self.collection_name: str = f"Collection_Name_{self.user_id.strip()}_2025"

    @property
    def cached_resource(self):
        if self._resources is None:
            self._resources = self._getting_resource(user_id=self.user_id)
        return self._resources

    @property
    def client_for_vector(self):
        if self._client is None:
            self._client = self._milvus_client(self.user_id)
        return self._client

    def _milvus_client(self, user_id: str):
        if user_id in self.vector_by_collection:
            return self.vector_by_collection[user_id]

        client = MilvusClient(
            uri=os.getenv('CLIENT_URI'),
            token=os.getenv('CLIENT_TOKEN')
        )
        self.vector_by_collection[user_id] = client
        return client

    def _getting_resource(self, user_id: str) -> MilvusVectorStore:
        # collection_name = f"Collection_Name_{self.user_id.strip()}_2025"
        if user_id in self.vector_by_id:
            return self.vector_by_id[user_id]


        does_it_exist=self.client_for_vector.has_collection(
            collection_name=self.collection_name
        )
        vector_store = MilvusVectorStore(
            uri=os.getenv('CLIENT_URI'),
            token=os.getenv('CLIENT_TOKEN'),
            collection_name=self.collection_name,
            dim=1536,
            embedding_field='embeddings',
            enable_sparse=True,
            enable_dense=True,
            overwrite=False,  # CHANGE IT FOR DEVELOPMENT STAGE ONLY
            sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
            search_config={"nprobe": 60},
            similarity_metric="IP",
            consistency_level="Session",
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 120},
        )
        # except (MilvusException, AioRpcError, ImportError) as mai:
        if not does_it_exist:
            self.client_for_vector.alter_collection_properties(
                collection_name=self.collection_name,
                properties={"collection.ttl.seconds": ttl_conversion_to_day(15)}
            )

        self.vector_by_id[user_id] = vector_store

        return self.vector_by_id[user_id]

    def milvus_vector_store(self):
        try:
            vector_store = self.cached_resource
        except (MilvusException, KeyError) as me:
            print(f"MilvusException, KeyError: {me}")
            self._resources = None
            vector_store = self.cached_resource
        except (AioRpcError, ImportError, grpc.RpcError, UnboundLocalError) as aie:
            print(f"AioRpcError, ImportError, RpcError, UnboundLocalError: {aie}")
            self._resources = None
            vector_store = self.cached_resource
        except Exception as e:
            print(f"Unexpected Error: {e}")
            self._resources = None
            self._
            self.vector_by_id.pop(self.user_id, None)
            self.vector_by_collection.pop(self.user_id, None)
            return None
        finally:
            self.vector_by_id.expire()
            self.vector_by_collection.expire()

        return vector_store

def ttl_conversion_to_day(number_of_days: float):
    total = 86400 * number_of_days
    return total



