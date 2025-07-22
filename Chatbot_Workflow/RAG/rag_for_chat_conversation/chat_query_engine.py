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
from llama_index.llms.groq import Groq as Llama_Groq
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


from Chatbot_Workflow.RAG.rag_for_chat_conversation.chat_memory_vector import GetMilvusVectorStore
load_dotenv()




class FlowState(BaseModel):
    input_user_id: str = ""
    input_user_message: str = ""
    user_message_v1: str = ""
    # get_vector: Optional[MilvusVectorStore] = PrivateAttr(default=None)


class ChatHistoryQueryEngine(Flow[FlowState]):

    def __init__(self):
        self._vector_store: Optional[MilvusVectorStore] = None
        self.embed_model = CohereEmbedding(
            input_type="search_query",
            model_name="embed-v4.0",
            api_key=os.getenv("CLIENT_COHERE_API_KEY")
        )
        self.llm_for_rag = Llama_Groq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("CLIENT_GROQ_API_1"),
            temperature=0.3
        )
        super().__init__()

    # @property
    async def milvus_vector_store(self):
        if self._vector_store is None:
            _get_vector = GetMilvusVectorStore(input_user_id=self.state.input_user_id)
            self._vector_store = await _get_vector.milvus_vector_store()
        return self._vector_store

    @start()
    def starting_with_prompt(self):
        prompt_for_query = f"""
        ### System: 
        You are a Meticulous as a Researcher. 
        
        ### Instructions: 
        With the context provided, make a detailed report of the context and how it might
        relate to the user message. Context is only reference and should be explained if
        not related to the user message by simply summarizing current context. 
        
        ### User:
        {self.state.input_user_message}
        """
        return prompt_for_query


    @listen(starting_with_prompt)
    def initiate_logics(self, data_from_previous):
        prompt = data_from_previous

        storage_context = StorageContext.from_defaults(
            vector_store=self.milvus_vector_store
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=self.milvus_vector_store,
            embed_model=self.embed_model,
            storage_context=storage_context
        )
        return prompt, index

    @listen(initiate_logics)
    async def query_engine(self, data_from_previous):
        prompt, index = data_from_previous
        query_engine = index.as_query_engine(llm=self.llm_for_rag)

        response = await query_engine.aquery(prompt)
        return response

