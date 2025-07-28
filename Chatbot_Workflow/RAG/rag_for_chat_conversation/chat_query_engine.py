# import litellm
# litellm._turn_on_debug()
import asyncio
from collections import deque, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Deque, Type, Optional, Tuple, TypeVar, Generic
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
from Chatbot_Workflow.RAG.rag_for_chat_conversation.chat_embed_to_vector import ChatConversationVectorCache
load_dotenv()


# class QueryEngineException(Exception):
#     """Error happens in Query engine"""
#     pass
#
# class ResourcesInitializedError(QueryEngineException):
#     """Failed to initialization of the error or api calls"""
#     pass




@dataclass(frozen=True)
class QueryResult:
    ok: bool
    query_response: Optional[str] = None
    error: Optional[str] = None

dataclass(frozen=True)
class DataResources:
    pass

class FlowState(BaseModel):
    input_user_id: str = ""
    input_user_message: str = ""
    user_message_v1: str = ""
    index: Optional[VectorStoreIndex] = PrivateAttr(default=None)


class ChatHistoryQueryEngine(Flow[FlowState]):

    _caching_resources = TTLCache(maxsize=100, ttl=3600)

    def __init__(self):
        self._lock = asyncio.Lock()
        self._vector_class = GetMilvusVectorStore(input_user_id=self.state.input_user_id)
        self._embeddings = ChatConversationVectorCache(user_id=self.state.input_user_id)

        super().__init__()


    async def _llm_and_embed_resources(self) -> Tuple[CohereEmbedding, Llama_Groq]:
        async with self._lock:
            user_id = self.state.input_user_id
            _cached_resources = self._caching_resources.get(user_id)

            if isinstance(_cached_resources, tuple) and len(_cached_resources) == 2:
                embed_model, llm_for_rag = _cached_resources
                return embed_model, llm_for_rag

            embed_model = CohereEmbedding(
                input_type="search_query",
                model_name="embed-v4.0",
                api_key=os.getenv("CLIENT_COHERE_API_KEY")
            )
            llm_for_rag = Llama_Groq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                api_key=os.getenv("CLIENT_GROQ_API_1"),
                temperature=0.3
            )
            self._caching_resources.pop(user_id, None)
            self._caching_resources.expire()
            self._caching_resources[user_id] = (embed_model, llm_for_rag)
            return embed_model, llm_for_rag

    async def query_engine_core(self, input_message: str) -> QueryResult:

        async with self._lock:
            result_of_object = await self._vector_class.get_zilliz_vector_result()

            if not result_of_object.ok:
                return QueryResult(ok=False, error=result_of_object.error)

            vector_store = result_of_object.data

            try:
                embed_model, llm_for_rag = await self._llm_and_embed_resources()
            except Exception as e:
                resources_error = f"""Unexpected Error in initialization of llm and embed:\n
                Error Type is {type(e)} and error traceback is: {e}"""
                return QueryResult(ok=False, error=resources_error)

            try:

                # ... build engine, run query ...
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store, embed_model=embed_model, use_async=True
                )
                query_engine = index.as_query_engine(
                    llm=llm_for_rag,
                    vector_store_query_mode="hybrid",
                    similarity_top_k=7,
                    use_async=True,
                )
                retrieved_query = await query_engine.aquery(input_message)
                return QueryResult(ok=True, query_response=retrieved_query.response)

            except (ImportError, MilvusException, AioRpcError) as e1:
                return QueryResult(ok=False, error=f"Common Error occur: {e1!s}")

            except Exception as e2:
                return QueryResult(ok=False, error=f"Unexpected Error for Llama Index: {e2!s}")

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
    async def send_data_to_query_engine(self, data_from_previous):
        prompt = data_from_previous
        result = await self.query_engine_core(input_message=prompt)

        if result.ok:
            self.state.user_message_v1 = result.query_response
            return "PASSED"
        else:
            self.state.user_message_v1 = result.error
            return "FAILED"

    @router(send_data_to_query_engine)
    def error_handling(self, data_from_previous):
        response = data_from_previous
        if response == "PASSED":
            return "PASSED"
        else:
            return "FAILED"

    @listen("PASSED")
    async def value_preserved(self):
        mem = await self._embeddings.add_conversation_with_extractor(
            input_user_message=self.state.input_user_message,
            input_assistant_message=self.state.user_message_v1
        )
        if mem.ok:
            return self.state.user_message_v1
        else:
            return mem.error

    @listen("FAILED")
    def value_failure(self):
        return self.state.user_message_v1









    @listen(starting_with_prompt)
    async def initiate_logics(self, data_from_previous):
        prompt = data_from_previous
        vector_store, embed_model, llm_for_rag = await self.init_query_engine_resources()

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            use_async=True
        )
        self.state.holding_index = index
        return prompt, index

    @router(starting_with_prompt)
    async def error_handling(self):
        try:
            embed_model, _ = await self.llm_and_embed_resources()
            vector_store = await self.milvus_vector_store()
            return "success"

        except ZillizCloudConnectionError:
            return "failed"

    @listen(initiate_logics)
    async def query_engine(self, data_from_previous):

        prompt, index = data_from_previous
        _, llm = await self.llm_and_embed_resources()
        query_engine = index.as_query_engine(
            llm=llm,
            vector_store_query_mode="hybrid",
            similarity_top_k=7,
            use_async=True,
        )
        query_eng = self.state.holding_index.as_query_engine()

        response = await query_engine.aquery(prompt)
        return response

