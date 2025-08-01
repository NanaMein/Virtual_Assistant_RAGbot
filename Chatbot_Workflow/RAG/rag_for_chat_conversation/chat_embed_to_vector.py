# import litellm
# litellm._turn_on_debug()
import asyncio
from collections import deque, defaultdict
from functools import lru_cache
from typing import Deque, Type, Optional, Tuple, Any, TypeVar, Generic
from llama_index.core.storage.chat_store.base_db import MessageStatus
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    SimpleDirectoryReader
)
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, AgentWorkflow
from llama_index.core.memory.memory import Memory
from llama_index.core.workflow import Context
from llama_index.llms.groq import Groq as LlamaGroq
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
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    DocumentContextExtractor
)
from llama_index.core.ingestion import IngestionPipeline
from Chatbot_Workflow.RAG.rag_for_chat_conversation.chat_memory_vector import (
    GetMilvusVectorStore,
    VectorObjectResult
)
from dataclasses import dataclass


T = TypeVar("T")

@dataclass(frozen=True)
class VectorObjectResult(Generic[T]):
    """ok: bool\n
    data: Optional[T] = None[This is the data object]\n
    err_name: Optional[str] = None[This is the name of the error]\n
    err_desc: Optional[str] = None[this is the description of the error]\n
    err_loc: Optional[str] = None[this is the location or where the error occurred]\n
    opt_err: Optional[str] = None[This is for the optional error like traceback]\n
    overall_err: @property [This is a read only overall error or combination of other parameters.
    Used for less boilerplate code and string builder for all error parameters]"""

    ok: bool
    data: Optional[T] = None
    err_name: str | None = None
    err_desc: str | None = None
    err_loc: str | None = None
    opt_err: Exception | str | None = None
load_dotenv()

T = TypeVar("T")

@dataclass(frozen=True)
class ResourcesResult(Generic[T]):
    ok: bool
    data: Optional[T] = None
    err_name: str | None = None
    err_desc: str | None = None
    err_loc: str | None = None
    opt_err: Exception | str | None = None

    @property
    def overall_err(self) -> str:
        return f"""
            Error name: {self.err_name}\n
            Error is: {self.err_desc}\n
            Error location is: {self.err_loc}\n
            Optional error traceback: [{self.opt_err}]
            """

@dataclass(frozen=True)
class DocumentProcessingResult:
    ok: bool
    error: Optional[str] = None

@dataclass(frozen=True)
class DataResources:
    vector_store: MilvusVectorStore
    embed_model: CohereEmbedding
    llm_for_rag: LlamaGroq

class ChatConversationVectorCache:

    _cache = TTLCache(maxsize=100, ttl=3600)

    def __init__(self, user_id: str):
        self.user_id:str = user_id
        self._i_will = GetMilvusVectorStore(input_user_id=self.user_id)
        self._lock = asyncio.Lock()

    async def _internal_cache(self) -> Optional[Tuple[CohereEmbedding, LlamaGroq]]:

        async with self._lock:
            inside_cache = self._cache.get(self.user_id)

            if isinstance(inside_cache, tuple) and len(inside_cache)==2:
                embed_model, llm = inside_cache
                return embed_model, llm

            llm = LlamaGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                api_key=os.getenv("CLIENT_GROQ_API_1"),
                temperature=0.25
            )
            embed_model = CohereEmbedding(
                model_name="embed-v4.0",
                api_key=os.getenv('CLIENT_COHERE_API_KEY'),
                input_type="search_document"
            )
            self._cache.pop(self.user_id, None)
            self._cache.expire()
            self._cache[self.user_id] = embed_model, llm
            return embed_model, llm

    async def resources_with_validation(self)-> ResourcesResult:
        async with self._lock:
            try:
                embed, llm = await self._internal_cache()
            except Exception as ex:
                err_name = "Unexpected Resources Initialization"
                err_desc = "Unexpected Error occurred. Might take for a while"
                err_loc = "Chat Embed to Vector Layer"
                return ResourcesResult(
                    ok=False,
                    err_name=err_name,
                    err_desc=err_desc,
                    err_loc=err_loc,
                    opt_err=ex
                )


            vector_result = await self._i_will.get_zilliz_vector_result()
            if not vector_result.ok:
                err_name = "Vector Error"
                err_desc = "Theres a problem in Vector layer, need to check"
                err_loc = "Chat Embed to Vector"
                return ResourcesResult(
                    ok=False,
                    err_name=err_name,
                    err_desc=err_desc,
                    err_loc=err_loc
                )

            else:
                vector_store = vector_result.data
                data = DataResources(vector_store=vector_store, embed_model=embed, llm_for_rag=llm)
                return ResourcesResult(ok=True, data=data)




    @staticmethod
    def utc_to_ph():
        utc = pytz.UTC
        utc_now = datetime.now(utc)
        set_timezone = pytz.timezone(os.getenv('TIMEZONE'))
        timezone_ready = utc_now.astimezone(set_timezone)

        date_now = timezone_ready.strftime('%Y-%m-%d')
        time_now = timezone_ready.strftime('%H:%M:%S')

        return date_now, time_now


    async def add_conversation_with_extractor(
        self, input_user_message: str = None,
        input_assistant_message: str = None
    ) -> DocumentProcessingResult:

        async with self._lock:
            try:
                result: ResourcesResult[DataResources] = await self.resources_with_validation()
                if not result.ok:
                    error= f"""This is Embedding layer. Original error is: \n{result.error}"""
                    return DocumentProcessingResult(ok=False, error=error)

                vector_store = result.data.vector_store
                embed_model = result.data.embed_model
                llm = result.data.llm_for_rag

                if not input_user_message or not input_assistant_message:
                    error = """an input message from user and assistant is missing from the parameter"""
                    return DocumentProcessingResult(ok=False, error=error)


                user = f"<conversation_turn>role=user content={input_user_message} "
                assistant = f" role=assistant content={input_assistant_message}</conversation_turn>"

                date_now, time_now = self.utc_to_ph()

                document_input = user + assistant
                _docs = [Document(
                    text=document_input,
                        metadata={
                            "Date": date_now,
                            "Time": time_now
                        }
                    )
                ]
                extractors = [
                    TitleExtractor(nodes=3, llm=llm),
                    QuestionsAnsweredExtractor(questions=2, llm=llm),
                    SummaryExtractor(summaries=["prev", "self", "next"], llm=llm),
                ]

                text_splitter = TokenTextSplitter(chunk_size=350, chunk_overlap=60)
                pipeline = IngestionPipeline(transformations=[text_splitter] + extractors)
                _nodes = pipeline.run(documents=_docs)
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=embed_model,
                    use_async=True
                )
                await index.ainsert_nodes(nodes=_nodes)
                return DocumentProcessingResult(ok=True)

            except Exception as e:
                error = f"Unexpected error in Document processing: {e}"
                return DocumentProcessingResult(ok=False, error=error)