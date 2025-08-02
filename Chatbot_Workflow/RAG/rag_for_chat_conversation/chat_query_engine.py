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


from Chatbot_Workflow.RAG.rag_for_chat_conversation.chat_memory_vector import GetMilvusVectorStore
from Chatbot_Workflow.RAG.rag_for_chat_conversation.chat_embed_to_vector import ChatConversationVectorCache
load_dotenv()

T = TypeVar("T")
@dataclass(frozen=True)
class FlowObjectResult(Generic[T]):
    """
    ok: [This is if data is Success or Failure]\n
    data: [This is the data object]\n
    err_name: [This is the name of the error]\n
    err_desc: [this is the description of the error]\n
    err_loc: [this is the location or where the error occurred]\n
    opt_err: [This is for the optional error like traceback]\n
    overall_err: [This is a read only overall error or combination of other parameters.
    Used for less boilerplate code and string builder for all error parameters]"""

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
class QueryEngineResult(Generic[T]):
    """
    ok: [This is if data is Success or Failure]\n
    data: [This is the data object]\n
    err_name: [This is the name of the error]\n
    err_desc: [this is the description of the error]\n
    err_loc: [this is the location or where the error occurred]\n
    opt_err: [This is for the optional error like traceback]\n
    overall_err: [This is a read only overall error or combination of other parameters.
    Used for less boilerplate code and string builder for all error parameters]"""

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

class FlowState(BaseModel):
    input_user_message: str = ""
    user_message_v1: str = ""
    output_message: str = ""
    save_chat_error: str = ""



class ChatHistoryQueryEngine(Flow[FlowState]):

    _caching_resources = TTLCache(maxsize=100, ttl=3600)

    def __init__(self, input_user_id: str, **kwargs):
        self.input_user_id: str = input_user_id
        self._qe_lock = asyncio.Lock()
        self._res_lock = asyncio.Lock()
        self._vector_class = GetMilvusVectorStore(input_user_id=self.input_user_id)
        self.chat_history = ChatConversationVectorCache(user_id=self.input_user_id)

        super().__init__(**kwargs)


    async def _llm_and_embed_resources(self) -> Tuple[CohereEmbedding, Llama_Groq]:
        async with self._res_lock:
            _cached_resources = self._caching_resources.get(self.input_user_id)

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
            self._caching_resources.pop(self.input_user_id, None)
            self._caching_resources.expire()
            self._caching_resources[self.input_user_id] = (embed_model, llm_for_rag)
            return embed_model, llm_for_rag

    async def main_query_engine_core(self, input_message: str) -> QueryEngineResult:
        async with self._qe_lock:
            result_of_object = await self._vector_class.get_zilliz_vector_result()

            #FIRST CONDITIONAL
            if not result_of_object.ok:
                err_name = "Vector Object Error"
                err_desc = "Error might occurred in Vector Layer"
                err_loc = "Chat Query Engine Layer"
                return QueryEngineResult(
                    ok=False,
                    err_name=err_name,
                    err_desc=err_desc,
                    err_loc=err_loc,
                    opt_err=result_of_object.opt_err
                )
                # return QueryEngineResult(ok=False, error=result_of_object.overall_err)
            vector_store = result_of_object.data

            #SECOND CONDITIONAL
            try:
                embed_model, llm_for_rag = await self._llm_and_embed_resources()
            except Exception as res_ex:
                err_name = "Unexpected Error in Flow"
                err_desc = "Error occurred might be from initialization of resources or cache"
                err_loc = "Chat Query Engine Layer"
                return QueryEngineResult(
                    ok=False,
                    err_name=err_name,
                    err_desc=err_desc,
                    err_loc=err_loc,
                    opt_err=res_ex
                )

            #THIRD CONDITIONAL
            try:
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
                return QueryEngineResult(
                    ok=True, data=retrieved_query.response
                )
            except Exception as ex:
                err_name = "Unexpected Query Engine Failure"
                err_desc = "Error might be in query engine or along the lines"
                err_loc = "Chat Query Engine Layer"
                return QueryEngineResult(
                    ok=False, err_name=err_name, err_desc=err_desc, err_loc=err_loc, opt_err=ex
                )


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
        self.state.user_message_v1 = prompt_for_query


    @router(starting_with_prompt)
    async def query_to_vector_with_error_handling(self):

        result = await self.main_query_engine_core(input_message=self.state.user_message_v1)

        if result.ok:
            self.state.output_message = result.data
            return "QUERY_PASSED"
        else:
            self.state.output_message = result.overall_err
            return "QUERY_FAILED"


    @listen("QUERY_FAILED")
    def query_failed(self):
        opt_err = self.state.output_message
        err_name = "Query Engine Failed in Class"
        err_desc = "Error occurred inside flow but not sure of the error"
        err_loc = "Chat Query Engine (FLOW) Layer"
        return FlowObjectResult(
            ok=False, err_name=err_name, err_desc=err_desc, err_loc=err_loc, opt_err=opt_err
        )


    @listen("QUERY_PASSED")
    async def query_passed(self):
        result = await self.chat_history.add_conversation_with_extractor(
            input_user_message=self.state.input_user_message,
            input_assistant_message=self.state.output_message
        )
        if not result.ok:
            self.state.save_chat_error = result.error
        return result.ok


    @router(query_passed)
    def saving_chat_validation(self, _data):
        result_ok = _data
        if result_ok:
            return "SAVING_PASSED"
        else:
            return "SAVING_FAILED"


    @listen("SAVING_PASSED")
    def saving_passed(self):
        output_data = self.state.output_message
        return FlowObjectResult(
            ok=True, data=output_data
        )


    @listen("SAVING_FAILED")
    def saving_failed(self):
        opt_err = self.state.save_chat_error
        err_name = "Embedding To Vector Error"
        err_desc = "Error occurred inside flow but error might be in embed layer"
        err_loc = "Chat Query Engine (FLOW) Layer"
        return FlowObjectResult(
            ok=False, err_name=err_name, err_desc=err_desc, err_loc=err_loc, opt_err=opt_err
        )
