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
    ZillizCommonError,
    ZillizUnexpectedError
)


load_dotenv()



class IngestionPipelineError(Exception):
    """Failure in processing the documents to saving in vector"""
    pass

class DocumentTransformingError(IngestionPipelineError):
    """Error in document processing"""
    pass

class InitializedResourcesError(IngestionPipelineError):
    """Error occurring in the embed or llm"""
    pass


class ChatConversationVectorCache:

    _cache = TTLCache(maxsize=100, ttl=3600)

    def __init__(self, user_id: str):
        self.user_id:str = user_id
        self._get = GetMilvusVectorStore(input_user_id=self.user_id)
        self._lock = asyncio.Lock()

    async def _internal_cache(self) -> Tuple[CohereEmbedding, LlamaGroq]:

        async with self._lock:
            try:
                embed_model, llm = self._cache[self.user_id]
                return embed_model, llm

            except KeyError:
                pass

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

    async def resources(self) -> Tuple[VectorStoreIndex, CohereEmbedding, LlamaGroq]:
        """
        returns:
            Vector store, Embed Model, and LLM
        """
        try:
            vector = await self._get.zilliz_vector_cloud()
        except ZillizCommonError:
            raise
        except ZillizUnexpectedError:
            raise

        embed, llm = await self._internal_cache()

        return vector, embed, llm

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
        self, input_user_message: str,
        input_assistant_message: str
    ) -> bool:

        async with self._lock:

            try:
                vector_store, embed_model, llm = await self.resources()
            except ZillizCommonError as ce:
                raise InitializedResourcesError("Error in database, can be used later") from ce
            except ZillizUnexpectedError as ue:
                raise InitializedResourcesError("Unexpected Error, Vector calls might be critical") from ue
            except Exception as ex:
                raise InitializedResourcesError("Unexpected Error, API Calls or Program might be critical") from ex

            try:
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
                return True


            except Exception as e:
                raise DocumentTransformingError("Error is hard to identify, please try again later") from e