import asyncio
from typing import Deque, Type, Optional, Tuple, Union, TypeVar, Generic
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from dotenv import load_dotenv
from cachetools import TTLCache, cached
from pymilvus import MilvusException, MilvusClient
from dataclasses import dataclass
import grpc
import os
from grpc.aio import AioRpcError

load_dotenv()


T = TypeVar("T")
@dataclass(frozen=True)
class VectorObjectResult(Generic[T]):
    """ok: bool => If data is Ok or not\n
    data: Optional[T] => Data to be passed on\n
    error: Optional[str] => error traceback """

    ok: bool
    data: Optional[T] = None
    error: Optional[str] = None



class GetMilvusVectorStore:
    """
    Asycncronously make a vector connection to zilliz with error handling, lock and reconnection
    mechanism by providing a user id to and each user id will have its own vector store
    for chat conversation. It has a time to live of 15 days so it will be clear when certain
    time has passed
    """
    _vector_cache = TTLCache(maxsize=100, ttl=300)
    _client_cache = TTLCache(maxsize=100, ttl=3600)


    def __init__(self, input_user_id: str):
        self.user_id: str = input_user_id
        self.collection_name: str = f"Collection_Name_{self.user_id.strip()}_2025"
        self._vector_init: Optional[MilvusVectorStore] = None
        self._client_init: Optional[MilvusClient] = None
        self._vector_lock = asyncio.Lock()
        self._client_lock = asyncio.Lock()

    async def _client(self) -> MilvusClient:
        if self._client_init is None:
            self._client_init = await self._get_connection_to_client_server(user_id=self.user_id)
        return self._client_init

    async def _vector(self) -> MilvusVectorStore:
        if self._vector_init is None:
            self._vector_init = await self._get_connection_to_milvus_vector_store(user_id=self.user_id)
        return self._vector_init

    async def _get_connection_to_client_server(self, user_id: str) -> MilvusClient:
        """simple low level connection to zilliz client server.
        Only used if collection name exist and alter properties"""
        async with self._client_lock:

            client_in_cache = self._client_cache.get(user_id)

            if client_in_cache:
                return client_in_cache

            client = MilvusClient(
                uri=os.getenv('CLIENT_URI'),
                token=os.getenv('CLIENT_TOKEN')
            )
            self._client_cache[user_id] = client
            return client

    async def _get_connection_to_milvus_vector_store(self, user_id: str) -> MilvusVectorStore:
        """Getting vector store from """
        async with self._vector_lock:

            vector_store_in_cache = self._vector_cache.get(user_id)

            if vector_store_in_cache:
                return vector_store_in_cache

            client = await self._client()
            existing_collection = client.has_collection(
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
                use_async=True,
            )

            if not existing_collection:
                client.alter_collection_properties(
                    collection_name=self.collection_name,
                    properties={"collection.ttl.seconds": ttl_conversion_to_day(15)}
                )
            self._vector_cache[user_id] = vector_store
            return vector_store

    async def _refresh_and_get_vector(self):
        """Will check connection if working, then reconnect all api to milvus
        and return vector"""
        async with self._client_lock:
            try:
                await self._client()
            except MilvusException:
                self._client_cache.pop(self.user_id, None)
                self._client_cache.expire()

        async with self._vector_lock:
            try:
                await self._vector()
            except (AioRpcError, MilvusException):
                self._vector_cache.pop(self.user_id, None)
                self._vector_cache.expire()

        return await self._vector()


    async def get_zilliz_vector_result(self) -> VectorObjectResult:
        """Uses refresh and reconnect if an error occurred in vector function.
        Then raise error if an unexpected error occur"""
        try:
            refreshed_vector = await self._refresh_and_get_vector()
            return VectorObjectResult(ok=True, data=refreshed_vector)

        except (AioRpcError, MilvusException, UnboundLocalError, ImportError) as ce:
            common_error = f"""
            Error Originate From: Vector Layer\n
            Status: Catching common error: {ce}\n
            Additional Information: Error type is {type(ce)}
            Solution: Retry or wait for a moment before proceeding
            Original Error: """
            return VectorObjectResult(ok=False, error=common_error)

        except Exception as ex:
            unexpected_error = f"""
            Error Originate From: Vector Layer\n
            Status: Unexpected error: {ex}\n
            Additional Information: Error type is {type(ex)}
            Solution: Restart system or debug and test the program
            """

            return VectorObjectResult(ok=False, error=unexpected_error)

def ttl_conversion_to_day(number_of_days: float):
    total = 86400 * number_of_days
    return total
