import asyncio
from typing import Deque, Type, Optional, Tuple
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from dotenv import load_dotenv
from cachetools import TTLCache, cached
from pymilvus import MilvusException, MilvusClient
import grpc
import os
from grpc.aio import AioRpcError


load_dotenv()


class ZillizCloudError(Exception):
    """Main errors for all related to zilliz cloud"""
    pass

class ZillizCommonError(ZillizCloudError):
    pass

class ZillizUnexpectedError(ZillizCloudError):
    pass

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
            try:
                client = self._client_cache[user_id]
                return client

            except KeyError:
                pass

            client = MilvusClient(
                uri=os.getenv('CLIENT_URI'),
                token=os.getenv('CLIENT_TOKEN')
            )
            self._client_cache.pop(user_id, None)
            self._client_cache[user_id] = client
            return client

    async def _get_connection_to_milvus_vector_store(self, user_id: str) -> MilvusVectorStore:
        """Getting vector store from """
        async with self._vector_lock:
            try:
                vector_store = self._vector_cache[user_id]
                return vector_store
            except KeyError:
                pass

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
        #Reconnection and trying the refreshed vector or normally pass the async lock
        #with reconnection to cache
        try:
            return await self._vector()

        except (AioRpcError, MilvusException, UnboundLocalError, ImportError):
            raise

    async def zilliz_vector_cloud(self) -> MilvusVectorStore:
        """Uses refresh and reconnect if an error occurred in vector function.
        Then raise error if an unexpected error occur"""
        try:
            refreshed_vector = await self._refresh_and_get_vector()
            return refreshed_vector
        except (AioRpcError, MilvusException, UnboundLocalError, ImportError) as e:
            raise ZillizCommonError("Common Error happened, please try again later, : ") from e
        except Exception as e:
            raise ZillizUnexpectedError("Unexpected Error happened, wait for few minutes and try again later") from e


def ttl_conversion_to_day(number_of_days: float):
    total = 86400 * number_of_days
    return total



# async def milvus_vector_store(self) -> Optional[MilvusVectorStore]:
#     for attempt in range(2):
#         try:
#             return await self._vector()
#
#         except (AioRpcError, UnboundLocalError, ImportError, MilvusException):
#             async with self._client_lock:
#                 self._client_cache.pop(self.user_id, None)
#                 self._client_cache.expire()
#                 self._client_init = None
#             async with self._vector_lock:
#                 self._vector_cache.pop(self.user_id, None)
#                 self._vector_cache.expire()
#                 self._vector_init = None
#
#     raise ZillizCloudConnectionError("Unexpected Error occurred, please try again later")