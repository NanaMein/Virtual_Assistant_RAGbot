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



class GetMilvusVectorStore:
    """
    Asycncronously make a vector connection to zilliz with error handling, lock and reconnection
    mechanism by providing a user id to and each user id will have its own vector store
    for chat conversation. It has a time to live of 15 days so it will be clear when certain
    time has passed
    """
    _vector_by_id = TTLCache(maxsize=100, ttl=300)
    _vector_by_collection = TTLCache(maxsize=100, ttl=3600)
    _vector_lock = asyncio.Lock()
    _client_lock = asyncio.Lock()

    def __init__(self, input_user_id: str):
        self.user_id: str = input_user_id
        self._vector: Optional[MilvusVectorStore] = None
        self._client: Optional[MilvusClient] = None
        self.collection_name: str = f"Collection_Name_{self.user_id.strip()}_2025"





    async def client_for_vector(self) -> MilvusClient:
        if self._client is None:
            self._client = await self._milvus_client(user_id=self.user_id)
        return self._client

    async def vector_store(self) -> MilvusVectorStore:
        if self._vector is None:
            self._vector = await self._getting_resource(user_id=self.user_id)
        return self._vector

    async def _milvus_client(self, user_id: str) -> MilvusClient:

        async with self._client_lock:
            try:
                client = self._vector_by_collection[user_id]


            except KeyError:
                client = MilvusClient(
                    uri=os.getenv('CLIENT_URI'),
                    token=os.getenv('CLIENT_TOKEN')
                )
                self._vector_by_collection.pop(user_id, None)
                self._vector_by_collection.expire()
                self._vector_by_collection[user_id] = client
                return client

            else:
                return client


    async def _getting_resource(self, user_id: str) -> MilvusVectorStore:

        async with self._vector_lock:
            try:
                vector_store = self._vector_by_id[user_id]
            except KeyError:
                client = await self.client_for_vector()
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

                self._vector_by_id.expire()

                self._vector_by_id[user_id] = vector_store
                return vector_store
            else:
                return vector_store


    async def milvus_vector_store(self) -> Optional[MilvusVectorStore]:
        try:
            vector_store = await self.vector_store()

        except (AioRpcError, UnboundLocalError, MilvusException) as aie:
            print(f"AioRpcError, UnboundLocalError: {aie}")
            async with self._vector_lock:
                self._vector_by_id.pop(self.user_id, None)
                self._vector_by_id.expire()
            async with self._client_lock:
                self._vector_by_collection.pop(self.user_id, None)
                self._vector_by_collection.expire()
            try:
                return await self.vector_store()
            except Exception:
                return None
        else:
            return vector_store




def ttl_conversion_to_day(number_of_days: float):
    total = 86400 * number_of_days
    return total



