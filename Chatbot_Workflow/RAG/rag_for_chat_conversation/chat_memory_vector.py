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

    vector_by_id = TTLCache(maxsize=100, ttl=300)
    vector_by_collection = TTLCache(maxsize=100, ttl=3600)
    _lock = asyncio.Lock()

    def __init__(self, input_user_id: str):

        self.user_id: str = input_user_id
        self._resources: Optional[MilvusVectorStore] = None
        self._client: Optional[MilvusClient] = None
        self.collection_name: str = f"Collection_Name_{self.user_id.strip()}_2025"

    # @property
    async def cached_resource(self) -> MilvusVectorStore:
        if self._resources is None:
            self._resources = await self._getting_resource(user_id=self.user_id)
        return self._resources

    # @property
    async def client_for_vector(self) -> MilvusClient:
        if self._client is None:
            self._client = await self._milvus_client(self.user_id)
        return self._client

    async def _milvus_client(self, user_id: str):
        if user_id in self.vector_by_collection:
            return self.vector_by_collection[user_id]

        # async with self._lock:
        client = MilvusClient(
            uri=os.getenv('CLIENT_URI'),
            token=os.getenv('CLIENT_TOKEN')
        )
        self.vector_by_collection[user_id] = client
        self.vector_by_collection.expire()
        return client

    async def _getting_resource(self, user_id: str) -> MilvusVectorStore:
        if user_id in self.vector_by_id:
            return self.vector_by_id[user_id]

        async with self._lock:
            client = await self.client_for_vector()

            does_it_exist = client.has_collection(
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
            if not does_it_exist:
                client.alter_collection_properties(
                    collection_name=self.collection_name,
                    properties={"collection.ttl.seconds": ttl_conversion_to_day(15)}
                )


            self.vector_by_id[user_id] = vector_store
            self.vector_by_id.expire()
            return self.vector_by_id[user_id]

    async def milvus_vector_store(self):
        try:
            vector_store = await self.cached_resource()
        except (MilvusException, KeyError) as me:
            print(f"MilvusException, KeyError: {me}")
            self._resources = None
            vector_store = await self.cached_resource()
        except (AioRpcError, ImportError, grpc.RpcError, UnboundLocalError) as aie:
            print(f"AioRpcError, ImportError, RpcError, UnboundLocalError: {aie}")
            self._resources = None
            vector_store = await self.cached_resource()
        except Exception as e:
            print(f"Unexpected Error: {e}")
            self._resources = None
            self._client = None
            return None

        return vector_store

def ttl_conversion_to_day(number_of_days: float):
    total = 86400 * number_of_days
    return total



