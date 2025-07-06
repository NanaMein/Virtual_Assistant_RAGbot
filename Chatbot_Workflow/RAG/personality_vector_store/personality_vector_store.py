# def milvus_context_history_cohere():
#     vector_store = None  # this will act as our private "cached" value
#     def get_zilliz():
#         nonlocal vector_store  # reference the outer variable
#         if vector_store is None:
#             print("Initializing Baby_Mirai_and_Mio_v1 Vector Store With Cohere...")  # for debug
#             vector_store = MilvusVectorStore(
#                 uri=os.getenv('NEW_URI'),
#                 token=os.getenv('NEW_TOKEN'),
#                 collection_name='Baby_Mirai_and_Mio_v1',
#                 dim=1536,
#                 embedding_field='embeddings',
#                 enable_sparse=True,
#                 enable_dense=True,
#                 overwrite=False,  # CHANGE IT FOR DEVELOPMENT STAGE ONLY
#                 sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
#                 search_config={"nprobe": 20},
#                 similarity_metric="IP",
#                 consistency_level="Session",
#                 hybrid_ranker="WeightedRanker",
#                 hybrid_ranker_params={"weights": [0.75, 1.0]},
#             )
#         return vector_store #, embed_model
#
#     return get_zilliz
# mc = milvus_context_history_cohere()