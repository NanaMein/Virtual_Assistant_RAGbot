# import litellm
# litellm._turn_on_debug()
import asyncio
from collections import deque
from functools import lru_cache
from typing import Deque, Type, Optional, List
from crewai.agents.agent_builder.base_agent import BaseAgent
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
from llama_index.llms.groq import Groq
from datetime import datetime, timezone, timedelta
from crewai.flow import Flow, start, listen, router, or_, and_, persist
from llama_index.core.base.llms.base import ChatMessage  # schema import ChatMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import tool, BaseTool
from crewai.project import CrewBase, task, agent, crew
from cachetools import TTLCache, cached
import os
from collections import defaultdict
import pytz
from llama_index.core.schema import MetadataMode
import time
import grpc
from pymilvus import connections
from pymilvus.exceptions import MilvusException


load_dotenv()



class InitialAssessmentCrew:

    for_cache = TTLCache(ttl=3600, maxsize=100)
    vector_cache = TTLCache(ttl=300, maxsize=10)
    agents = List[BaseAgent]
    tasks = List[Task]

    def __init__(self, user_id: str):
        self.user_id: str = user_id


    agent_config = "config/agents.yaml"
    task_config = "config/tasks.yaml"

    @agent
    def agent(self):
        return Agent(
            config=self.agent_config["research_agent"], # type: ignore[index]
            verbose=True,
        ) # type: ignore[index]

    @task
    def task(self):
        return Task(
            config=self.task_config["task"],

        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,  # type: ignore[index]
            tasks=self.tasks,  # type: ignore[index]
            process=Process.sequential,
            planning=True
        )


