# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Define custom schema to incorporate ACL filters in the chatbot request

from dataclasses import dataclass, field, asdict
from mlflow.models.rag_signatures import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    StringResponse,
)
from typing import Optional, Dict

@dataclass
class CustomInputs():
    filters: Dict[str, str] = field(default_factory=lambda: {"departments": "*"})


# Additional input fields must be marked as Optional and have a default value
@dataclass
class CustomChatCompletionRequest(ChatCompletionRequest):
    custom_inputs: Optional[CustomInputs] = field(default_factory=CustomInputs)

# COMMAND ----------

# load the config
import mlflow

model_config = mlflow.models.ModelConfig(
    development_config="./config/rag_chain_config.yaml"
)

databricks_resources = model_config.get("databricks_resources")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

# COMMAND ----------

# Define chatbot agent with ACL

from operator import itemgetter
import mlflow
import os
from typing import Dict, List, Optional
from databricks.vector_search.client import VectorSearchClient

from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain_core.runnables import RunnableLambda, RunnableGenerator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import ConfigurableField
from typing import Iterator, Dict, Any
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    MessageLikeRepresentation,
)
from mlflow.models.rag_signatures import (
    ChatCompletionResponse,
    ChainCompletionChoice,
    Message,
)
from random import randint
from dataclasses import asdict
import logging

import json

## Enable MLflow Tracing
mlflow.langchain.autolog()


############
# Helper functions
############
# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# combine dynamic and static filters for vector search
def create_configurable_with_filters(input: Dict, retriever_config: Dict) -> Dict:
    """
    create configurable object with filters.

    Args:
        input: The input data containing filters.

    Returns:
        A configurable object with filters added to the search_kwargs.
    """
    if "custom_inputs" in input:
        filters = input["custom_inputs"]["filters"]
    else:
        filters = {}
    configurable = {
        "configurable": {
            "search_kwargs": {
                "k": retriever_config.get("parameters")["k"],
                "query_type": retriever_config.get("parameters")["query_type"],
                "filters": filters,
            }
        }
    }
    return configurable


############
# Connect to the Vector Search Index
############
vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(
    endpoint_name=databricks_resources.get("vector_search_endpoint_name"),
    index_name=f"{databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.{databricks_resources.get('vector_search_index')}",
)
vector_search_schema = retriever_config.get("schema")

############
# Turn the Vector Search index into a LangChain retriever
############
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column=vector_search_schema.get("chunk_text"),
    columns=[
        vector_search_schema.get("primary_key"),
        vector_search_schema.get("chunk_text"),
        vector_search_schema.get("document_uri"),
    ],
).as_retriever(search_kwargs=retriever_config.get("parameters"))

configurable_vs_retriever = vector_search_as_retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

############
# Required to:
# 1. Enable the RAG Studio Review App to properly display retrieved chunks
# 2. Enable evaluation suite to measure the retriever
############

mlflow.models.set_retriever_schema(
    primary_key=vector_search_schema.get("primary_key"),
    text_column=vector_search_schema.get("chunk_text"),
    doc_uri=vector_search_schema.get(
        "document_uri"
    ),  # Review App uses `doc_uri` to display chunks from the same document in a single view
)


############
# Method to format the docs returned by the retriever into the prompt
############
def format_context(docs):
    chunk_template = retriever_config.get("chunk_template")
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
            document_uri=d.metadata[vector_search_schema.get("document_uri")],
        )
        for d in docs
    ]
    return "".join(chunk_contents)


############
# Prompt Template for generation
############
prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions
            "system",
            llm_config.get("llm_system_prompt_template"),
        ),
        # User's question
        ("user", "{question}"),
    ]
)

############
# LLM for generation
############
model = ChatDatabricks(
    endpoint=databricks_resources.get("llm_endpoint_name"),
    extra_params=llm_config.get("llm_parameters"),
)

############
# RAG Chain
############
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": RunnablePassthrough()
        | RunnableLambda(
            lambda input: configurable_vs_retriever.invoke(
                extract_user_query_string(input["messages"]),
                config=create_configurable_with_filters(input, retriever_config),
            )
        )
        | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)

## Tell MLflow logging where to find your chain.
# `mlflow.models.set_model(model=...)` function specifies the LangChain chain to use for evaluation and deployment.  This is required to log this chain to MLflow with `mlflow.langchain.log_model(...)`.

mlflow.models.set_model(model=chain)

# COMMAND ----------

# test with HR filter
input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Can you tell me about ABC company's HR department?", # Replace with a question relevant to your use case
        }
    ],
    "custom_inputs": {"filters": {"departments": "HR"}},
}
chain.invoke(input_example)

# COMMAND ----------

# test with Finance filter
input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Can you tell me about ABC company's HR department?",  # Replace with a question relevant to your use case
        }
    ],
    "custom_inputs": {"filters": {"departments": "Finance"}},
}
chain.invoke(input_example)

# COMMAND ----------

# test with no filters
input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Can you tell me about ABC company's HR department?", # Replace with a question relevant to your use case
        }
    ],
}
chain.invoke(input_example)
