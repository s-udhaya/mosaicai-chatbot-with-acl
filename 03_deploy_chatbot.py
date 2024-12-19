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

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Can you tell me about ABC company's HR department?",  # Replace with a question relevant to your use case
        }
    ],
    "custom_inputs": {"filters": {"departments": "Finance"}},
}


# COMMAND ----------

import os
import mlflow
from mlflow.models import infer_signature
from mlflow.types.llm import CHAT_MODEL_INPUT_SCHEMA
from mlflow.models.rag_signatures import StringResponse
from mlflow.models.signature import ModelSignature
import mlflow
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksVectorSearchIndex,
)
from mlflow.models.rag_signatures import ChatCompletionResponse
from mlflow.models import infer_signature
from dataclasses import asdict

signature = infer_signature(asdict(CustomChatCompletionRequest()), StringResponse())

mlflow.langchain.autolog()


resources = [
    DatabricksServingEndpoint(
        endpoint_name=databricks_resources.get("embedding_endpoint_name")
    ),
    DatabricksServingEndpoint(
        endpoint_name=databricks_resources.get("llm_endpoint_name")
    ),
    DatabricksVectorSearchIndex(
        index_name=f"{databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.{databricks_resources.get('vector_search_index')}",
    ),
]

with mlflow.start_run():
    model_info = mlflow.langchain.log_model(
        # Pass the path to the saved model file
        os.path.join(
            os.getcwd(),
            "02_single_turn_chatbot_with_acl",
        ),
        "agent",
        model_config="./config/rag_chain_config.yaml",
        input_example=input_example,
        signature=signature,
        pip_requirements=[
            "mlflow",
            "databricks-langchain",
            "databricks-vectorsearch",
        ],
        resources=resources,
        registered_model_name=f"{databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.{databricks_resources.get('model_name')}",  
    )

# COMMAND ----------

# validate the model serving request and response before deployment
from mlflow.models import convert_input_example_to_serving_input, validate_serving_input

serving_input = convert_input_example_to_serving_input(
    input_example
)
validate_serving_input(model_info.model_uri, serving_input=serving_input)

# COMMAND ----------

from databricks import agents

# Deploy the model to the review app and a model serving endpoint
agents.deploy(
    f"{databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.{databricks_resources.get('model_name')}",
    model_info.registered_model_version, endpoint_name=databricks_resources.get("chatbot_endpoint_name")
)
