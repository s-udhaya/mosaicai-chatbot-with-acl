# Databricks notebook source
# MAGIC %md
# MAGIC # Test Endpoint
# MAGIC
# MAGIC ###  🛑 The deployed endpoint should be up and runing before this notebook is executed

# COMMAND ----------

# load config
import yaml
from pathlib import Path

conf = yaml.safe_load(Path("./config/rag_chain_config.yaml").read_text())
databricks_resources = conf.get("databricks_resources")

# COMMAND ----------

import os
import requests
import json

## Defining endpoint name
serving_endpoint_name = databricks_resources.get("chatbot_endpoint_name")

## Getting the workspace host name
host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"


def call_endpoint(query: str, department: str):
    url = f"{host}/serving-endpoints/{serving_endpoint_name}/invocations"

    headers = {
        "Authorization": f"Bearer <fill here your databricks PAT or OAUTH token>",
        "Content-Type": "application/json",
    }

    custom_inputs = {
        "filters": {
            "departments": [department],
        }
    }
    model_input_sample = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ],
        "custom_inputs": custom_inputs,
    }

    data_json = json.dumps(
        model_input_sample,
    )
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)

    if response.status_code != 200:
        return f"{response.text}"
    return json.loads(response.text)

# COMMAND ----------

call_endpoint(query="Can you tell me about ABC company's HR department?", department="HR")

# COMMAND ----------

call_endpoint(query="Can you tell me about ABC company's HR department?", department="Finance")
