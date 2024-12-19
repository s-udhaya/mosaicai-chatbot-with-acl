# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks vector search index creation
# MAGIC This notebook assumes that you have your data prepared and written to a delta table with the following schema.
# MAGIC
# MAGIC <code>
# MAGIC path:string <br>chunked_text:string<br>chunk_id:string<br>departments:string
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install libraries & import packages

# COMMAND ----------

# MAGIC %pip install -qqqq -U databricks-vectorsearch 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Embed documents & sync to Vector Search index

# COMMAND ----------

# load config
import yaml
from pathlib import Path

conf = yaml.safe_load(Path("./config/rag_chain_config.yaml").read_text())
databricks_resources = conf.get("databricks_resources")

# COMMAND ----------

# DBTITLE 1,Index Management Workflow
# create vector search index
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# index creation will take a couple of minutes, wait until the index is created
try:
    index = vsc.create_delta_sync_index(
        endpoint_name=databricks_resources.get("vector_search_endpoint_name"),
        index_name=f"{databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.{databricks_resources.get('vector_search_index')}",
        primary_key="chunk_id",
        source_table_name=f"{databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.{databricks_resources.get('chunked_data_table')}",
        pipeline_type="triggered",
        embedding_source_column="chunked_text",
        embedding_model_endpoint_name=databricks_resources.get(
            "embedding_endpoint_name"
        ),
    )
    display(index.describe())
except Exception as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print("Index already exists. Skipping index creation.")
    else:
        raise (e)

# COMMAND ----------

# MAGIC %md
# MAGIC ###  🛑 index creation will take a couple of minutes, wait until the index is created

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the index

# COMMAND ----------

# DBTITLE 1,Testing the Index
# test querying the index
index = vsc.get_index(
    databricks_resources.get("vector_search_endpoint_name"),
    f"{databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.{databricks_resources.get('vector_search_index')}",
)
index.similarity_search(
    columns=["chunked_text", "chunk_id", "path"], query_text="your query text"
)
