# Databricks notebook source
# MAGIC %md
# MAGIC # Source data preparation
# MAGIC To simplify this blog post, we've stored sample data in a CSV file. We'll load this CSV file into a Delta table and use it as the source for our vector search index. In real-world scenarios, you might encounter diverse data sources and formats like PDFs, PPTs, and Confluence pages.  This often requires writing custom data parsing and chunking techniques to process the data into a usable format, similar to the sample data in this example. For guidance on handling different file formats, refer to the [Databricks GenAI cookbook](https://github.com/databricks/genai-cookbook/blob/v0.2.0/rag_app_sample_code/A_POC_app/pdf_uc_volume/02_poc_data_pipeline.py).

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🛑 Make sure to update the `./config/rag_chain_config.yaml` with the relevant values.

# COMMAND ----------

import yaml
from pathlib import Path

source_data_file_path = "./sample_data/source_data.csv"

conf = yaml.safe_load(Path("./config/rag_chain_config.yaml").read_text())
databricks_resources = conf.get("databricks_resources")

# COMMAND ----------

# load the chunked data as a spark dataframe
import os

sdf = (
    spark.read.format("csv")
    .option("header", "true")
    .option("sep", ";")
    .option("multiline", True)
    .load(f"file:{os.getcwd()}/{source_data_file_path}")
)
display(sdf)

# COMMAND ----------

# write the chunked data to a delta table
sdf.write.format("delta").mode("overwrite").saveAsTable(
    f"{databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.{databricks_resources.get('chunked_data_table')}"
)

# COMMAND ----------

# Enable CDC for Vector Search Delta Sync
spark.sql(
    f"ALTER TABLE {databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.{databricks_resources.get('chunked_data_table')} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
