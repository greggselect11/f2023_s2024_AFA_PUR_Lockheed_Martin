
import langchain_community
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd

from transformers import BloomForCausalLM
from transformers import BloomForTokenClassification
from transformers import BloomForTokenClassification
from transformers import BloomTokenizerFast
import torch
from transformers import pipeline
import tensorflow as tf
import keras


import pyodbc
#Add your own SQL Server IP address, PORT, UID, PWD and Database
conn = pyodbc.connect(
    'DRIVER={PostgreSQL Unicode};SERVER=localhost;PORT=5432;DATABASE=postgres;UID=postgres;PWD=mysecretpassword', autocommit=True)
cur = conn.cursor()

# Update rows with empty attributes in Vulnerabilities table
cur.execute("UPDATE Vulnerabilities SET vulnerability_id = 'None' WHERE vulnerability_id = ''")
cur.execute("UPDATE Vulnerabilities SET description = 'None' WHERE description = ''")
cur.execute("UPDATE Vulnerabilities SET severity = 'None' WHERE severity = ''")
cur.execute("UPDATE Vulnerabilities SET required_action = 'None' WHERE required_action = ''")

# Update rows with empty attributes in AffectedProducts table
cur.execute("UPDATE AffectedProducts SET vulnerability_id = 'None' WHERE vulnerability_id = ''")
cur.execute("UPDATE AffectedProducts SET product_name = 'None' WHERE product_name = ''")
cur.execute("UPDATE AffectedProducts SET version = 'None' WHERE version = ''")

# Update rows with empty attributes in ReferenceData table
cur.execute("UPDATE ReferenceData SET vulnerability_id = 'None' WHERE vulnerability_id = ''")
cur.execute("UPDATE ReferenceData SET url = 'None' WHERE url = ''")
cur.execute("UPDATE ReferenceData SET description = 'None' WHERE description = ''")

cur.execute("ALTER TABLE Vulnerabilities ALTER COLUMN published_date TYPE TEXT")
#cur.execute("UPDATE Vulnerabilities SET published_date = TO_CHAR(published_date, 'YYYY-MM-DD') WHERE published_date IS NOT NULL")

import pyodbc
#Add your own SQL Server IP address, PORT, UID, PWD and Database
conn = pyodbc.connect(
    'DRIVER={PostgreSQL Unicode};SERVER=localhost;PORT=5432;DATABASE=postgres;UID=postgres;PWD=mysecretpassword', autocommit=True)
cur = conn.cursor()

cur.execute("SELECT * FROM Vulnerabilities WHERE published_date > '2024-01-01' LIMIT 100")
rows = cur.fetchall()
for row in rows:
    print(row)
    print("\n\n\n")

from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri("postgresql://postgres:mysecretpassword@localhost")


import torch
import time
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain_experimental.sql import SQLDatabaseSequentialChain
from langchain.chains import create_sql_query_chain
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import notebook_login
 
#from transformers import BloomTokenizerFast
#tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom", add_prefix_space=True, is_split_into_words=True)

#tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
token = ""


model_id="meta-llama/Llama-2-70b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100
)
print("Made it here 1: /n")
llm = HuggingFacePipeline(pipeline=pipe)

#instruct_pipeline = pipeline(model="meta-llama/Llama-2-70b-chat-hf",token = token, trust_remote_code=True, use_auth_token=True, use_fast=False, device_map="auto", return_full_text=True, do_sample=False, max_new_tokens=128)
#hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)
#chain = SQLDatabaseSequentialChain.from_llm(llm=hf_pipe, db=db, verbose=True)
chain = create_sql_query_chain(llm=llm, db=db)
response = chain.invoke({"question": "How many new CVES since January of 2024?"})
print("Made it here 2: /n")
print(response)