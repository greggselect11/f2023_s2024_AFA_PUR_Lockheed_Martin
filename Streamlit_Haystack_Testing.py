import os
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceTGIGenerator
import pyodbc

from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import ConversationalRetrievalChain
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
import streamlit as st

def sql_data_load():

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

    conn = pyodbc.connect(
        'DRIVER={PostgreSQL Unicode};SERVER=localhost;PORT=5432;DATABASE=postgres;UID=postgres;PWD=mysecretpassword', autocommit=True)
    cur = conn.cursor()

    cur.execute("SELECT * FROM Vulnerabilities")
    rows = cur.fetchall()
    return rows


def haystack_llm_query(question, chat_history):
    # Set the environment variable OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = "insert your own key here"

    # Write documents to InMemoryDocumentStore
    document_store = InMemoryDocumentStore()
    rows = sql_data_load()
    for row in rows:
        document_store.write_documents([
            Document(content=str(row)), 
        ])

    # Build a RAG pipeline
    prompt_template = """
    Given these documents, answer the question.
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    Question: {{question}}
    Answer:
    """

    retriever = InMemoryBM25Retriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=prompt_template)
    llm = OpenAIGenerator()

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)
    #rag_pipeline.add_component("llm", HuggingFaceTGIGenerator(model='mistralai/Mistral-7B-v0.1'))
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    # Ask a question
    question = str(question) + """, 
    If I am asking for a threat report, use the following example as a format:

    Threat Title: 
    Summary: 
    Impact: 
    CVE: 

    Otherwise, just speak in plain english.
    """ + "The previous questions and responses were: " + str(chat_history)
    results = rag_pipeline.run(
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
    )

    return results

if __name__ == '__main__':

    st.header("QA ChatBot")
    # ChatInput
    prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
    if "chat_history" not in st.session_state:
       st.session_state["chat_history"]=[]

    if prompt:
       with st.spinner("Generating......"):
           output=haystack_llm_query(question=prompt, chat_history = st.session_state["chat_history"])

          # Storing the questions, answers and chat history

           st.session_state["chat_answers_history"].append(output["llm"]["replies"])
           st.session_state["user_prompt_history"].append(prompt)
           st.session_state["chat_history"].append((prompt,output["llm"]["replies"]))

    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
       for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)