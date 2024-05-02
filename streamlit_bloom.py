import os
import json
import openai
import pandas as pd
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BloomForCausalLM
from transformers import BloomForTokenClassification
from transformers import BloomForTokenClassification
from transformers import BloomTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

## CONSTANTS
MODEL_NAME_OR_PATH = "C:/Users/Andrew/.cache/huggingface/hub/models--bigscience--bloomz-560m"
DATA_PATH = "../data/"

## Clear cuda cache ?
# torch.cuda.empty_cache()

## Load dotenv file
load_dotenv(dotenv_path="../.env")

## Load API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_and_process_json_data(file_path):
    
    df = pd.read_json(file_path)
    
    ## Convert date fields to datetime format
    df['dateAdded'] = pd.to_datetime(df['dateAdded'])
    df['dueDate'] = pd.to_datetime(df['dueDate'])
    
    ## Handle missing vals; if cvss score is not available, drop the row
    df = df.dropna(subset=['cvss'])
    
    ## Handle any remaining missing text fields
    text_columns = ['vendorProject', 'product', 'vulnerabilityName', 'shortDescription', 'requiredAction', 'knownRansomwareCampaignUse', 'notes']
    for col in text_columns:
        df[col] = df[col].fillna("UNKNOWN")
    
    print(df.describe())
    
    return df

def convert_df_to_documents(df):
    
    documents = [Document(content=str(row.to_dict())) for index, row in df.iterrows()]
    return documents

def haystack_llm_query(question, chat_history):
    vulnerabilities_df = load_and_process_json_data("../data/cvss_vulnerabilities.json")
    documents = convert_df_to_documents(vulnerabilities_df)
    
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    
    ## Build a RAG pipeline
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
    
    # ## Check whether a GPU is available
    # if torch.cuda.is_available():
    #     torch.set_default_device('cuda')
    #     torch.set_default_dtype(torch.float32)
    #     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # else:
    #     print("Using CPU")
    
    # ## Load a local LLM model
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    # llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm = OpenAIGenerator() ## Try OpenAI 
    
    # ### Force the model to load to GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    
    
    ## Build the pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    
    ## Generate response
    question = f"{question} The previous questions and responses were: {str(chat_history)}"
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




# def create_document_store(df):
    
#     document_store = InMemoryDocumentStore()
    
#     documents = []
#     for index, row in df.iterrows():
#         documents.append(Document(text=row['notes'], meta={"vulnerabilityName": row['vulnerabilityName'], "cvss": row['cvss']}))
    
#     document_store.write_documents(documents)
    
#     return document_store