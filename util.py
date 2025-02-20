from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings
import openai
from langchain_community.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:%(message)s:%(funcName)s')
file_handler = logging.FileHandler('s3_draft_missing.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
load_dotenv()

qdrant_url=os.getenv('qdrant_url')
qdrant_api_key=os.getenv('qdrant_api_key')
User = os.environ.get('user')
Password = os.environ.get('password')
Host = os.environ.get('host')
Port = os.environ.get('port')
Database = os.environ.get('dbname')
embed_fn = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
open_api_key  = os.getenv("OPENAI_API_KEY")
chat_llm=ChatOpenAI(model="gpt-3.5-turbo",api_key=open_api_key)

class IncomingFileProcessor():
    def __init__(self, chunk_size=750) -> None:
        self.chunk_size = chunk_size
        user_id: str
        name: int
        user_query: str
        system_response: str
def load_local_vectordb_using_qdrant(vectordb_folder_path):
    qdrant_client = QdrantClient(
        url=qdrant_url, 
        # prefer_grpc=True,
        api_key=qdrant_api_key,
    )
    qdrant_store= Qdrant(qdrant_client, vectordb_folder_path, embed_fn)
    return qdrant_store  

def db_connection():
    try: 
        conn = psycopg2.connect(user = User, password = Password, host = Host, port = Port, database = Database)
        logger.info("Successfully connected the database")
    except psycopg2.Error as e: 
        logger.critical(f"Error: connection can't establish: {e}")
        raise Exception("Error: Could not make connection to the Postgres database")

    try: 
        cur = conn.cursor()
        logger.info("Successfully connected the cursor")

    except psycopg2.Error as e: 
        logger.critical(f"Error: can't connect cursor: {e}")
        raise Exception("Error: Could not get curser to the Database")
        
    conn.set_session(autocommit=True)
    return cur, conn
    
def retrive_check_history(user_id,cur):
    try:
        user_id = int(user_id)
        check_prev_record = f"""
                            SELECT query, response FROM history WHERE user_id = '{user_id}' ORDER BY "Date" DESC LIMIT 5;
                        """
        cur.execute(check_prev_record)
        responses = cur.fetchall()
        return responses 
    except Exception as e:
        logger.critical("Failed ")  


def save_history(user_id ,query , response , cur, conn):
    try:
        user_id = int(user_id)
        query = query
        response = response
        save_history = f"""
                        INSERT INTO public.history(user_id, response, "Date", query)
                                                    VALUES ('{user_id}','{response}',NOW(),'{query}')
                        """
        cur.execute(save_history)
        conn.commit()
        logger.info("Successfully saved the history")

    except Exception as e:
        logger.critical("Failed to save the history")


def conversation_retrieval_chain(query, vectordb,vectordb_name,user_id=1):
    num_chunks = 5
    cur, conn = db_connection()
    chat_history = retrive_check_history(user_id, cur)
    
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(_template)
    
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
    
    qdrant_store = load_local_vectordb_using_qdrant(vectordb_name)
    retriever = qdrant_store.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
    
    def _combine_documents(docs):
        return "\n\n".join(docs)

    _inputs = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    )
    
    _context = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }
    
    if len(chat_history) == 0:
        rephrase_question = query
    else: 
        rephrase_question = _inputs.invoke({
            "question": query,
            "chat_history": chat_history
        })
    
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    
    context = setup_and_retrieval.invoke(rephrase_question)
    docs = context['context']
    doc_list = {'context': docs, "question": rephrase_question}
    doc_dict = {
    'context': "\n\n".join([doc.page_content for doc in doc_list['context']]),
    'question': doc_list['question']
    }

    answer_prompt= ANSWER_PROMPT.invoke(doc_dict)
    output_parser= StrOutputParser()
    final_response= chat_llm.invoke(answer_prompt)
    response= output_parser.invoke(final_response)
    
    save_history(user_id, query ,response, cur, conn)
    return response
    

def get_buffer_string(chat_history):
    return "\n".join([f"User: {entry[0]}\nBot: {entry[1]}" for entry in chat_history])

def format_document(doc, document_prompt):
    return document_prompt.format(page_content=doc.page_content)


