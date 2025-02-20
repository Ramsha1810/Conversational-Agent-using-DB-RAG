from util import *
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# from langchain_community.embeddings import HuggingFaceEmbeddings

import os
import openai
from dotenv import load_dotenv
import uvicorn
from fastapi import status




app = FastAPI()

load_dotenv()

origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
    )


@app.get("/")
def startup():
    return JSONResponse("Conversation using DB")


@app.post("/conversational_lcel_chain_with_db")
async def conversationChainLCEL(query:str, id:int):
    try:
        vectordb_name= 'xevensolutions.pdf2024-07-08'
        vectordb= load_local_vectordb_using_qdrant(vectordb_name) 
        response = conversation_retrieval_chain(query, vectordb,vectordb_name,user_id=1)
        return JSONResponse(content={"message": "Response Generated Successfully!", "Response": response}, status_code=status.HTTP_200_OK)
    except Exception as ex:
        return JSONResponse(content={"error": str(ex)}, status_code=status.HTTP_400_BAD_REQUEST)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=9393)