# Import Libs :
import os
import fitz 
import re
import langchain_text_splitters
import faiss
import redis
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv




## Verify proper fitz installed or not :
print(fitz.__file__)
print(fitz.__version__ if hasattr(fitz, '__version__') else 'No version')
print(dir(fitz))


## pdf extraction
doc = fitz.open("AMFI_MF.pdf")
text = ""
for page in doc:
    text += page.get_text("text")

print(text)

## Pre-processing text 

# collapse all whitespace to single space
documents = re.sub(r'\s+', ' ', text)

# remove isolated characters (except numbers or words)
documents = re.sub(r'\b\w\b', '', documents)

documents = documents.strip()

print(documents)

## Length of documents
print()
print("Length of extracted document after pre-processing : ", len(documents))


## character-based chunks with overlap:

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30,
                                          separators=["\n\n", "\n", " ", ""])
chunks = splitter.split_text(documents)


print(f"Total chunks: {len(chunks)}")
print("Preview first chunk --> ", chunks[0])  # preview first chunk


## Create embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embedding = model.encode(chunks, convert_to_numpy=True).astype('float32')  # FAISS requires float32

doc_embedding

doc_embedding.shape


# Build FAISS index
dim = doc_embedding.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(doc_embedding)


## Redis connect to server :
redis_client = redis.Redis(host='localhost', port=6379, db=0)
print(redis_client.ping())  ## verfiy redis-server is running



## RAG - 

doc_texts = chunks

def retrieve_docs(query, top_k=1):
    print("From fucntion retrive_docs :", query)
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), top_k)
    print("Distances:", D)
    print("Indexes:", I[0])
 
    retrieved = [doc_texts[i] for i in I[0]]
    print("retrived from function --> : ", retrieved)
    return retrieved


def generate_answer(query, context):
    return f"Based on context: {context[0]} \nAnswer: {query} → {context[0]}"
        
    # combined_context = " ".join(context)
    # return f"Based on context: {combined_context}\nAnswer: {query} → {combined_context}"



## redis get and set

def rag_with_cache(query):
    # Step 1: Check Redis cache
    print("Query from function: ", query)
    cached_answer = redis_client.get(query)
    if cached_answer:
        print("[Answer from Redis] :")
        return f"[CACHE HIT] {cached_answer.decode()}"

    # Step 2: If not cached → do RAG
    context = retrieve_docs(query)
    answer = generate_answer(query, context)
    print("Answer from RAG - Knowledge base : ",answer )

    # Step 3: Store in Redis cache
    #redis_client.set(query, answer)  # expires in 60s
    redis_client.set(query, answer, ex=60)  # expires in 60s
    return f"[RAG GENERATED] {answer}"


## Use LLM to response 

## to run LLM we need gemini API 
load_dotenv()

# Your API Key setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("Google API Key is missing. Please set it in the .env file.")

# Configure the Generative AI library with the API key
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("api key configured")


## Now use LLM

def generate_with_gemini(query, retrieved_chunks):
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Combine chunks into a single context
    context = "\n\n".join(retrieved_chunks)

    # Prepare the prompt
    prompt = f"""
                You are a financial assistant AI. Use only the context provided below to answer the user query clearly and concisely.

                Instructions:
                - First, look ONLY in the context below to find an answer.
                - If you find relevant information in the context, use it to answer the question clearly and concisely.
                - BUT — if the context does not contain enough information, then you may generate a concise, general answer using your own financial knowledge.
                - DO NOT include any disclaimers or document headers.
                - If the user's question is unrelated to finance, politely respond:
                  "I'm here to assist with financial topics. Could you please ask a question related to investments or mutual funds?"
                - Keep your answers short, informative, and beginner-friendly.



    CONTEXT:
    {context}

    USER QUESTION:
    {query}

    Answer:
    """

    # Generate response
    response = model.generate_content(prompt)

    return response.text.strip()




## run your query :
query ="what is open-ended schemes?"
#query ="what is open-ended schemes "
# print(query)
answer = rag_with_cache(query)
#print(answer)

response = generate_with_gemini(query, answer)
print()
print("Response from LLM --> ")
print(response)


## run your query :
query ="what ingrediants required to cook Biryani?"
answer = rag_with_cache(query)
print(answer)

response = generate_with_gemini(query, answer)
print()
print("Response from LLM --> ")
print(response)


## run your query :
query ="Can MF provide any type of insurance or sip with insurance?"
answer = rag_with_cache(query)
print(answer)

response = generate_with_gemini(query, answer)
print()
print("Response from LLM --> ")
print(response)

'''
## run multiple queries :

queries = ["Suggest different types of funds available " ,
           "what is Balanced or hybrid funds",
           "who is Authorities to handle mf" ,
           "how to start investing in MF industries",
           "give me some introduction about mf",
           "what is Balanced or hybrid funds",
            "what is FoF Scheme" ]

print(" *** Demo *** RAG+CAG with or without Latency \n ")


for q in queries:
    print(f"\nQ: {q}")  
    answer = rag_with_cache(q)
    print(answer)
    response = generate_with_gemini(q, answer)
    print()
    print("Response from LLM --> ")
    print(response)
'''    