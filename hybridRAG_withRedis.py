### Building Hybrid RAG + Redis + LLM (Hybrid : Lexical + Semantic Matching)
## Redis - using upstash


import fitz  # PyMuPDF
import re
import langchain_text_splitters
import faiss
import redis
import numpy as np
import nltk
import os
import google.generativeai as genai

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize,sent_tokenize

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

##verify fitz version
# print(fitz.__file__)
# print(fitz.__version__ if hasattr(fitz, '__version__') else 'No version')
# print(dir(fitz))

## stop repeteadly downloads 
#nltk.download("punkt")
#nltk.download('punkt_tab')

for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)


# ## Redis connection
# load_dotenv()

# redis_client = redis.Redis(
#                 host = os.getenv("REDIS_HOST"),
#                 port = int(os.getenv("REDIS_PORT")),
#                 password = os.getenv("REDIS_PASSWORD"),
#                 ssl=True
#                 )

# redis_client.set('foo', 'bar')
# value = redis_client.get('foo')
# print(value.decode())


## Gemini Key 
load_dotenv()

# Your API Key setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("Google API Key is missing. Please set it in the .env file.")

# Configure the Generative AI library with the API key
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("🔑 Gemini API Key (partial):", GOOGLE_API_KEY[:8])  
    print("api key configured")


## PDF Extracted
#doc = fitz.open("CleanBot_Robotic_Vacuum_Cleaner_FAQ.pdf")
doc = fitz.open("AMFI_MF.pdf")
print(doc)
text = ""
skip_keywords = [ "disclaimer", "table of contents", ]

for i, page in enumerate(doc):

    page_text = page.get_text("text")

    if i == 0:                          ## Specifically add this to avoid first page which is having disclaimer also
        # print("First page preview:\n")
        # print(doc[0].get_text("text"))
        continue

    if any(kw in page_text for kw in skip_keywords):
        continue        

    text += page_text

## Text Pre-processing
import re

# collapse all whitespace to single space
documents = re.sub(r'\s+', ' ', text)

# remove isolated characters (except numbers or words)
documents = re.sub(r'\b\w\b', '', documents)

documents = documents.strip()

print(len(documents))


## character-based chunks with overlap:

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,
                                          separators=["\n\n", "\n", " ", ""])
chunks = splitter.split_text(documents)


print(f"Total chunks: {len(chunks)}")
print(chunks[0])  # preview first chunk


## Create embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")
dense_embedding = model.encode(chunks, convert_to_numpy=True).astype('float32')  # FAISS requires float32

dense_embedding

dense_embedding.shape


# Build FAISS index

dim = dense_embedding.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(dense_embedding)


## Sparse setup
## Note: Sentence tokenized created 352 sentences from documents

tokenized_corpus = word_tokenize(documents) ## --> this is for documents 
print("total word after tokenized: ", len(tokenized_corpus))

## Preprocess chunks for BM25 (tokenize each chunk) --> because for Dense I used chunks so for sparse also I used chunks

tokenized_chunks = [word_tokenize(i) for i in chunks]

print("tokenized_chunks : ", len(tokenized_chunks))

bm25 = BM25Okapi(tokenized_chunks)


# Semantic retrieval (Dense)
def retrieve_dense(query, k=3):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), k)
    print ("Dense distance: ", D)
    return I[0]  # return indices only


# Lexical retrieval (Sparse using BM25)
def retrieve_sparse(query, k=3):
    tokenized_query = word_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    print(scores.shape) 
    I = np.argsort(scores)[::-1][:k]
    
    return I  # indices


## Hybrid retrieval logic (combine indices):

def hybrid_retrieve(query, k_dense=3, k_sparse=3):
    dense_indices = retrieve_dense(query, k_dense)
    print("dense_indicies : ", dense_indices)
    sparse_indices = retrieve_sparse(query, k_sparse)
    print("sparse_indicies : ", sparse_indices)

    # Union of indices (or you can do weighted fusion if desired)
    hybrid_indices = list(set(dense_indices) | set(sparse_indices))
    print("hybrid_indices : ", hybrid_indices)

    # Get the actual documents (from chunks/doc_texts/etc.)
    retrieved_docs = [chunks[i] for i in hybrid_indices]  # or doc_texts[i]
    print("retrieved_docs : ", retrieved_docs)
    return retrieved_docs


## Now use LLM

def generate_with_gemini(query, context_str):
    model = genai.GenerativeModel("gemini-2.5-flash")

   
    # Prepare the prompt
    prompt = f"""
                You are a financial assistant AI. Use only the context provided below to answer the user query clearly and concisely.

                Important rules:
                - ONLY use the context below to answer.
                - DO NOT include any disclaimers or headers.
                - If the answer is not present in the context, respond with: "I'm sorry, I don't have enough information to answer that."


    CONTEXT:
    {context_str}

    USER QUESTION:
    {query}

    Answer:
    """

    # Generate response
    response = model.generate_content(prompt)
    print()
    #print("Response from LLM --> ", response.text.strip())
    return response.text.strip()


# Answer Generation Function (can be improved later):

def generate_answer(query, context):
    context_str = "\n---\n".join(context)
    
    # Call Gemini LLM with query + context
    answer = generate_with_gemini(query, context_str)
    
    #return f"Based on the following context:\n{context_str}\n\nAnswer: {query} → [You can now call your LLM here]"
    return answer


# Final RAG with cache (use hybrid retrieval):

def rag_with_cache(query, redis_client, k_dense=3, k_sparse=3):
    # Step 1: Check Redis cache
    cached_answer = redis_client.get(query)
    if cached_answer:
        print(" [Redis Cache] :")
        #print(cached_answer.decode())
        return f"[CACHE HIT] {cached_answer.decode()}"
    
    # Step 2: Hybrid Retrieval (combine dense + sparse)
    context = hybrid_retrieve(query, k_dense, k_sparse)

    # Step 3: Generate answer
    answer = generate_answer(query, context)

     # Step 4: Cache the answer
    redis_client.set(query, answer, ex=60)  # cache for 60s
    return f"[RAG GENERATED] {answer}"




if __name__ == "__main__":
    import redis
    import os

    # Create redis_client locally for quick test
    redis_client = redis.Redis(
        host = os.getenv("REDIS_HOST"),
        port = int(os.getenv("REDIS_PORT")),
        password = os.getenv("REDIS_PASSWORD"),
        ssl=True
        )

    #query= "what is cut off timing and its role in determining the applicable NAV for transactions in Mutual Fund schemes?"
    query = "what in money-market in MF?"

    answer = rag_with_cache(query,redis_client=redis_client)  ## --> hybrid Retrival and LLM response
    print("answer : ", answer)
