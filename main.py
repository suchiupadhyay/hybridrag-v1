## lightweight FastAPI app that imports hybridrag.py functions and exposes HTTP endpoints

from fastapi import FastAPI, Request
from hybridRAG_withRedis import rag_with_cache  # your core logic
import os
import redis

app = FastAPI()

# Setup Redis client (or pass it into your functions)
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=True
)

print("Testing Redis connection...")
pong = redis_client.ping()
print("Redis PING response:", pong)


@app.post("/query")
async def query_endpoint(request: Request):
    data = await request.json()
    query = data.get("query")
    if not query:
        return {"error": "No query provided"}


    # Call your function with redis_client injected or global
    answer = rag_with_cache(query, redis_client=redis_client)
    print("Answer sent back:", answer)  # log before sending response
    return {"answer": answer}
