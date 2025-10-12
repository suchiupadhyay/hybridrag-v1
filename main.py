## lightweight FastAPI app that imports hybridrag.py functions and exposes HTTP endpoints

from fastapi import FastAPI, Request
import uvicorn
import os
from hybridRAG_withRedis import rag_with_cache  # your core logic
import os
import redis

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


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello from Railway"}

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

## run locally with python main.py ... 
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))  # Railway provides a dynamic port
#     uvicorn.run("main:app", host="0.0.0.0", port=8000)

