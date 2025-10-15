## lightweight FastAPI app that imports hybridrag.py functions and exposes HTTP endpoints

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
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


VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

app = FastAPI()


# [Webhook Verification]
# Meta (WhatsApp) sends a GET request here during webhook setup.
# It includes a "hub.verify_token" which must match your VERIFY_TOKEN.
# If matched, respond with "hub.challenge" to confirm ownership.

@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("✅ Webhook verified successfully")
        return PlainTextResponse(content=challenge, status_code=200)
    else:
        print("❌ Webhook verification failed")
        return PlainTextResponse(content="Verification failed", status_code=403)



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


# [Receive WhatsApp Messages]
# WhatsApp will send POST requests here whenever a user sends you a message.
# You can extract the message from the payload and trigger your RAG logic or auto-response.

@app.post("/webhook")
async def receive_whatsapp_message(request: Request):
    data = await request.json()
    print("📩 Incoming WhatsApp message:", data)
    return {"status": "received"}


## run locally with python main.py ... 
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))  # Railway provides a dynamic port
#     uvicorn.run("main:app", host="0.0.0.0", port=8000)

