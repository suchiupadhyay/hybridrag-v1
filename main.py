## lightweight FastAPI app that imports hybridrag.py functions and exposes HTTP endpoints

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import uvicorn
import os
from hybridRAG_withRedis import rag_with_cache  # your core logic
import os
import redis
import httpx

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
WHATSAPP_API_URL = "https://graph.facebook.com/v18.0"

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





# [Receive WhatsApp Messages]
# WhatsApp will send POST requests here whenever a user sends you a message.
# You can extract the message from the payload and trigger your RAG logic or auto-response.

## Full POST handle
@app.post("/webhook")
async def receive_whatsapp_message(request: Request):
    data = await request.json()
    print("📩 Incoming WhatsApp webhook:", data)

    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]["value"]
        messages = changes.get("messages")

        if not messages:
            return {"status": "ignored"}  # Ignore non-message events

        message = messages[0]
        sender = message["from"]
        user_message = message["text"]["body"]
        phone_number_id = changes["metadata"]["phone_number_id"]

        print(f"User ({sender}) said: {user_message}")

        # Generate RAG answer
        answer = rag_with_cache(user_message, redis_client=redis_client)

        # Prepare reply payload
        send_url = f"{WHATSAPP_API_URL}/{phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {VERIFY_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "messaging_product": "whatsapp",
            "to": sender,
            "type": "text",
            "text": {"body": answer}
        }

        # Send reply asynchronously
        async with httpx.AsyncClient() as client:
            response = await client.post(send_url, headers=headers, json=payload)
            print("✅ Sent reply:", response.status_code, response.text)

    except Exception as e:
        print("❌ Error handling webhook:", str(e))

    return {"status": "processed"}

        ## Simple webhook POST : from Whatsapp Business Account(mobile msg) to verify in RAilway deploy log : 
        # @app.post("/webhook")
        # async def receive_whatsapp_message(request: Request):
        #     data = await request.json()
        #     print("📩 Incoming WhatsApp message:", data)
        #     return {"status": "received"}




## root /  and query - RAG chatbot standalone endpoints...setup upto Railway  deployment
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

