from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
from tts import stream_tts
from query_engine.classifier_engine import classify_domain, classify_intent
from query_engine.rag_pipeline import ask, load_all_vectorstores

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstores = load_all_vectorstores()

with open ("query_engine/intents.json", "r", encoding = "utf-8") as file:
    all_intents = json.load(file)

@app.websocket("/ws/query")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()

            print(f"[User]: {query}")

            domain = classify_domain(query)
            intent = classify_intent(query, domain)

            if domain and intent : response = all_intents.get(domain).get(intent)
            elif domain : response = ask(query, vectorstores[domain])       
            else: response = ask(query, vectorstores["banking"])

            print(f"[Maya]: {response}")

            await websocket.send_text(json.dumps({"type": "text", "data": response}))

            async for chunk in stream_tts(response):
                await websocket.send_bytes(chunk)
    except Exception as e:
        print("WebSocket disconnected:", e)