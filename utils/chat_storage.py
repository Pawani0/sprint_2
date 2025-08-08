from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import os
from typing import Dict
from query_engine.rag_pipeline import session_store
from pymongo import MongoClient

IST = timezone(timedelta(hours=5, minutes=30))

load_dotenv()
conversation_log: Dict[str, list] = {}

client = MongoClient(os.getenv("MONGO_URI"))
db = client["Fincov_db"]
collection = db["conversations"]

def store_message(SID: str, query: str, response: str, domain: str, intent: str):
    if domain is None:
        domain = "general"
    if intent is None:
        intent = "unknown"
    if SID not in conversation_log:
            conversation_log[SID] = []
            
    conversation_log[SID].append({
                "timestamp": datetime.now(IST).isoformat(),
                "domain": domain,
                "intent": intent,
                "user": query,
                "assistant": response,
            })

def dump_session_to_mongo(session_id):

    document = {
        "session_id": session_id,
        "timestamp": datetime.now(IST).isoformat(),
        "info": get_session_info(session_id),
        "messages": conversation_log.get(session_id, [])
    }
    result = collection.insert_one(document)
    print(f"[MongoDB] Session stored with ID: {session_id}")
    clear_session(session_id)
    print(f"[MongoDB] Cleared session data for ID: {session_id}")

# Clear session memory
def clear_session(call_sid: str):
    """Clear the conversation history for a specific session"""
    if call_sid in session_store:
        session_store.pop(call_sid, None)
    if call_sid in conversation_log:
        conversation_log.pop(call_sid, None)

# Get session history (for debugging purposes)
def get_session_info(call_sid: str):
    """Get information about a specific session"""
    if call_sid in conversation_log:
        history = conversation_log[call_sid]
        return f"Session {call_sid} has {len(history)} messages"
    else:
        return f"No session found for SID: {call_sid}"