import os
import json
from groq import Groq
from dotenv import load_dotenv

# loading the credentials and connecting to the Groq API.

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# loading all the intets from the json file.

with open ("query_engine/intents.json", "r", encoding = "utf-8") as file:
    all_intents = json.load(file)

# domain classification using llm.

def classify_domain(query : str):
    prompt = "you are a domain classifier for financial domain classification. Your goal is to classify the user query into the most suitable or relavent domains from : banking, loan, investment, insurance, tax. Respond only with the domain name."
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt.strip()},
                  {"role": "user", "content": query}],
        model="llama3-70b-8192",
        temperature=0.2,
        max_tokens=10,
    )
    domain_text = response.choices[0].message.content.strip().lower()
    return domain_text if domain_text in ["banking", "loan", "investment", "insurance", "tax"] else None

# intent classification using llm.

def classify_intent(query : str, domain : str):
    if domain == None : return None
    domain_intents = all_intents.get(domain).keys()
    prompt = f"You are an intent classifier for the domain : {domain}. Your goal is to classify user query into the most suitable or best fit intent from : {', '.join(domain_intents)}. Respond only with the intent name or else give 'unknown' if no intent matches or confidence is low."
    response = client.chat.completions.create(
        messages = [{"role": "system", "content": prompt.strip()},
                    {"role": "user", "content": query}],
        model = "llama3-70b-8192",
        temperature = 0.2,
        max_tokens = 15,
    )
    intent = response.choices[0].message.content.strip().lower()
    if intent == "unknown" : intent = None
    return intent
