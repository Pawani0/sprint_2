from twilio.rest import Client
from dotenv import load_dotenv
import os

# Ensure .env file is loaded from the correct path
load_dotenv(override=True)

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
verify_sid = os.getenv("TWILIO_VERIFY_SID")

client = Client(account_sid, auth_token)

def send_verification_code(phone_number: str):
    # Ensure phone number is in E.164 format
    phone_number = phone_number.replace(" ", "").replace("-", "")
    if not phone_number.startswith("+"):
        if phone_number.startswith("91"):
            phone_number = "+" + phone_number
        else:
            phone_number = "+91" + phone_number
            
    verification = client.verify.v2.services(verify_sid).verifications.create(
        to=phone_number,
        channel="sms"
    )
    return verification.status

def check_verification_code(phone_number: str, code: str):
    verification_check = client.verify.v2.services(verify_sid).verification_checks.create(
        to=phone_number,
        code=code
    )
    return verification_check.status == "approved"
