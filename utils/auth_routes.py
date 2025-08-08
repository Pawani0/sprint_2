from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.twilio_verify import send_verification_code, check_verification_code

router = APIRouter()

class PhoneRequest(BaseModel):
    phone: str

class VerifyRequest(BaseModel):
    phone: str
    code: str

@router.post("/auth/send-otp", tags=["Mobile Verification"])
async def send_otp(data: PhoneRequest):
    status = send_verification_code(data.phone)
    if status != "pending":
        raise HTTPException(status_code=400, detail="Failed to send OTP")
    return {"message": "OTP sent successfully"}

@router.post("/auth/verify-otp", tags=["Mobile Verification"])
async def verify_otp(data: VerifyRequest):
    if check_verification_code(data.phone, data.code):
        return {"verified": True, "message": "Phone verified"}
    raise HTTPException(status_code=400, detail="Invalid OTP")
