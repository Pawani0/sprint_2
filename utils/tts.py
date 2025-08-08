import edge_tts

async def stream_tts(text: str, voice: str = "en-IN-NeerjaNeural"):
    communicate = edge_tts.Communicate(text, voice=voice)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]
