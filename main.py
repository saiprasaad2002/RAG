from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel
import uvicorn
from services import process_upload, async_process_chat

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post('/upload')
async def uploadFile(file: UploadFile = File(...)):
    try:
        result = await process_upload(file)
        return result
    except Exception as e:
        print(f"Upload error: {repr(e)}")
        print(f"Upload error type: {type(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat')
async def chat(request: Request, chat_request: ChatRequest):
    try:
        user_id = get_client_ip(request)
        print(f"[Chat] user_id={user_id}")
        result = await async_process_chat(chat_request.question, user_id=user_id)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Chat error: {repr(e)}")
        print(f"Chat error type: {type(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def get_client_ip(request: Request):
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        ip = x_forwarded_for.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "anonymous"
    return ip


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
