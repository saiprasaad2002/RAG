import os
import tempfile
from fastapi import UploadFile, HTTPException
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document
from vectordb import store_embeddings, get_top_k_chunks, get_redis
import google.generativeai as genai
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from dotenv import load_dotenv
import requests

load_dotenv()

_executor = ThreadPoolExecutor(max_workers=4)

def extract_text_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")

def extract_text_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        raise RuntimeError(f"DOCX extraction failed: {e}")

def extract_text_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"TXT extraction failed: {e}")

def chunk_text(text: str) -> List[str]:
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)
    except Exception as e:
        raise RuntimeError(f"Text splitting failed: {e}")

async def process_upload(file: UploadFile):
    filename = file.filename
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            text = extract_text_pdf(tmp_path)
        elif suffix == ".docx":
            text = extract_text_docx(tmp_path)
        elif suffix == ".txt":
            text = extract_text_txt(tmp_path)
        else:
            os.remove(tmp_path)
            raise HTTPException(status_code=400, detail="Unsupported file type.")

    except HTTPException as e:
        os.remove(tmp_path)
        raise e
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))
    try:
        chunks = chunk_text(text)
    except HTTPException as e:
        os.remove(tmp_path)
        raise e
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))
    try:
        store_embeddings(chunks, filename)

    except HTTPException as e:
        os.remove(tmp_path)
        raise e
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))
    os.remove(tmp_path)
    return {"message": f"{filename} processed and stored successfully"}

async def async_get_top_k_chunks(query: str, k: int = 3):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, get_top_k_chunks, query, k)

async def async_generate_llm_response(context_chunks: list, question: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, generate_llm_response, context_chunks, question)

def call_groq_llama(prompt: str, max_tokens: int = 256) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not set.")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

async def async_classify_intent_and_get_chunks(question: str, k: int = 5):
    try:
        r = get_redis()
        cache_key = f"intent_chunks:{hashlib.sha256(question.encode()).hexdigest()}:{k}"
        cached = r.get(cache_key)
        if cached:
            data = json.loads(cached)
            return data["is_relevant"], data["chunks"]
        chunks = await async_get_top_k_chunks(question, k=k)
        if not chunks:
            r.set(cache_key, json.dumps({"is_relevant": False, "chunks": []}), ex=3600)
            return False, []
        context_text = "\n".join(chunks)
        prompt = f"""You are an expert intent classifier for document Q&A systems. Your task is to determine if a question is relevant to the provided document content.

DOCUMENT CONTENT:
{context_text}

QUESTION: \"{question}\"

CLASSIFICATION RULES:
- Answer \"yes\" if the question asks about ANY information that could be found in the document content
- Answer \"yes\" if the question seeks facts, details, explanations, or insights from the document
- Answer \"yes\" if the question is about topics, concepts, or subjects mentioned in the document
- Answer \"no\" ONLY if the question is completely unrelated to the document's subject matter
- Be generous in classification - if there's any potential relevance, classify as \"yes\"

Examples of relevant questions:
- Questions about specific facts in the document
- Questions about concepts explained in the document  
- Questions about people, places, or things mentioned
- Questions about processes, procedures, or methods described
- Questions about data, statistics, or findings presented

Examples of irrelevant questions:
- Questions about completely different topics not mentioned
- Personal questions unrelated to the document
- Questions about external events not referenced

Based on the document content above, is the question relevant? Respond with only \"yes\" or \"no\":

Response:"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise Exception("GEMINI_API_KEY not set.")
            loop = asyncio.get_event_loop()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemma-3n-e4b-it")
            def call_gemini():
                response = model.generate_content(prompt)
                return response.text.strip().lower()
            result = await loop.run_in_executor(_executor, call_gemini)
            is_relevant = any(positive in result for positive in ["yes", "relevant", "true", "1"])
        except Exception as llm_error:
            try:
                def call_groq():
                    return call_groq_llama(prompt, max_tokens=8).lower()
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(_executor, call_groq)
                is_relevant = any(positive in result for positive in ["yes", "relevant", "true", "1"])
            except Exception as groq_e:
                print(f"Both LLMs failed for intent classification: {llm_error}, {groq_e}")
                is_relevant = True
        r.set(cache_key, json.dumps({"is_relevant": is_relevant, "chunks": chunks}), ex=3600)
        return is_relevant, chunks
    except HTTPException as e:
        raise e
    except Exception as e:
        try:
            chunks = await async_get_top_k_chunks(question, k=k)
            return len(chunks) > 0, chunks
        except:
            return False, []

def check_llm_rate_limit(user_id: str, max_calls: int = 5, window_seconds: int = 60):
    r = get_redis()
    key = f"llm_rate:{user_id}"
    current = r.get(key)
    print(f"[RateLimit] user_id={user_id}, current={current}")
    if current is not None and int(current) >= max_calls:
        print(f"[RateLimit] BLOCKED user_id={user_id}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded: Max 5 LLM calls per minute.")
    pipe = r.pipeline()
    pipe.incr(key, 1)
    pipe.expire(key, window_seconds)
    pipe.execute()

async def async_process_chat(question: str, user_id: str = "anonymous"):
    try:
        check_llm_rate_limit(user_id)
        is_relevant, chunks = await async_classify_intent_and_get_chunks(question, k=5)
        print(chunks)
        if not is_relevant:
            return {"message": "Question not relevant to document content extraction."}
        if not chunks:
            return {"message": "No relevant content found in documents."}
        answer = await async_generate_llm_response(chunks, question)
        return {"answer": answer, "context": chunks}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_llm_response(context_chunks: list, question: str) -> str:
    try:
        r = get_redis()
        cache_key = f"llm:{hashlib.sha256((' '.join(context_chunks) + question).encode()).hexdigest()}"
        cached = r.get(cache_key)
        if cached:
            return cached.decode()
        context = "\n".join(context_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise Exception("GEMINI_API_KEY not set.")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemma-3n-e4b-it")
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            try:
                answer = call_groq_llama(prompt)
            except Exception as groq_e:
                return {"message": f"Failed to generate response from both models: {str(e)}, {str(groq_e)}"}
        r.set(cache_key, answer, ex=600)
        return answer
    except HTTPException as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"LLM response generation failed: {e}") 