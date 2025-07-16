from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List
from dotenv import load_dotenv
import redis
import hashlib
import json
import os

load_dotenv()

_embeddings = None
_db = None
_redis = None

def get_redis():
    global _redis
    if _redis is None:
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        _redis = redis.Redis(host=redis_host, port=redis_port, db=0)
    return _redis

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def get_db():
    global _db
    if _db is None:
        _db = Chroma(persist_directory="./chroma_db", embedding_function=get_embeddings())
    return _db

def store_embeddings(chunks: List[str], source: str):
    try:
        embeddings = get_embeddings()
        metadatas = [{"source": source} for _ in chunks]
        db = Chroma.from_texts(chunks, embeddings, metadatas=metadatas, persist_directory="./chroma_db")
        get_redis().flushdb()
    except Exception as e:
        raise RuntimeError(f"Embedding/DB storage failed: {e}")

def get_top_k_chunks(query: str, k: int = 3):
    try:
        r = get_redis()
        cache_key = f"simsearch:{hashlib.sha256(query.encode()).hexdigest()}:{k}"
        cached = r.get(cache_key)
        if cached:
            return json.loads(cached)
        db = get_db()
        results = db.similarity_search(query, k=k)
        chunks = [doc.page_content for doc in results]
        r.set(cache_key, json.dumps(chunks), ex=600)
        return chunks
    except Exception as e:
        raise RuntimeError(f"Similarity search failed: {e}") 