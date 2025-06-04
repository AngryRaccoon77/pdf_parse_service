import os
import hashlib
import sqlite3
import asyncio
from pathlib import Path
from typing import Optional, List
import httpx
import tempfile

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from unstructured.partition.pdf import partition_pdf


# Конфигурация
class Settings(BaseSettings):
    watch_folder: str = "./data"
    sqlite_db: str = "files.db"
    embedding_service_url: str = "http://localhost:8085/embed_passage"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "context_embeddings"
    check_interval: int = 60  # seconds

    class Config:
        env_file = ".env"


settings = Settings()

# Инициализация приложения
app = FastAPI()
observer = Observer()
db_conn = sqlite3.connect(settings.sqlite_db, check_same_thread=False)


# Модели данных
class FileMetadata(BaseModel):
    file_path: str
    file_hash: str
    processed: bool = False


class TextInput(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: List[float]


# Инициализация БД
def init_db():
    cursor = db_conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            file_hash TEXT,
            processed BOOLEAN,
            processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    db_conn.commit()


init_db()


# Вспомогательные функции PDF-обработки
async def process_pdf(file_path: str) -> str:
    try:
        elements = partition_pdf(filename=file_path, strategy="hi_res")
        return "\n".join([element.text for element in elements if hasattr(element, 'text')])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


async def get_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# Эндпоинты PDF-обработки
@app.post("/process-pdf/")
async def handle_pdf_processing(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            contents = await file.read()
            temp_pdf.write(contents)
            temp_path = temp_pdf.name

        text = await process_pdf(temp_path)
        os.remove(temp_path)
        return {"text": text}

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


# Остальные вспомогательные функции
async def get_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.embedding_service_url}",
            json={"text": text}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Embedding failed")
        return response.json()["embedding"]


async def upsert_to_qdrant(file_id: int, embedding: List[float], text: str):
    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points",
            json={
                "points": [{
                    "id": file_id,
                    "vector": embedding,
                    "payload": {
                        "text": text,
                        "source": file_id
                    }
                }]
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Qdrant upsert failed")


# Обработчик файловой системы
class FileHandler(FileSystemEventHandler):
    def __init__(self):
        self.lock = asyncio.Lock()

    async def process_file(self, file_path: str):
        async with self.lock:
            file_path = str(file_path)
            if not file_path.endswith(".pdf") or not os.path.isfile(file_path):
                return

            current_hash = await get_file_hash(file_path)
            cursor = db_conn.cursor()

            cursor.execute('''
                SELECT file_hash, processed FROM files 
                WHERE file_path = ?
            ''', (file_path,))
            result = cursor.fetchone()

            if result and result[0] == current_hash and result[1]:
                return

            try:
                text = await process_pdf(file_path)
                embedding = await get_embedding(text)

                cursor.execute('''
                    INSERT OR REPLACE INTO files 
                    (file_path, file_hash, processed)
                    VALUES (?, ?, ?)
                ''', (file_path, current_hash, True))
                file_id = cursor.lastrowid
                db_conn.commit()

                await upsert_to_qdrant(file_id, embedding, text)

            except Exception as e:
                cursor.execute('''
                    UPDATE files SET processed = ?
                    WHERE file_path = ?
                ''', (False, file_path))
                db_conn.commit()
                print(f"Error processing file {file_path}: {str(e)}")

    def on_created(self, event):
        if not event.is_directory:
            asyncio.run(self.process_file(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            asyncio.run(self.process_file(event.src_path))


# Фоновые задачи
@app.on_event("startup")
async def startup_event():
    observer.schedule(FileHandler(), settings.watch_folder, recursive=True)
    observer.start()


@app.on_event("shutdown")
def shutdown_event():
    observer.stop()
    observer.join()
    db_conn.close()


# Эндпоинты управления
@app.get("/status")
async def get_status():
    cursor = db_conn.cursor()
    cursor.execute('''
        SELECT COUNT(*) as total, 
               SUM(processed) as processed 
        FROM files
    ''')
    result = cursor.fetchone()
    return {
        "total_files": result[0],
        "processed_files": result[1],
        "watch_folder": settings.watch_folder
    }


@app.post("/sync")
async def force_sync():
    for file_path in Path(settings.watch_folder).rglob("*"):
        if file_path.is_file():
            asyncio.run(FileHandler().process_file(str(file_path)))
    return {"status": "sync started"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8090)