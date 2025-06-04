import os
import hashlib
import sqlite3
import asyncio
from pathlib import Path
from typing import Optional, List, Any  # Any можно заменить на asyncio.AbstractEventLoop
import httpx  # Для embedding_service_url
import tempfile
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.concurrency import run_in_threadpool  # Added for new PDF processing
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from unstructured.partition.pdf import partition_pdf
import logging  # Added for new PDF processing

# Импорты для qdrant-client
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# from qdrant_client.http.exceptions import UnexpectedResponse # Может понадобиться для детальной обработки ошибок

# Инициализация логгера
logger = logging.getLogger(__name__)


# Конфигурация
class Settings(BaseSettings):
    watch_folder: str = "./data"
    sqlite_db: str = "files.db"
    embedding_service_url: str = "http://localhost:8085/embed_passage"
    qdrant_url: str = "http://localhost:6333"  # URL для Qdrant
    qdrant_collection: str = "context_embeddings"
    qdrant_vector_size: int = 2048  # УТОЧНИТЕ ЭТО ЗНАЧЕНИЕ для вашей модели эмбеддингов
    qdrant_distance_metric: str = "Cosine"  # COSINE, EUCLID, DOT
    # qdrant_api_key: Optional[str] = None # Если Qdrant требует API ключ
    check_interval: int = 60  # seconds

    class Config:
        env_file = ".env"


settings = Settings()
EMBEDDING_MODEL_NAME = "ai-sage/Giga-Embeddings-instruct"
MODEL_CACHE_DIR = "cache_dir"
tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL_NAME,
    cache_dir=MODEL_CACHE_DIR,
)

# Инициализация приложения
app = FastAPI()
observer = Observer()
db_conn = sqlite3.connect(settings.sqlite_db, check_same_thread=False)
file_handler_instance: Optional['FileHandler'] = None


async def ensure_qdrant_collection_exists(client: AsyncQdrantClient):
    """
    Проверяет существование коллекции в Qdrant и создает ее, если она не существует,
    используя предоставленный qdrant-client.
    """
    try:
        await client.get_collection(collection_name=settings.qdrant_collection)
        print(f"Коллекция '{settings.qdrant_collection}' уже существует в Qdrant.")
    except Exception as e:
        print(
            f"Коллекция '{settings.qdrant_collection}' не найдена или ошибка при проверке ({type(e).__name__}: {e}). Попытка создания...")
        try:
            await client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=settings.qdrant_vector_size,
                    distance=Distance[settings.qdrant_distance_metric.upper()]
                )
            )
            print(f"Коллекция '{settings.qdrant_collection}' успешно создана.")
        except Exception as creation_error:
            print(
                f"Ошибка при создании коллекции '{settings.qdrant_collection}': {type(creation_error).__name__} - {creation_error}")
            # raise HTTPException(status_code=500, detail=f"Не удалось создать коллекцию Qdrant: {creation_error}") from creation_error


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
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            file_hash TEXT,
            processed BOOLEAN,
            processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            content TEXT,
            chunk_index INTEGER,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
    ''')
    db_conn.commit()


init_db()


def split_text_into_chunks(text: str, max_tokens: int = 4096, overlap_tokens: int = 256):
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    chunks = []
    start = 0
    chunk_index = 0
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append((chunk_index, chunk_text))
        chunk_index += 1
        start += (max_tokens - overlap_tokens)
    return chunks


# Вспомогательные функции PDF-обработки (обновлено)
async def process_pdf(file_path: str) -> str:
    try:
        elements = await run_in_threadpool(
            partition_pdf,
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            languages=["rus"]
            # include_page_breaks=False, # Можно раскомментировать, если нужны маркеры разрыва страниц
        )

        document_parts_in_order = []
        for element in elements:
            element_text_content = getattr(element, 'text', '')

            if hasattr(element, "category") and element.category == "Table":
                html_table = getattr(element.metadata, 'text_as_html', None)
                if html_table:
                    document_parts_in_order.append(html_table)
                elif element_text_content and element_text_content.strip():
                    # Fallback to text if HTML is not available but text is
                    document_parts_in_order.append(element_text_content)
            elif element_text_content and element_text_content.strip():
                document_parts_in_order.append(element_text_content)

        combined_text_with_tables = "\n\n".join(document_parts_in_order)
        return combined_text_with_tables
    except Exception as e:
        logger.error(f"PDF processing failed for {file_path}: {str(e)}", exc_info=True)
        raise  # Перевыброс исключения, чтобы оно было поймано вызывающей функцией


async def get_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


async def process_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"TXT processing failed for {file_path}: {str(e)}")
        raise


async def process_md(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"MD processing failed for {file_path}: {str(e)}")
        raise


# Эндпоинты файловой обработки
@app.post("/process-file/")
async def handle_file_processing(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    filename = file.filename.lower()
    if not filename.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, and MD files are allowed"
        )

    temp_path = ""
    try:
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        if filename.endswith(".pdf"):
            processed_content = await process_pdf(temp_path)
            return {"extracted_text": processed_content}
        elif filename.endswith(".txt"):
            text_content = await process_txt(temp_path)
            return {"text": text_content}
        elif filename.endswith(".md"):
            text_content = await process_md(temp_path)
            return {"text": text_content}
        else:
            # This case should ideally not be reached due to the check above
            raise HTTPException(status_code=400, detail="Unsupported file type after initial check.")

    except Exception as e:
        logger.error(f"Error in handle_file_processing for {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e_remove:
                logger.error(f"Не удалось удалить временный файл {temp_path}: {str(e_remove)}")


# Вспомогательные функции для эмбеддингов и Qdrant
async def get_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.embedding_service_url}",
            json={"text": text}
        )
        if response.status_code != 200:
            print(f"Embedding failed with status {response.status_code}: {response.text}")
            raise Exception("Embedding failed")
        return response.json()["embedding"]


async def upsert_to_qdrant(client: AsyncQdrantClient, point_id: int, embedding: List[float], text: str,
                           document_id: int):
    """
    Добавляет или обновляет точку в Qdrant, используя предоставленный qdrant-client.
    """
    try:
        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        "document_id": document_id
                    }
                )
            ]
        )
    except Exception as e:
        print(f"Ошибка при добавлении/обновлении точки {point_id} в Qdrant: {str(e)}")
        raise Exception(f"Ошибка Qdrant upsert: {str(e)}") from e


# Обработчик файловой системы
class FileHandler(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop, qdrant_client: AsyncQdrantClient):
        self.loop = loop
        self.qdrant_client = qdrant_client
        self.lock = asyncio.Lock()

    async def process_file(self, file_path: str):
        file_path_str = str(file_path)
        document_id: Optional[int] = None
        cursor = db_conn.cursor()  # Define cursor at the beginning of try block potentially
        try:
            async with self.lock:
                if not file_path_str.endswith((".pdf", ".txt", ".md")):
                    return

                current_hash = await get_file_hash(file_path_str)

                cursor.execute('''
                    SELECT id, file_hash, processed FROM documents 
                    WHERE file_path = ?
                ''', (file_path_str,))
                result = cursor.fetchone()

                if result and result[1] == current_hash and result[2]:
                    print(f"Файл {file_path_str} не изменился и уже обработан.")
                    return

                if not result:
                    cursor.execute('''
                        INSERT INTO documents (file_path, file_hash, processed)
                        VALUES (?, ?, ?)
                    ''', (file_path_str, current_hash, False))
                    document_id = cursor.lastrowid
                    db_conn.commit()
                    print(f"Новый файл {file_path_str} добавлен в БД с ID: {document_id}.")
                else:
                    document_id = result[0]
                    print(f"Файл {file_path_str} (ID: {document_id}) будет переобработан.")
                    cursor.execute('''
                        UPDATE documents SET file_hash = ?, processed = ?
                        WHERE id = ?
                    ''', (current_hash, False, document_id))
                    db_conn.commit()

                if document_id is None:  # Should not happen if logic above is correct
                    print(f"Ошибка: document_id не был установлен для {file_path_str}")
                    return

                cursor.execute('DELETE FROM chunks WHERE document_id = ?', (document_id,))
                db_conn.commit()
                print(f"Старые чанки для документа ID {document_id} удалены.")

                text_content = ""
                if file_path_str.endswith(".pdf"):
                    text_content = await process_pdf(file_path_str)
                elif file_path_str.endswith(".txt"):
                    text_content = await process_txt(file_path_str)
                elif file_path_str.endswith(".md"):
                    text_content = await process_md(file_path_str)

                chunks = split_text_into_chunks(text_content)

                for chunk_index, chunk_text in chunks:
                    embedding = await get_embedding(chunk_text)
                    cursor.execute('''
                        INSERT INTO chunks (document_id, content, chunk_index)
                        VALUES (?, ?, ?)
                    ''', (document_id, chunk_text, chunk_index))
                    chunk_id = cursor.lastrowid
                    db_conn.commit()
                    if chunk_id is None:  # Safety check
                        print(f"Ошибка: chunk_id не был получен для чанка документа {document_id}")
                        continue  # Or raise an error
                    await upsert_to_qdrant(self.qdrant_client, chunk_id, embedding, chunk_text,
                                           document_id)

                cursor.execute('''
                    UPDATE documents SET processed = ?, processed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (True, document_id))
                db_conn.commit()
                print(f"Файл {file_path_str} (ID: {document_id}) успешно обработан и добавлен.")

        except Exception as e:
            logger.error(f"Error processing file {file_path_str} in process_file: {str(e)}", exc_info=True)
            if document_id is not None:
                try:
                    # Ensure cursor is available; it might not be if error occurred before its initialization in try
                    # Re-obtaining cursor if necessary, or ensuring it's defined in a broader scope.
                    # For simplicity, assuming cursor is available or re-obtained if needed.
                    if db_conn:  # Check if db_conn is still valid
                        error_cursor = db_conn.cursor()
                        error_cursor.execute('''
                            UPDATE documents SET processed = ?
                            WHERE id = ?
                        ''', (False, document_id))
                        db_conn.commit()
                except Exception as db_err:
                    logger.error(f"Failed to update document status after error for {file_path_str}: {db_err}",
                                 exc_info=True)

    def on_created(self, event):
        if not event.is_directory:
            print(f"File created: {event.src_path}")
            asyncio.run_coroutine_threadsafe(self.process_file(event.src_path), self.loop)

    def on_modified(self, event):
        if not event.is_directory:
            print(f"File modified: {event.src_path}")
            asyncio.run_coroutine_threadsafe(self.process_file(event.src_path), self.loop)


# Фоновые задачи и жизненный цикл FastAPI
@app.on_event("startup")
async def startup_event():
    app.state.qdrant_client = AsyncQdrantClient(
        url=settings.qdrant_url,
        # api_key=settings.qdrant_api_key,
        # prefer_grpc=True,
    )

    await ensure_qdrant_collection_exists(app.state.qdrant_client)

    global file_handler_instance
    loop = asyncio.get_running_loop()
    file_handler_instance = FileHandler(loop=loop, qdrant_client=app.state.qdrant_client)
    observer.schedule(file_handler_instance, settings.watch_folder, recursive=True)
    observer.start()
    print(f"Наблюдатель запущен для папки: {settings.watch_folder}")


@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'qdrant_client') and app.state.qdrant_client:
        await app.state.qdrant_client.close()
        print("Соединение с Qdrant закрыто.")

    observer.stop()
    observer.join()
    db_conn.close()
    print("Наблюдатель остановлен, соединение с БД закрыто.")


# Эндпоинты управления
@app.get("/status")
async def get_status():
    cursor = db_conn.cursor()
    cursor.execute('''
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN processed = 1 THEN 1 ELSE 0 END) as processed 
        FROM documents
    ''')
    doc_result = cursor.fetchone()
    cursor.execute('''
        SELECT COUNT(*) FROM chunks
    ''')
    chunk_count_result = cursor.fetchone()
    chunk_count = chunk_count_result[0] if chunk_count_result else 0

    return {
        "total_documents": doc_result[0] if doc_result else 0,
        "processed_documents": doc_result[1] if doc_result and doc_result[1] is not None else 0,
        "total_chunks": chunk_count,
        "watch_folder": settings.watch_folder
    }


@app.post("/sync")
async def force_sync():
    if file_handler_instance is None:
        raise HTTPException(status_code=503, detail="File handler not initialized. Server might be starting up.")

    print("Принудительная синхронизация запущена...")
    tasks = []
    # Ensure watch_folder exists before attempting to scan
    watch_folder_path = Path(settings.watch_folder)
    if not watch_folder_path.is_dir():
        logger.warning(f"Watch folder {settings.watch_folder} does not exist. Cannot perform sync.")
        return {"status": "Sync triggered, but watch folder not found."}

    for file_path_obj in watch_folder_path.rglob("*"):
        file_path_str = str(file_path_obj)
        if file_path_obj.is_file() and file_path_str.endswith((".pdf", ".txt", ".md")):
            print(f"Найдено для синхронизации: {file_path_str}")
            # Using asyncio.create_task for futures that can be awaited if needed,
            # but here we're just firing them off.
            # For tasks that should run in the background and are managed by the event loop directly:
            asyncio.ensure_future(file_handler_instance.process_file(file_path_str))
            tasks.append(file_path_str)  # Keep track of what was scheduled

    if not tasks:
        return {"status": "Sync triggered, no files found to process."}

    return {"status": f"Sync triggered for {len(tasks)} files. Processing in background."}


if __name__ == "__main__":
    import uvicorn

    Path(settings.watch_folder).mkdir(parents=True, exist_ok=True)
    print(f"Папка для данных: {Path(settings.watch_folder).resolve()}")

    uvicorn.run(app, host="0.0.0.0", port=8090)