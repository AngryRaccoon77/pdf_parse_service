from fastapi import FastAPI, UploadFile, File, HTTPException
from unstructured.partition.pdf import partition_pdf
import tempfile
import os

app = FastAPI()


@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    # Проверка формата файла
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF файлы разрешены")

    try:
        # Сохранение файла во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            contents = await file.read()
            temp_pdf.write(contents)
            temp_pdf_path = temp_pdf.name

        # Обработка PDF
        elements = partition_pdf(filename=temp_pdf_path, strategy="hi_res")

        # Извлечение текста
        text = "\n".join([element.text for element in elements if hasattr(element, 'text')])

        # Удаление временного файла
        os.remove(temp_pdf_path)

        # Возврат результата
        return {"text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки PDF: {str(e)}")


@app.get("/health", summary="Health check endpoint")
def health_check():
    return {"status": "ok"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")