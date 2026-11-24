import io
import os
import base64
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf

app = FastAPI(title="Fashion-MNIST Classifier")

# Подключение папки шаблонов
templates = Jinja2Templates(directory="templates")

# Названия 10 классов одежды Fashion-MNIST
CLASSES = [
    "Футболка / Топ (T-shirt/top)",
    "Брюки (Trouser)",
    "Пуловер / Свитер (Pullover)",
    "Платье (Dress)",
    "Пальто (Coat)",
    "Сандалии (Sandal)",
    "Рубашка (Shirt)",
    "Кроссовки (Sneaker)",
    "Сумка (Bag)",
    "Ботильоны (Ankle boot)"
]

ICONS = ["👕", "👖", "🧥", "👗", "🧥", "👡", "👔", "👟", "👜", "👢"]

# Загрузка обученной модели
MODEL_PATH = "fashion_mnist_cnn.keras"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "fashion_mnist_dense.keras"

print(f"Загрузка модели из {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Модель успешно загружена!")

def smart_preprocess(image_bytes: bytes) -> tuple[np.ndarray, str, str]:
    """
    Интеллектуальная предобработка реальных фото одежды под стандарт Fashion-MNIST:
    1. Автоповорот EXIF
    2. Перевод в градации серого
    3. Автоматическое определение фона по краям изображения
    4. Автокроп (выделение ограничивающей рамки объекта без пустых полей)
    5. Инверсия фона (если фон светлый, делаем его черным, а одежду - яркой)
    6. Пропорциональное масштабирование в область 22x22 и центрирование на холсте 28x28
    """
    raw_img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(raw_img).convert("L")
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape
    
    # 1. Превью исходного фото
    buf_orig = io.BytesIO()
    raw_img.copy().convert("RGB").save(buf_orig, format="JPEG", quality=85)
    orig_b64 = base64.b64encode(buf_orig.getvalue()).decode("utf-8")
    
    # 2. Определение цвета фона по рамке изображения
    edges = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    bg = float(np.median(edges))
    
    # 3. Сегментация объекта и автокроп
    if bg < 50:
        # Фон уже темный
        mask = arr > 25
        if np.any(mask):
            y_idx, x_idx = np.where(mask)
            cropped = img.crop((x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()))
        else:
            cropped = img
    else:
        # Светлый фон (стандартное студийное фото)
        diff = np.abs(arr - bg)
        mask = diff > 15
        if np.any(mask):
            y_idx, x_idx = np.where(mask)
            ymin, ymax = y_idx.min(), y_idx.max()
            xmin, xmax = x_idx.min(), x_idx.max()
            # Добавляем 4% отступа
            pad_y = int((ymax - ymin) * 0.04) + 1
            pad_x = int((xmax - xmin) * 0.04) + 1
            ymin, ymax = max(0, ymin - pad_y), min(h, ymax + pad_y)
            xmin, xmax = max(0, xmin - pad_x), min(w, xmax + pad_x)
            cropped = img.crop((xmin, ymin, xmax, ymax))
        else:
            cropped = img
            
        # Инвертируем: делаем фон черным (0), а объект ярким (>0)
        inv = ImageOps.invert(cropped)
        inv_arr = np.array(inv, dtype=np.float32)
        corner_val = np.median([inv_arr[0,0], inv_arr[0,-1], inv_arr[-1,0], inv_arr[-1,-1]])
        inv_arr = np.maximum(0.0, inv_arr - corner_val)
        if inv_arr.max() > 0:
            inv_arr = (inv_arr / inv_arr.max()) * 255.0
        cropped = Image.fromarray(inv_arr.astype(np.uint8))
        
    # 4. Пропорциональный ресайз до max 22px
    cw, ch = cropped.size
    scale = 22.0 / max(cw, ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
    
    # 5. Центрирование на черном холсте 28x28
    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(resized, ((28 - nw) // 2, (28 - nh) // 2))
    
    # Превью обработанного тензора 28x28
    buf_prep = io.BytesIO()
    canvas.resize((140, 140), Image.Resampling.NEAREST).save(buf_prep, format="PNG")
    prep_b64 = base64.b64encode(buf_prep.getvalue()).decode("utf-8")
    
    # Нормализация [0, 1] и добавление осей
    tensor = (np.array(canvas, dtype=np.float32) / 255.0).reshape(1, 28, 28, 1)
    return tensor, orig_b64, prep_b64

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Главная страница с формой загрузки."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """Обработка загруженного изображения и вывод предсказания."""
    try:
        content = await file.read()
        tensor, orig_b64, prep_b64 = smart_preprocess(content)
        
        # Получение предсказания
        preds = model.predict(tensor, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = f"{preds[class_idx]:.2%}"
        predicted_class = f"{ICONS[class_idx]} {CLASSES[class_idx]}"
        
        # Топ-3 классов
        top3_idx = np.argsort(preds)[::-1][:3]
        top3 = [
            {
                "name": f"{ICONS[i]} {CLASSES[i]}",
                "prob": f"{preds[i]:.2%}",
                "percent": round(float(preds[i]) * 100, 1)
            }
            for i in top3_idx
        ]
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "orig_b64": orig_b64,
                "prep_b64": prep_b64,
                "top3": top3
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Ошибка при обработке файла: {e}"
            }
        )

# JSON API эндпоинт
@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    """API-эндпоинт: возвращает JSON с предсказанным классом и вероятностями."""
    content = await file.read()
    tensor, _, _ = smart_preprocess(content)
    preds = model.predict(tensor, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    return {
        "class_id": class_idx,
        "class_name": CLASSES[class_idx],
        "confidence": float(preds[class_idx]),
        "all_probabilities": {CLASSES[i]: float(preds[i]) for i in range(10)}
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
