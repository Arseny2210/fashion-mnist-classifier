import io
import os
import base64
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf

app = FastAPI(title="Классификатор одежды")

templates = Jinja2Templates(directory="templates")

CLASSES = [
    "Футболка",
    "Брюки",
    "Свитер",
    "Платье",
    "Пальто",
    "Сандалии",
    "Рубашка",
    "Кроссовки",
    "Сумка",
    "Ботильоны"
]

MODEL_PATH = "fashion_mnist_cnn.keras"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "fashion_mnist_dense.keras"

print(f"Загрузка модели из {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Модель успешно загружена!")

def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, str]:
    raw_img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(raw_img).convert("L")
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape
    
    # Превью исходного фото
    buf = io.BytesIO()
    raw_img.copy().convert("RGB").save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    # Оценка фона по краям изображения
    edges = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    bg = float(np.median(edges))
    
    diff = np.abs(arr - bg)
    mask = diff > 15.0
    if mask.sum() < 100:
        mask = diff > 8.0
        
    y_idx, x_idx = np.where(mask)
    if len(y_idx) > 0:
        ymin, ymax = y_idx.min(), y_idx.max()
        xmin, xmax = x_idx.min(), x_idx.max()
        pad_y = int((ymax - ymin) * 0.04) + 1
        pad_x = int((xmax - xmin) * 0.04) + 1
        ymin, ymax = max(0, ymin - pad_y), min(h, ymax + pad_y)
        xmin, xmax = max(0, xmin - pad_x), min(w, xmax + pad_x)
    else:
        ymin, ymax, xmin, xmax = 0, h, 0, w
        
    crop = arr[ymin:ymax, xmin:xmax]
    crop_mask = mask[ymin:ymax, xmin:xmax]
    
    obj_mean = float(crop[crop_mask].mean()) if crop_mask.sum() > 0 else float(crop.mean())
    corners = [crop[0, 0], crop[0, -1], crop[-1, 0], crop[-1, -1]]
    corner_val = float(np.median(corners))
    
    if obj_mean < bg:
        # Темный объект на светлом фоне (например, обувь на белом фоне)
        inv = 255.0 - crop
        inv_corners = 255.0 - corner_val
        fg = np.maximum(0.0, inv - inv_corners)
    else:
        # Светлый объект на темном/сером фоне (например, белый свитер на сером фоне)
        fg = np.maximum(0.0, crop - corner_val)
        
    if fg.max() > 0:
        fg = (fg / fg.max()) * 255.0
        
    cw, ch = fg.shape[1], fg.shape[0]
    # Адаптивное масштабирование под пропорции предмета в Fashion-MNIST
    target_size = 22.0 if (cw / ch > 1.25) else 24.0
    scale = target_size / max(cw, ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = Image.fromarray(fg.astype(np.uint8)).resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(resized, ((28 - nw) // 2, (28 - nh) // 2))

    tensor = (np.array(canvas, dtype=np.float32) / 255.0).reshape(1, 28, 28, 1)
    return tensor, img_b64

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "classes": CLASSES
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        content = await file.read()
        tensor, img_b64 = preprocess_image(content)
        preds = model.predict(tensor, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence_val = float(preds[class_idx])
        confidence_str = f"{confidence_val * 100:.1f}%"
        predicted_class = CLASSES[class_idx]
        
        # Топ-3 альтернативных вариантов
        sorted_indices = np.argsort(preds)[::-1]
        other_preds = []
        for idx in sorted_indices[1:4]:
            prob = float(preds[idx])
            if prob > 0.005:  # показывать только если вероятность больше 0.5%
                other_preds.append({
                    "name": CLASSES[idx],
                    "prob": f"{prob * 100:.1f}%",
                    "percent": round(prob * 100, 1)
                })
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "predicted_class": predicted_class,
                "confidence": confidence_str,
                "confidence_percent": round(confidence_val * 100, 1),
                "other_preds": other_preds,
                "image_base64": img_b64,
                "classes": CLASSES
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Не удалось распознать фото: {e}",
                "classes": CLASSES
            }
        )

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        tensor, img_b64 = preprocess_image(content)
        preds = model.predict(tensor, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence_val = float(preds[class_idx])
        confidence_str = f"{confidence_val * 100:.1f}%"
        predicted_class = CLASSES[class_idx]
        
        sorted_indices = np.argsort(preds)[::-1]
        other_preds = []
        for idx in sorted_indices[1:4]:
            prob = float(preds[idx])
            if prob > 0.005:
                other_preds.append({
                    "name": CLASSES[idx],
                    "prob": f"{prob * 100:.1f}%",
                    "percent": round(prob * 100, 1)
                })
        
        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence_str,
            "confidence_percent": round(confidence_val * 100, 1),
            "other_preds": other_preds,
            "image_base64": img_b64
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Не удалось распознать фото: {e}"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
