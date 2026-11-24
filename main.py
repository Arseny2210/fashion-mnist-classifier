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
    
    # Автоопределение фона и автокроп границ объекта
    edges = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    bg = float(np.median(edges))
    
    if bg < 50:
        mask = arr > 25
        if np.any(mask):
            y_idx, x_idx = np.where(mask)
            cropped = img.crop((x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()))
        else:
            cropped = img
    else:
        diff = np.abs(arr - bg)
        mask = diff > 15
        if np.any(mask):
            y_idx, x_idx = np.where(mask)
            ymin, ymax = y_idx.min(), y_idx.max()
            xmin, xmax = x_idx.min(), x_idx.max()
            pad_y = int((ymax - ymin) * 0.04) + 1
            pad_x = int((xmax - xmin) * 0.04) + 1
            ymin, ymax = max(0, ymin - pad_y), min(h, ymax + pad_y)
            xmin, xmax = max(0, xmin - pad_x), min(w, xmax + pad_x)
            cropped = img.crop((xmin, ymin, xmax, ymax))
        else:
            cropped = img
            
        inv = ImageOps.invert(cropped)
        inv_arr = np.array(inv, dtype=np.float32)
        corner_val = np.median([inv_arr[0,0], inv_arr[0,-1], inv_arr[-1,0], inv_arr[-1,-1]])
        inv_arr = np.maximum(0.0, inv_arr - corner_val)
        if inv_arr.max() > 0:
            inv_arr = (inv_arr / inv_arr.max()) * 255.0
        cropped = Image.fromarray(inv_arr.astype(np.uint8))
        
    cw, ch = cropped.size
    scale = 22.0 / max(cw, ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
    
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
