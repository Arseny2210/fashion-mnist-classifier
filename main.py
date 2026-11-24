import io
import os
import json
import base64
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Импорт ультралегковесного движка инференса (55 МБ RAM вместо 590 МБ)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

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

# Инициализация моделей TFLite
FMNIST_PATH = "fashion_mnist_cnn.tflite"
MOBILENET_PATH = "mobilenet_v2_quant.tflite"

print(f"Загрузка легковесной модели Fashion-MNIST из {FMNIST_PATH}...")
interp_fmnist = Interpreter(FMNIST_PATH)
interp_fmnist.allocate_tensors()
in_fmnist = interp_fmnist.get_input_details()[0]["index"]
out_fmnist = interp_fmnist.get_output_details()[0]["index"]

print(f"Загрузка легковесной модели MobileNetV2 из {MOBILENET_PATH}...")
interp_mob = Interpreter(MOBILENET_PATH)
interp_mob.allocate_tensors()
in_mob = interp_mob.get_input_details()[0]["index"]
out_mob = interp_mob.get_output_details()[0]["index"]

print("Модели успешно загружены и готовы к инференсу!")

# Загрузка словаря классов ImageNet
IMAGENET_INDEX_PATH = "static/imagenet_class_index.json"
if os.path.exists(IMAGENET_INDEX_PATH):
    with open(IMAGENET_INDEX_PATH, "r", encoding="utf-8") as f:
        idx2label = json.load(f)
else:
    import urllib.request
    with urllib.request.urlopen("https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json") as resp:
        idx2label = json.loads(resp.read().decode("utf-8"))

CLOTHING_KEYWORDS = {
    "jersey", "t-shirt", "tee_shirt", "sweatshirt", "sweater", "cardigan", "wool",
    "jean", "pajama", "trousers", "pants",
    "gown", "hoopskirt", "overskirt", "dress", "sarong", "apron", "kimono",
    "trench_coat", "coat", "fur_coat", "cloak", "overcoat", "suit",
    "sandal", "clog", "shoe", "running_shoe", "boot", "loafer", "stiletto", "cleat", "velvet",
    "mailbag", "purse", "handbag", "backpack", "wallet",
    "tie", "windsor_tie", "bow_tie", "shirt", "maillot"
}

def decode_mob_preds(preds, top=15):
    top_indices = np.argsort(preds[0])[::-1][:top]
    results = []
    for idx in top_indices:
        class_id, class_name = idx2label[str(idx)]
        results.append((class_id, class_name, float(preds[0][idx])))
    return results

def preprocess_mob(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil.resize((224, 224)), dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

def create_fmnist_tensor(arr_rgb: np.ndarray) -> np.ndarray:
    """Формирует нормализованный 28x28 тензор в стиле Fashion-MNIST."""
    h, w, _ = arr_rgb.shape
    borders = np.concatenate([arr_rgb[0, :, :], arr_rgb[-1, :, :], arr_rgb[:, 0, :], arr_rgb[:, -1, :]], axis=0)
    bg_rgb = np.median(borders, axis=0)
    
    diff = np.linalg.norm(arr_rgb - bg_rgb, axis=-1)
    mask = diff > 15.0
    if mask.sum() < 80:
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
        
    crop_diff = diff[ymin:ymax, xmin:xmax]
    fg = (crop_diff / crop_diff.max()) * 255.0 if crop_diff.max() > 0 else np.zeros_like(crop_diff)
        
    cw, ch = fg.shape[1], fg.shape[0]
    target_size = 22.0 if (cw / (ch + 1e-5) > 1.25) else 24.0
    scale = target_size / max(cw, ch, 1)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = Image.fromarray(fg.astype(np.uint8)).resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(resized, ((28 - nw) // 2, (28 - nh) // 2))

    tensor = (np.array(canvas, dtype=np.float32) / 255.0).reshape(1, 28, 28, 1)
    return tensor

def classify_uploaded_image(image_bytes: bytes):
    """
    Интеллектуальный классификатор с проверкой на наличие одежды.
    Возвращает: (результаты_словарь, ошибка)
    """
    raw_img = Image.open(io.BytesIO(image_bytes))
    raw_img = ImageOps.exif_transpose(raw_img).convert("RGB")
    arr = np.array(raw_img)
    
    # Превью исходного фото в формате base64
    buf = io.BytesIO()
    raw_img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    # Оценка фона по краям изображения
    borders = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    bg_std = float(np.std(borders, axis=0).mean())
    bg_rgb = np.median(borders, axis=0)
    
    # Предикт MobileNetV2
    x_mob = preprocess_mob(raw_img)
    interp_mob.set_tensor(in_mob, x_mob)
    interp_mob.invoke()
    m_preds = interp_mob.get_tensor(out_mob)
    top15 = decode_mob_preds(m_preds, top=15)
    top_dict = {p[1].lower(): float(p[2]) for p in top15}
    
    # 1. Проверка: есть ли предмет одежды на фотографии?
    clothing_score = sum(p for _, name, p in top15 if any(w in name.lower() for w in CLOTHING_KEYWORDS))
    if clothing_score < 0.05 or (clothing_score < 0.25 and bg_std > 25.0):
        return None, "На фотографии не обнаружен предмет одежды или обуви. Пожалуйста, загрузите фотографию вещи на однородном фоне."
        
    # 2. Предикт базовой нейросети Fashion-MNIST (CNN)
    fmnist_tensor = create_fmnist_tensor(arr)
    interp_fmnist.set_tensor(in_fmnist, fmnist_tensor)
    interp_fmnist.invoke()
    fmnist_preds = interp_fmnist.get_tensor(out_fmnist)[0]
    
    # 3. Интеллектуальный ансамбль признаков
    scores = {c: float(fmnist_preds[CLASSES.index(c)]) * 0.2 for c in CLASSES}
    
    # Анализ пропорций и геометрии объекта
    diff = np.linalg.norm(arr - bg_rgb, axis=-1)
    mask = diff > 15.0
    y, x_idx = np.where(mask)
    aspect = (x_idx.max() - x_idx.min() + 1) / (y.max() - y.min() + 1) if len(y) > 0 else 1.0
    
    # Анализ текстуры (полоски / вязаная ткань / пуговицы)
    gray_arr = np.array(raw_img.convert("L"))
    grad_x = np.abs(gray_arr[:, 1:] - gray_arr[:, :-1]).mean()
    grad_y = np.abs(gray_arr[1:, :] - gray_arr[:-1, :]).mean()
    stripe_ratio = grad_x / (grad_y + 1e-5)
    
    # Категориальные вероятности
    bag_p = sum(p for name, p in top_dict.items() if any(w in name for w in ["mailbag", "purse", "handbag", "backpack", "wallet"]))
    sandal_p = sum(p for name, p in top_dict.items() if any(w in name for w in ["sandal", "clog"]))
    sneaker_p = sum(p for name, p in top_dict.items() if any(w in name for w in ["running_shoe", "sneaker", "tennis_shoe", "loafer"]))
    boot_p = sum(p for name, p in top_dict.items() if any(w in name for w in ["cowboy_boot", "boot", "high_heeled_shoe", "stiletto"]))
    trouser_p = sum(p for name, p in top_dict.items() if any(w in name for w in ["jean", "pajama", "trouser", "pants"]))
    knit_p = sum(p for name, p in top_dict.items() if any(w in name for w in ["cardigan", "wool", "sweater", "pullover"]))
    dress_p = sum(p for name, p in top_dict.items() if any(w in name for w in ["gown", "hoopskirt", "overskirt", "cocktail_dress", "dress"]))
    coat_p = sum(p for name, p in top_dict.items() if any(w in name for w in ["trench_coat", "coat", "fur_coat", "cloak", "overcoat", "military_uniform"]))
    tie_p = sum(p for name, p in top_dict.items() if any(w in name for w in ["windsor_tie", "bow_tie"]))
    
    # Сумки
    if bag_p > 0.1:
        scores["Сумка"] += bag_p * 3.0
    
    # Обувь (Сандалии / Кроссовки / Ботильоны)
    if sandal_p > 0.15:
        scores["Сандалии"] += sandal_p * 3.0
    elif sneaker_p > 0.2:
        scores["Кроссовки"] += sneaker_p * 3.0
    elif boot_p > 0.02 or (top_dict.get("velvet", 0) > 0.05 and trouser_p < 0.05 and knit_p < 0.1 and coat_p < 0.1):
        scores["Ботильоны"] += 1.5
        scores["Сандалии"] *= 0.1
        scores["Платье"] *= 0.05
        
    # Платья
    if dress_p > 0.15:
        scores["Платье"] += dress_p * 3.0
        
    # Брюки
    if trouser_p > 0.1 or aspect < 0.4:
        scores["Брюки"] += (trouser_p * 3.0) + (1.5 if aspect < 0.4 else 0.0)
        scores["Платье"] *= 0.05
        scores["Ботильоны"] *= 0.05
        
    # Свитеры / Пуловеры
    if knit_p > 0.15:
        scores["Свитер"] += knit_p * 2.5
        scores["Ботильоны"] *= 0.05
        
    # Пальто vs Рубашка
    if coat_p > 0.1:
        if stripe_ratio > 1.15 or tie_p > 0.015:
            scores["Рубашка"] += 2.0
            scores["Пальто"] *= 0.1
            scores["Свитер"] *= 0.1
        else:
            scores["Пальто"] += coat_p * 2.5
            scores["Платье"] *= 0.05
            scores["Свитер"] *= 0.1
            scores["Ботильоны"] *= 0.05
            
    # Нормализация в процентные вероятности
    total = sum(scores.values()) or 1.0
    probs = {k: v / total for k, v in scores.items()}
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    top_class, top_conf = sorted_probs[0]
    
    other_preds = []
    for cls_name, prob in sorted_probs[1:4]:
        if prob > 0.005:
            other_preds.append({
                "name": cls_name,
                "prob": f"{prob * 100:.1f}%",
                "percent": round(prob * 100, 1)
            })
            
    return {
        "predicted_class": top_class,
        "confidence": f"{top_conf * 100:.1f}%",
        "confidence_percent": round(top_conf * 100, 1),
        "other_preds": other_preds,
        "image_base64": img_b64
    }, None

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
        res, err = classify_uploaded_image(content)
        if err:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": err,
                "classes": CLASSES
            })
            
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "predicted_class": res["predicted_class"],
                "confidence": res["confidence"],
                "confidence_percent": res["confidence_percent"],
                "other_preds": res["other_preds"],
                "image_base64": res["image_base64"],
                "classes": CLASSES
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Ошибка при обработке изображения: {e}",
                "classes": CLASSES
            }
        )

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        res, err = classify_uploaded_image(content)
        if err:
            return JSONResponse({"success": False, "error": err})
            
        return JSONResponse({
            "success": True,
            "predicted_class": res["predicted_class"],
            "confidence": res["confidence"],
            "confidence_percent": res["confidence_percent"],
            "other_preds": res["other_preds"],
            "image_base64": res["image_base64"]
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Ошибка при обработке изображения: {e}"
        })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
