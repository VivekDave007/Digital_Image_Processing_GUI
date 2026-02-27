from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import base64

# Import your existing experiment modules here as needed
# from chapter2_experiments import ...

app = FastAPI(title="DIP Workbench API")

# Setup CORS to allow the React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development. In production, specify the frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper to convert OpenCV image to Base64 for the frontend ---
def img_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# --- Endpoints (Phase 1 Stub) ---
@app.get("/")
def read_root():
    return {"status": "DIP Engine Backend is Online"}

@app.post("/api/perception/simulate")
def simulate_perception(illumination: float = Form(...), mach_bands: bool = Form(...)):
    """
    Simulates the Visual Perception module (Module 1.1).
    Takes linear illumination (0.0 to 1.0) and a boolean for mach bands.
    """
    w, h = 600, 400
    if mach_bands:
        # Draw Mach Bands
        img = np.zeros((h, w, 3), dtype=np.uint8)
        steps = 10
        sw = w // steps
        for i in range(steps):
            val = int(255 * (i/steps) * illumination)
            cv2.rectangle(img, (i*sw, 0), ((i+1)*sw, h), (val, val, val), -1)
        
        return JSONResponse(content={
            "image": img_to_base64(img),
            "telemetry": {
                "mode": "Mach Band Demonstration",
                "steps": steps,
                "max_intensity_applied": int(255 * illumination)
            }
        })
    else:
        # Draw Flower
        img = np.zeros((h, w, 3), dtype=np.uint8)
        bg = int(255 * illumination)
        img[:] = (bg, bg, bg)
        
        is_scotopic = illumination < 0.2
        center = (w//2, h//2)
        radius = 120
        
        if is_scotopic:
            color_flower = (100, 100, 100) # Gray
            color_center = (50, 50, 50)
            vision_mode = "Scotopic (Rods)"
        else:
            color_flower = (50, 50, 255) # Red (BGR)
            color_center = (0, 255, 255) # Yellow (BGR)
            vision_mode = "Photopic (Cones)"
            
        cv2.circle(img, center, radius, color_flower, -1)
        cv2.circle(img, center, 45, color_center, -1)
        
        return JSONResponse(content={
            "image": img_to_base64(img),
            "telemetry": {
                "vision_mode": vision_mode,
                "background_intensity": bg,
                "scotopic_active": is_scotopic
            }
        })

@app.post("/api/stats/process")
async def process_image_statistics(
    mode: str = Form(...), 
    noise_param_1: float = Form(0.0), 
    noise_param_2: float = Form(0.0)
):
    """
    Simulates Image Statistics (Topic D).
    mode: 'contrast', 'noise_gaussian', 'noise_sp', 'basic'
    """
    # Generate Synthetic Base Image (Gradient + Rectangle)
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(200):
        img[i, :] = i
    cv2.rectangle(img, (50, 50), (150, 150), (100, 100, 100), -1)
    
    output_img = img.copy()
    telemetry = {}
    
    # 1. Basic Stats Helper
    def get_stats(image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        return {
            "mean": round(float(np.mean(image)), 2),
            "variance": round(float(np.var(image)), 2),
            "std_dev": round(float(np.std(image)), 2),
            "min": int(np.min(image)),
            "max": int(np.max(image)),
            "histogram": hist.tolist()
        }

    # 2. Logic Router
    if mode == "contrast":
        min_val, max_val = np.min(img), np.max(img)
        if max_val - min_val > 0:
            stretched = 255.0 * (img - min_val) / (max_val - min_val)
            output_img = np.uint8(stretched)
        telemetry["action"] = "Contrast Stretching"
        
    elif mode == "noise_gaussian":
        mean = noise_param_1
        sigma = noise_param_2
        gauss = np.random.normal(mean, sigma, img.shape)
        noisy = img.astype(np.float32) + gauss
        output_img = np.clip(noisy, 0, 255).astype(np.uint8)
        telemetry["action"] = f"Gaussian Noise (μ={mean}, σ={sigma})"
        
    elif mode == "noise_sp":
        prob = noise_param_1
        thres = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = np.random.random()
                if rdn < prob:
                    output_img[i][j] = 0
                elif rdn > thres:
                    output_img[i][j] = 255
        telemetry["action"] = f"Salt & Pepper (p={prob})"
        
    else: # Basic
        telemetry["action"] = "Original Image Analysis"

    telemetry["stats"] = get_stats(output_img)
    
    return JSONResponse(content={
        "image_original": img_to_base64(img),
        "image": img_to_base64(output_img),
        "telemetry": telemetry
    })

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
