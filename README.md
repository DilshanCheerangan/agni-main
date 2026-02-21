# AGNI – Adaptive Geo-secured Network for Intelligent Agriculture

Production-style, hackathon-friendly MVP: login, image upload, OpenCV-based vegetation/stress analysis, trust score, and advisory.

---

## How to open (complete steps)

1. **Install once** (if not done):
   ```powershell
   cd "c:\Users\Tony Stark\Desktop\projects\agni"
   pip install -r requirements.txt
   ```

2. **Start the backend** (leave this window open):
   ```powershell
   cd "c:\Users\Tony Stark\Desktop\projects\agni\backend"
   python -m uvicorn main:app --reload
   ```
   Wait until you see *Application startup complete*.

3. **Start the frontend** (open a **new** PowerShell window):
   ```powershell
   cd "c:\Users\Tony Stark\Desktop\projects\agni\frontend"
   python -m http.server 8080
   ```

4. **Open in browser:**  
   Go to **http://localhost:8080**

5. **Log in:**  
   Username: `agni` · Password: `farm2025`  
   Then upload a field/crop image and click **Analyze image**.

---

## Quick start (reference)

### 1. Install dependencies

**PowerShell (path has spaces – use quotes):**

```powershell
cd "c:\Users\Tony Stark\Desktop\projects\agni"
pip install -r requirements.txt
```

### 2. Run the backend

**Option A – Run script (easiest in PowerShell):**

```powershell
cd "c:\Users\Tony Stark\Desktop\projects\agni\backend"
.\run.ps1
```

**Option B – Manual (quote the path, then run uvicorn):**

```powershell
cd "c:\Users\Tony Stark\Desktop\projects\agni\backend"
python -m uvicorn main:app --reload
```

Do not pass the path as an argument to uvicorn; only run the two commands above in order.

Backend runs at **http://127.0.0.1:8000**.  
Docs: http://127.0.0.1:8000/docs

### 3. Open the frontend

**Option A – Simple HTTP server (recommended)**

```powershell
cd "c:\Users\Tony Stark\Desktop\projects\agni\frontend"
python -m http.server 8080
```

Then open **http://localhost:8080** in your browser.

**Option B – Open file directly**

Open `frontend/index.html` in your browser (double-click or drag into Chrome/Edge).  
If the backend is at `http://127.0.0.1:8000`, login and analyze should still work thanks to CORS.

### 4. Test login and analyze

- **Login**  
  - Username: `agni`  
  - Password: `farm2025`  
  - In the UI: enter credentials and click “Log in”.  
  - Or with curl:
    ```bash
    curl -X POST http://127.0.0.1:8000/login -H "Content-Type: application/json" -d "{\"username\":\"agni\",\"password\":\"farm2025\"}"
    ```
  - Copy the `access_token` from the response.

- **Analyze**  
  - In the UI: after login, choose an image and click “Analyze image”.  
  - Or with curl (replace `YOUR_TOKEN` and `path/to/image.jpg`):
    ```bash
    curl -X POST http://127.0.0.1:8000/analyze -H "Authorization: Bearer YOUR_TOKEN" -F "file=@path/to/image.jpg"
    ```

## Optional: Better building vs soil detection (free AI)

Buildings and bare soil can look similar (brown/tan). The pipeline does two things:

1. **Rule-based:** Brown regions are classified as buildings only if they are compact and not mostly soil (low overlap with bare-soil mask and reasonably rectangular shape).
2. **Optional AI (Hugging Face):** If you set a free [Hugging Face token](https://huggingface.co/settings/tokens), the backend can call a semantic segmentation model to add building pixels. Set either:
   - `BUILDING_SEGMENTATION_HF_TOKEN`, or  
   - `HF_TOKEN`  
   in the environment before starting the backend. No token = rule-based only (no API calls).

## Error handling

- **Invalid image** → 400 with message (e.g. “Invalid image: could not decode”).
- **Missing token** on `/analyze` → 401 “Missing token”.
- **Invalid or expired token** → 401 “Invalid or expired token”.
- All errors are JSON, e.g. `{"detail": "..."}`.

## Tech stack

- **Backend:** Python, FastAPI, OpenCV, NumPy, Uvicorn, JWT (python-jose).
- **Frontend:** HTML + Tailwind CSS (CDN), Fetch API.
- **Storage:** None (in-memory; mock user only).
