"""
AGNI - Adaptive Geo-secured Network for Intelligent Agriculture.
FastAPI backend: auth, real OpenCV image analysis, trust score, advisory, crop recommendations.
"""
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from auth import verify_user, create_access_token, verify_token
from image_processor import (
    analyze_image,
    get_segmentation_preview,
    get_overlay_mask,
    get_binary_mask_png,
    get_overlay_mask_png,
    get_landuse_mask_png,
    get_polygons_from_cultivated_mask,
    overlay_polygons_on_image,
    compute_deterministic_confidence,
    clip_cultivated_mask_by_boundary,
)
from trust_engine import get_advisory
from crop_database import recommend_crops, get_season, get_season_display_name

# Directory for saving cultivated overlay images (relative to backend)
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AGNI API", version="1.0.0")

# Serve saved overlay images at /outputs
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# In-memory analysis history (last N entries used by GET /history)
analysis_history: List[dict] = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response models ---
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _cultivation_classification(cultivated_percentage: float) -> str:
    """Classification from cultivated land share."""
    if cultivated_percentage > 70:
        return "Highly Cultivated"
    if cultivated_percentage >= 40:
        return "Moderately Cultivated"
    return "Poorly Cultivated"


class AnalyzeResponse(BaseModel):
    cultivated_percentage: float
    stress_percentage: float
    trust_score: float
    mask_confidence: float
    advisory: str
    is_blurry: bool
    classification: str
    total_pixels: int
    cultivated_pixels: int
    non_cultivated_pixels: int
    coverage_ratio: float
    contour_count: int
    sharpness_score: float
    veg_balance_score: float
    fragmentation_score: float
    noise_score: float
    laplacian_variance: float
    vegetation_ratio: float
    noise_ratio: float
    clarity_score: float
    edge_consistency_score: float
    area_plausibility_score: float
    exg_variance: float
    original_resolution: Tuple[int, int]
    processing_resolution: Tuple[int, int]
    segmentation_image_base64: Optional[str] = None
    binary_mask_base64: Optional[str] = None
    overlay_mask_base64: Optional[str] = None
    polygon_coordinates: Optional[List[List[List[float]]]] = None
    cultivated_mask_image_path: Optional[str] = None
    within_boundary: bool = False
    land_use_mask_base64: Optional[str] = None
    # Land-use breakdown percentages
    water_percentage: float = 0.0
    road_percentage: float = 0.0
    building_percentage: float = 0.0
    vegetation_percentage: float = 0.0
    farmable_space_percentage: float = 0.0


def get_token(authorization: Optional[str] = Header(None)) -> str:
    """Extract and validate Bearer token from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing token")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = parts[1]
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return token


@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """Login with username/password; returns JWT."""
    if not verify_user(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(data={"sub": req.username})
    return LoginResponse(access_token=token)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    file: UploadFile = File(...),
    boundary: Optional[str] = Form(None),
    token: str = Depends(get_token),
):
    """Analyze uploaded image with real OpenCV processing; optional farm boundary (geo polygon) for clipping."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image: file must be an image")

    try:
        contents = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: could not read file - {str(e)}")

    if not contents:
        raise HTTPException(status_code=400, detail="Invalid image: empty file")

    try:
        cultivated_percentage, stress_percentage, image, binary_mask, removed_pixels, original_resolution, processing_resolution = analyze_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OpenCV processing failed: {str(e)}")

    h, w = image.shape[:2]
    total_pixels = h * w
    cultivated_pixels = int(cv2.countNonZero(binary_mask))
    use_boundary = False
    polygon_lat_lng = None

    if boundary and boundary.strip():
        try:
            polygon_lat_lng = json.loads(boundary)
            if isinstance(polygon_lat_lng, list) and len(polygon_lat_lng) >= 3:
                clipped_mask, cultivated_pixels, total_pixels = clip_cultivated_mask_by_boundary(
                    binary_mask, polygon_lat_lng
                )
                binary_mask = clipped_mask
                cultivated_percentage = (cultivated_pixels / total_pixels * 100.0) if total_pixels else 0.0
                cultivated_percentage = round(cultivated_percentage, 2)
                use_boundary = True
            else:
                raise HTTPException(status_code=400, detail="Polygon must have at least 3 points")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid boundary JSON")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    non_cultivated_pixels = total_pixels - cultivated_pixels
    coverage_ratio = round(cultivated_pixels / total_pixels, 4) if total_pixels else 0.0

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    conf = compute_deterministic_confidence(
        image, binary_mask, contour_count, total_pixels, cultivated_pixels
    )
    mask_confidence = conf["confidence"]
    clarity_score = conf["clarity_score"]
    fragmentation_score = conf["fragmentation_score"]
    edge_consistency_score = conf["edge_consistency_score"]
    area_plausibility_score = conf["area_plausibility_score"]
    exg_variance = conf["exg_variance"]

    vegetation_ratio = coverage_ratio
    laplacian_variance = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    noise_ratio = round(removed_pixels / total_pixels, 4) if total_pixels else 0.0
    noise_score = max(0.0, min(1.0, 1.0 - noise_ratio))
    sharpness_score = round(min(laplacian_variance / 500.0, 1.0), 4)
    veg_balance_score = max(0.0, min(1.0, 1.0 - abs(0.5 - vegetation_ratio)))
    veg_balance_score = round(veg_balance_score, 4)

    is_blurry = laplacian_variance < 100
    trust_score_display = round(mask_confidence)
    advisory = get_advisory(cultivated_percentage, stress_percentage, trust_score_display)
    classification = _cultivation_classification(cultivated_percentage)
    analysis_history.append({
        "cultivated_percentage": cultivated_percentage,
        "stress_percentage": stress_percentage,
        "trust_score": mask_confidence,
        "advisory": advisory,
        "classification": classification,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    seg_bytes = get_segmentation_preview(image, binary_mask)
    seg_b64 = base64.b64encode(seg_bytes).decode("ascii")
    overlay_bgr = get_overlay_mask(image, binary_mask)
    binary_b64 = base64.b64encode(get_binary_mask_png(binary_mask)).decode("ascii")
    overlay_b64 = base64.b64encode(get_overlay_mask_png(overlay_bgr)).decode("ascii")
    land_use_bytes = get_landuse_mask_png(image, binary_mask)
    land_use_b64 = base64.b64encode(land_use_bytes).decode("ascii")

    # --- Land-use breakdown percentages ---
    from image_processor import detect_water, detect_roads, detect_buildings
    water_mask = detect_water(image)
    road_mask = detect_roads(image)
    building_mask = detect_buildings(image, binary_mask)
    water_pixels = int(cv2.countNonZero(water_mask))
    road_pixels = int(cv2.countNonZero(road_mask))
    building_pixels = int(cv2.countNonZero(building_mask))
    water_pct = round(water_pixels / total_pixels * 100, 2) if total_pixels else 0.0
    road_pct = round(road_pixels / total_pixels * 100, 2) if total_pixels else 0.0
    building_pct = round(building_pixels / total_pixels * 100, 2) if total_pixels else 0.0
    veg_pct = round(cultivated_pixels / total_pixels * 100, 2) if total_pixels else 0.0
    farmable_space_pct = round(max(0, 100 - water_pct - road_pct - building_pct), 2)

    # Polygon mask from cultivated area (findContours + approxPolyDP), JSON-serializable
    polygon_coordinates = get_polygons_from_cultivated_mask(binary_mask)
    polygon_overlay_bgr = overlay_polygons_on_image(image.copy(), polygon_coordinates)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    overlay_filename = f"cultivated_{ts}.png"
    overlay_path = OUTPUTS_DIR / overlay_filename
    cv2.imwrite(str(overlay_path), polygon_overlay_bgr)
    cultivated_mask_image_path = f"/outputs/{overlay_filename}"

    return AnalyzeResponse(
        cultivated_percentage=cultivated_percentage,
        stress_percentage=stress_percentage,
        trust_score=mask_confidence,
        mask_confidence=mask_confidence,
        advisory=advisory,
        is_blurry=is_blurry,
        classification=classification,
        total_pixels=total_pixels,
        cultivated_pixels=cultivated_pixels,
        non_cultivated_pixels=non_cultivated_pixels,
        coverage_ratio=coverage_ratio,
        contour_count=contour_count,
        sharpness_score=sharpness_score,
        veg_balance_score=veg_balance_score,
        fragmentation_score=fragmentation_score,
        noise_score=noise_score,
        laplacian_variance=laplacian_variance,
        vegetation_ratio=vegetation_ratio,
        noise_ratio=noise_ratio,
        clarity_score=clarity_score,
        edge_consistency_score=edge_consistency_score,
        area_plausibility_score=area_plausibility_score,
        exg_variance=exg_variance,
        original_resolution=original_resolution,
        processing_resolution=processing_resolution,
        segmentation_image_base64=seg_b64,
        binary_mask_base64=binary_b64,
        overlay_mask_base64=overlay_b64,
        polygon_coordinates=polygon_coordinates,
        cultivated_mask_image_path=cultivated_mask_image_path,
        within_boundary=use_boundary,
        land_use_mask_base64=land_use_b64,
        water_percentage=water_pct,
        road_percentage=road_pct,
        building_percentage=building_pct,
        vegetation_percentage=veg_pct,
        farmable_space_percentage=farmable_space_pct,
    )


@app.get("/history")
def history(token: str = Depends(get_token)) -> List[dict]:
    """Return last 5 analyses. Protected by JWT."""
    return analysis_history[-5:][::-1]


@app.get("/weather")
def get_weather(city: str, token: str = Depends(get_token)):
    """Get weather data for a city using OpenWeatherMap API."""
    try:
        import requests
    except ImportError:
        raise HTTPException(status_code=500, detail="requests library not installed. Run: pip install requests")
    
    WEATHER_API_KEY = "42a28b0639a88def3bb8150c4d61f466"
    if not WEATHER_API_KEY:
        raise HTTPException(status_code=500, detail="Weather API key not configured")
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        
        # Check if request was successful
        if response.status_code != 200:
            error_data = response.json() if response.text else {}
            raise HTTPException(
                status_code=400, 
                detail=f"Weather API error: {error_data.get('message', f'HTTP {response.status_code}')}"
            )
        
        data = response.json()
        
        # Check API response code
        if data.get("cod") == 200:
            return {
                "success": True,
                "name": data.get("name"),
                "temp": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
                "wind_speed": data.get("wind", {}).get("speed"),
                "weather_main": data.get("weather", [{}])[0].get("main") if data.get("weather") else "Unknown",
                "weather_description": data.get("weather", [{}])[0].get("description") if data.get("weather") else "unknown",
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Weather API error: {data.get('message', 'Unknown error')}"
            )
    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=500, detail="Weather API request timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch weather: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/weather-by-coords")
def get_weather_by_coords(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    token: str = Depends(get_token),
):
    """Get weather data by coordinates using OpenWeatherMap API. Auto-detects city name."""
    WEATHER_API_KEY = "42a28b0639a88def3bb8150c4d61f466"
    if not WEATHER_API_KEY:
        raise HTTPException(status_code=500, detail="Weather API key not configured")
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            error_data = response.json() if response.text else {}
            raise HTTPException(
                status_code=400,
                detail=f"Weather API error: {error_data.get('message', f'HTTP {response.status_code}')}"
            )
        data = response.json()
        if data.get("cod") == 200:
            return {
                "success": True,
                "name": data.get("name", "Unknown"),
                "country": data.get("sys", {}).get("country", ""),
                "lat": lat,
                "lon": lon,
                "temp": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
                "wind_speed": data.get("wind", {}).get("speed"),
                "weather_main": data.get("weather", [{}])[0].get("main") if data.get("weather") else "Unknown",
                "weather_description": data.get("weather", [{}])[0].get("description") if data.get("weather") else "unknown",
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Weather API error: {data.get('message', 'Unknown error')}"
            )
    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=500, detail="Weather API request timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch weather: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/recommend-crops")
def get_crop_recommendations(
    temp: float = Query(..., description="Temperature in Celsius"),
    humidity: float = Query(..., description="Humidity percentage"),
    lat: float = Query(..., description="Latitude"),
    cultivated_pct: float = Query(50.0, description="Currently cultivated percentage"),
    water_pct: float = Query(5.0, description="Water detected percentage"),
    farmable_space_pct: float = Query(50.0, description="Farmable space percentage"),
    token: str = Depends(get_token),
):
    """Get crop recommendations based on conditions."""
    now = datetime.now()
    month = now.month
    result = recommend_crops(
        temperature=temp,
        humidity=humidity,
        latitude=lat,
        month=month,
        cultivated_pct=cultivated_pct,
        water_pct=water_pct,
        farmable_space_pct=farmable_space_pct,
    )
    return {"success": True, **result}


@app.get("/health")
def health():
    return {
        "status": "active",
        "model_version": "2.0",
        "security_layer": "JWT enabled",
        "features": ["image_analysis", "weather", "crop_recommendations", "land_use_detection"],
    }
