"""
AGNI - Real OpenCV image analysis (vegetation and stress detection).
No mock data; all values from actual image processing.
Resolution-agnostic vegetation detection via Excess Green (ExG) index.
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any

# Resolution-agnostic pipeline: max dimension for ExG processing
EXG_MAX_DIMENSION = 640  # Slightly higher for better detail
MIN_VEGETATION_COMPONENT_PIXELS = 200  # Keep smaller green patches
MORPH_KERNEL_SIZE = (7, 7)  # Larger kernel to fill gaps in vegetation
MORPH_CLOSE_ITERATIONS = 2  # Extra closing to merge nearby green
# Include dark green: pixels with normalized ExG >= this and green-dominant (G>R, G>B) are vegetation
EXG_DARK_GREEN_MIN = 8  # Lower threshold to capture more green
EXG_PALE_GREEN_MIN = 5   # Very pale green still counts if G-dominant

# HSV green range (OpenCV H 0-180): typical vegetation green
HSV_GREEN_H_LOWER, HSV_GREEN_H_UPPER = 28, 85
HSV_GREEN_S_LOWER, HSV_GREEN_S_UPPER = 40, 255
HSV_GREEN_V_LOWER, HSV_GREEN_V_UPPER = 40, 255

# Non-agricultural object detection parameters
MIN_BUILDING_AREA = 1000  # Minimum area for building detection
MIN_ROAD_LINE_LENGTH = 80  # Stricter: longer lines only (reduces false roads)

# Texture-based cultivated filtering (7x7 window)
TEXTURE_WINDOW_SIZE = 7
TEXTURE_VARIANCE_THRESHOLD_HIGH = 250     # Remove regions where normalized variance > this (0-255) - very lenient
TEXTURE_VARIANCE_THRESHOLD_LOW = 3       # Keep only moderate texture (exclude very flat) - very lenient
MIN_TEXTURE_COMPONENT_PIXELS = 500       # Remove small isolated components after texture filter
MIN_VEGETATION_RETENTION = 0.10          # Minimum 10% of vegetation must remain after texture filter (safeguard)

# Deterministic confidence: score scaling (no arbitrary subtraction)
EXG_VARIANCE_SCALE = 2000.0   # clarity = max(0, 1 - exg_variance / scale)
MAX_CONTOURS_FRAGMENTATION = 200.0  # fragmentation_score = max(0, 1 - count / max)
AREA_PLAUSIBLE_LOW = 0.10   # 10%
AREA_PLAUSIBLE_HIGH = 0.80  # 80%

# HSV ranges for stress (yellow) only (OpenCV: H 0-180, S 0-255, V 0-255)
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

# HSV for bare soil / non-green land (brown, tan, beige): H 8-25, S 30-180, V 60-220
BARE_SOIL_H_LOWER, BARE_SOIL_H_UPPER = 8, 25
BARE_SOIL_S_LOWER, BARE_SOIL_S_UPPER = 30, 180
BARE_SOIL_V_LOWER, BARE_SOIL_V_UPPER = 60, 220
MIN_BARE_SOIL_COMPONENT_PIXELS = 300


def _resize_max_dimension(img: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image so the longest side is max_dim; preserve aspect ratio. Returns (resized, scale, (orig_w, orig_h))."""
    h, w = img.shape[:2]
    orig_shape = (w, h)
    if max(h, w) <= max_dim:
        return img.copy(), 1.0, orig_shape
    scale = max_dim / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale, orig_shape


def detect_water(img: np.ndarray) -> np.ndarray:
    """
    Detect water (blue-dominant pixels): B > G AND B > R.
    Refine using morphological opening.
    Returns binary mask (255 = water, 0 = non-water).
    """
    b, g, r = cv2.split(img.astype(np.float32))
    blue_dominant = (b > g) & (b > r)
    water_mask = (blue_dominant.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    return water_mask


def detect_roads(img: np.ndarray) -> np.ndarray:
    """
    Detect roads using edge detection and HoughLinesP.
    Steps: grayscale, Canny edges, HoughLinesP, convert lines to mask.
    Returns binary mask (255 = road, 0 = non-road).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    h, w = gray.shape
    min_line_length = max(MIN_ROAD_LINE_LENGTH, int(min(h, w) * 0.1))
    max_line_gap = int(min(h, w) * 0.05)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=50,
        minLineLength=min_line_length, maxLineGap=max_line_gap
    )
    road_mask = np.zeros((h, w), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(road_mask, (x1, y1), (x2, y2), 255, 3)
        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv2.dilate(road_mask, kernel, iterations=2)
    return road_mask


def detect_buildings(img: np.ndarray, vegetation_mask: np.ndarray) -> np.ndarray:
    """
    Detect buildings: low-vegetation high-contrast regions with rectangular shapes.
    Steps: find low-vegetation areas, detect high-contrast regions, find rectangular contours.
    Returns binary mask (255 = building, 0 = non-building).
    """
    h, w = img.shape[:2]
    building_mask = np.zeros((h, w), dtype=np.uint8)
    non_vegetation = (vegetation_mask == 0).astype(np.uint8) * 255
    if cv2.countNonZero(non_vegetation) == 0:
        return building_mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_masked = cv2.bitwise_and(edges, edges, mask=non_vegetation)
    contours, _ = cv2.findContours(edges_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_BUILDING_AREA:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) >= 4:
            x, y, w_rect, h_rect = cv2.boundingRect(approx)
            aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0
            if 0.3 <= aspect_ratio <= 3.0:
                cv2.drawContours(building_mask, [approx], -1, 255, -1)
    kernel = np.ones((3, 3), np.uint8)
    building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, kernel)
    return building_mask


def get_non_agricultural_mask(img: np.ndarray, vegetation_mask: np.ndarray) -> np.ndarray:
    """
    Combine water, road, and building masks into non-agricultural mask.
    Returns binary mask (255 = non-agricultural, 0 = agricultural).
    """
    water_mask = detect_water(img)
    road_mask = detect_roads(img)
    building_mask = detect_buildings(img, vegetation_mask)
    non_agriculture_mask = cv2.bitwise_or(water_mask, road_mask)
    non_agriculture_mask = cv2.bitwise_or(non_agriculture_mask, building_mask)
    return non_agriculture_mask


def get_non_agricultural_mask_no_roads(img: np.ndarray, vegetation_mask: np.ndarray) -> np.ndarray:
    """
    Water + buildings only (no roads). Use this when building vegetation/farmable masks
    so that false road detections (Hough lines) do not create thick black line artifacts.
    Roads are still shown on the land-use map via detect_roads() separately.
    """
    water_mask = detect_water(img)
    building_mask = detect_buildings(img, vegetation_mask)
    return cv2.bitwise_or(water_mask, building_mask)


def detect_vegetation_hsv(img: np.ndarray) -> np.ndarray:
    """
    Detect green vegetation using HSV. Use alongside ExG for fuller coverage.
    Returns binary mask (255 = green vegetation).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([HSV_GREEN_H_LOWER, HSV_GREEN_S_LOWER, HSV_GREEN_V_LOWER])
    upper = np.array([HSV_GREEN_H_UPPER, HSV_GREEN_S_UPPER, HSV_GREEN_V_UPPER])
    green = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    return green


def detect_bare_soil(img: np.ndarray) -> np.ndarray:
    """
    Detect bare soil / non-green cultivable land (brown, tan, beige) using HSV.
    Excludes very dark (shadow) and very bright (sand/road). Returns binary mask (255 = bare soil).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([BARE_SOIL_H_LOWER, BARE_SOIL_S_LOWER, BARE_SOIL_V_LOWER])
    upper = np.array([BARE_SOIL_H_UPPER, BARE_SOIL_S_UPPER, BARE_SOIL_V_UPPER])
    bare = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    bare = cv2.morphologyEx(bare, cv2.MORPH_CLOSE, kernel)
    bare = cv2.morphologyEx(bare, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(bare, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(bare)
    for c in contours:
        if cv2.contourArea(c) >= MIN_BARE_SOIL_COMPONENT_PIXELS:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


def get_farmable_mask(
    img: np.ndarray,
    vegetation_mask: np.ndarray,
) -> np.ndarray:
    """
    Farmable = vegetation OR bare soil, excluding only water and buildings (not roads).
    Avoids false road detections creating line artifacts. Returns binary mask (255 = farmable).
    """
    bare_soil_mask = detect_bare_soil(img)
    combined = cv2.bitwise_or(vegetation_mask, bare_soil_mask)
    non_ag = get_non_agricultural_mask_no_roads(img, vegetation_mask)
    farmable = cv2.subtract(combined, non_ag)
    farmable = np.clip(farmable, 0, 255).astype(np.uint8)
    return farmable


def _compute_local_variance_map(gray: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute local variance map using sliding window (box filter).
    Variance = E[X^2] - E[X]^2 per window.
    """
    w = window_size
    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, -1, (w, w), normalize=True)
    mean_sq = cv2.boxFilter(gray_f * gray_f, -1, (w, w), normalize=True)
    variance = mean_sq - (mean * mean)
    variance = np.maximum(variance, 0.0)
    return variance


def apply_texture_filter(cultivated_mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Enhance cultivated mask using texture analysis.
    Cultivated fields have moderate uniform texture (lower variance than forests).
    Remove high-variance regions, keep moderate range, closing, remove small components.
    Includes safeguard: if too much vegetation is removed, fall back to original mask.
    Returns final cultivated_mask.
    """
    original_pixels = int(cv2.countNonZero(cultivated_mask))
    if original_pixels == 0:
        return cultivated_mask
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance_map = _compute_local_variance_map(gray, TEXTURE_WINDOW_SIZE)
    v_min, v_max = variance_map.min(), variance_map.max()
    v_range = v_max - v_min
    if v_range <= 0:
        variance_norm = np.zeros_like(variance_map, dtype=np.float32)
    else:
        variance_norm = ((variance_map - v_min) / v_range * 255.0).astype(np.float32)
    variance_uint8 = np.clip(variance_norm, 0, 255).astype(np.uint8)

    # Remove regions where local_variance > threshold_high; keep moderate range
    high_var_mask = (variance_uint8 > TEXTURE_VARIANCE_THRESHOLD_HIGH)
    low_var_mask = (variance_uint8 < TEXTURE_VARIANCE_THRESHOLD_LOW)
    texture_keep = ~high_var_mask & ~low_var_mask
    texture_keep_uint8 = (texture_keep.astype(np.uint8)) * 255

    # Apply to cultivated mask: keep only where texture is in moderate range
    filtered_mask = cv2.bitwise_and(cultivated_mask, texture_keep_uint8)
    
    # Safeguard: if texture filter removed too much, use original mask
    filtered_pixels = int(cv2.countNonZero(filtered_mask))
    retention_ratio = filtered_pixels / original_pixels if original_pixels > 0 else 0.0
    if retention_ratio < MIN_VEGETATION_RETENTION:
        # Texture filter too aggressive, return original mask with just morphological operations
        kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
        cultivated_mask = cv2.morphologyEx(cultivated_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cultivated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(cultivated_mask)
        for c in contours:
            if cv2.contourArea(c) >= MIN_TEXTURE_COMPONENT_PIXELS:
                cv2.drawContours(out, [c], -1, 255, -1)
        return out

    # Morphological closing again
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

    # Remove small isolated components
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(filtered_mask)
    for c in contours:
        if cv2.contourArea(c) >= MIN_TEXTURE_COMPONENT_PIXELS:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


def exg_vegetation_pipeline(img: np.ndarray) -> Tuple[np.ndarray, int, int, int, Tuple[int, int], Tuple[int, int]]:
    """
    Resolution-agnostic vegetation detection using Excess Green (ExG) index.
    Steps: resize to max 512, ExG, Otsu, morphological closing, remove small components,
    exclude non-agricultural objects (water, roads, buildings), then texture-based
    filtering (remove high-variance e.g. forest, keep moderate uniform texture),
    closing again, remove small isolated components.
    Returns (cultivated_mask, vegetation_pixel_count, total_pixel_count, removed_pixel_count,
             original_resolution, processing_resolution).
    """
    h_orig, w_orig = img.shape[:2]
    original_resolution = (w_orig, h_orig)
    total_pixel_count = h_orig * w_orig
    if total_pixel_count == 0:
        raise ValueError("Invalid image: zero dimensions")

    working, scale, (orig_w, orig_h) = _resize_max_dimension(img, EXG_MAX_DIMENSION)
    h_proc, w_proc = working.shape[:2]
    processing_resolution = (w_proc, h_proc)
    working_float = working.astype(np.float32)
    b, g, r = cv2.split(working_float)
    exg = 2.0 * g - r - b
    exg_min, exg_max = exg.min(), exg.max()
    exg_range = exg_max - exg_min
    if exg_range <= 0:
        exg_norm = np.zeros_like(exg, dtype=np.float32)
    else:
        exg_norm = ((exg - exg_min) / exg_range * 255.0).astype(np.float32)
    blurred = cv2.GaussianBlur(exg_norm, (5, 5), 0)
    blurred_uint8 = blurred.astype(np.uint8)
    _, binary_uint8 = cv2.threshold(blurred_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Include dark and pale green: green-dominant pixels (G>R, G>B) with ExG above threshold
    green_dominant = (g > r) & (g > b)
    dark_green = (exg_norm >= EXG_DARK_GREEN_MIN) & green_dominant
    pale_green = (exg_norm >= EXG_PALE_GREEN_MIN) & green_dominant
    dark_green_uint8 = (dark_green.astype(np.float32) * 255).astype(np.uint8)
    pale_green_uint8 = (pale_green.astype(np.float32) * 255).astype(np.uint8)
    binary_uint8 = cv2.bitwise_or(binary_uint8, dark_green_uint8)
    binary_uint8 = cv2.bitwise_or(binary_uint8, pale_green_uint8)
    # Merge HSV green for fuller vegetation coverage (catches greens ExG can miss)
    hsv_green_working = detect_vegetation_hsv(working)
    binary_uint8 = cv2.bitwise_or(binary_uint8, hsv_green_working)
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    closed = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)
    for _ in range(MORPH_CLOSE_ITERATIONS - 1):
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    pixels_before_removal = int(cv2.countNonZero(closed))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_working = np.zeros_like(closed)
    for c in contours:
        if cv2.contourArea(c) >= MIN_VEGETATION_COMPONENT_PIXELS:
            cv2.drawContours(mask_working, [c], -1, 255, -1)
    vegetation_pixel_count_working = int(cv2.countNonZero(mask_working))
    removed_pixel_count_working = max(0, pixels_before_removal - vegetation_pixel_count_working)
    
    # Upscale vegetation mask to original size if needed
    if scale >= 1.0:
        vegetation_mask_full = mask_working
    else:
        vegetation_mask_full = cv2.resize(
            mask_working, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
    
    # Exclude only water and buildings (NOT roads) to avoid Hough line artifacts in mask
    non_agriculture_mask = get_non_agricultural_mask_no_roads(img, vegetation_mask_full)
    non_ag_pixels = int(cv2.countNonZero(non_agriculture_mask))
    veg_pixels_before = int(cv2.countNonZero(vegetation_mask_full))
    
    # Safeguard: if non-agricultural mask covers too much (>80%), skip exclusion
    if veg_pixels_before > 0 and (non_ag_pixels / veg_pixels_before) > 0.8:
        cultivated_mask = vegetation_mask_full.copy()
    else:
        cultivated_mask = cv2.subtract(vegetation_mask_full, non_agriculture_mask)
        cultivated_mask = np.clip(cultivated_mask, 0, 255).astype(np.uint8)

    # Texture-based filtering: remove high-variance (e.g. forest), keep moderate uniform texture
    pixels_before_texture = int(cv2.countNonZero(cultivated_mask))
    cultivated_mask_filtered = apply_texture_filter(cultivated_mask, img)
    
    # Final safeguard: if texture filter removed too much, use pre-texture mask
    final_pixels = int(cv2.countNonZero(cultivated_mask_filtered))
    if pixels_before_texture > 0 and (final_pixels / pixels_before_texture) < MIN_VEGETATION_RETENTION:
        # Texture filter too aggressive, use mask before texture filtering
        cultivated_mask = cultivated_mask
    else:
        cultivated_mask = cultivated_mask_filtered

    vegetation_pixel_count = int(cv2.countNonZero(cultivated_mask))
    if scale >= 1.0:
        removed_pixel_count = removed_pixel_count_working
    else:
        working_pixels = mask_working.shape[0] * mask_working.shape[1]
        removed_pixel_count = int(round(removed_pixel_count_working * total_pixel_count / working_pixels)) if working_pixels else 0
        removed_pixel_count = max(0, removed_pixel_count)
    
    return cultivated_mask, vegetation_pixel_count, total_pixel_count, removed_pixel_count, original_resolution, processing_resolution


def get_segmentation_preview(img: np.ndarray, vegetation_mask: np.ndarray) -> bytes:
    """Build segmentation overlay for display: green = vegetation (ExG mask), amber = stress (HSV yellow). Returns PNG bytes."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    h, w = img.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = (45, 45, 45)
    out[vegetation_mask > 0] = (0, 180, 0)
    out[yellow_mask > 0] = (0, 165, 255)
    _, png = cv2.imencode(".png", out)
    return png.tobytes()


def get_binary_cultivated_mask(img: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Cultivated land masking via ExG vegetation pipeline: white = vegetation, black = non-vegetation.
    Returns (binary_mask, removed_pixel_count) for noise-ratio computation.
    """
    vegetation_mask, _, _, removed_pixel_count, _, _ = exg_vegetation_pipeline(img)
    return vegetation_mask, removed_pixel_count


def compute_deterministic_confidence(
    img: np.ndarray,
    cultivated_mask: np.ndarray,
    contour_count: int,
    total_pixels: int,
    cultivated_pixels: int,
) -> Dict[str, Any]:
    """
    Deterministic confidence from: ExG clarity, mask fragmentation, edge consistency, area plausibility.
    No arbitrary subtraction; all scores in [0,1]; confidence = weighted sum scaled to 0-100.
    Returns dict with: clarity_score, fragmentation_score, edge_consistency_score,
    area_plausibility_score, confidence (0-100), exg_variance (for transparency).
    """
    # 1) Vegetation clarity: ExG variance consistency (lower variance in cultivated region = higher score)
    b, g, r = cv2.split(img.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_min, exg_max = exg.min(), exg.max()
    exg_range = exg_max - exg_min
    if exg_range > 0:
        exg_norm = (exg - exg_min) / exg_range * 255.0
    else:
        exg_norm = np.zeros_like(exg)
    cultivated_flat = exg_norm[cultivated_mask > 0]
    if cultivated_flat.size == 0:
        exg_variance = 0.0
        clarity_score = 1.0
    else:
        exg_variance = float(np.var(cultivated_flat))
        clarity_score = max(0.0, min(1.0, 1.0 - (exg_variance / EXG_VARIANCE_SCALE)))

    # 2) Mask fragmentation: number_of_contours normalized (fewer = higher score)
    fragmentation_score = max(0.0, min(1.0, 1.0 - (contour_count / MAX_CONTOURS_FRAGMENTATION)))

    # 3) Edge consistency: ratio of cultivated_mask edges to total edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    total_edges = int(np.count_nonzero(edges))
    cultivated_edges = int(np.count_nonzero((edges > 0) & (cultivated_mask > 0)))
    if total_edges == 0:
        edge_consistency_score = 1.0
    else:
        edge_consistency_score = max(0.0, min(1.0, cultivated_edges / total_edges))

    # 4) Area plausibility: 10%-80% cultivated → higher confidence
    coverage = cultivated_pixels / total_pixels if total_pixels else 0.0
    if coverage < AREA_PLAUSIBLE_LOW:
        area_plausibility_score = coverage / AREA_PLAUSIBLE_LOW
    elif coverage > AREA_PLAUSIBLE_HIGH:
        area_plausibility_score = (1.0 - coverage) / (1.0 - AREA_PLAUSIBLE_HIGH)
    else:
        area_plausibility_score = 1.0
    area_plausibility_score = max(0.0, min(1.0, area_plausibility_score))

    # Combine: 0.3 * clarity + 0.25 * fragmentation + 0.25 * edge_consistency + 0.2 * area_plausibility
    confidence_raw = (
        0.3 * clarity_score
        + 0.25 * fragmentation_score
        + 0.25 * edge_consistency_score
        + 0.2 * area_plausibility_score
    )
    confidence = round(max(0.0, min(100.0, confidence_raw * 100.0)), 2)

    return {
        "clarity_score": round(clarity_score, 4),
        "fragmentation_score": round(fragmentation_score, 4),
        "edge_consistency_score": round(edge_consistency_score, 4),
        "area_plausibility_score": round(area_plausibility_score, 4),
        "confidence": confidence,
        "exg_variance": round(exg_variance, 2),
    }


def polygon_to_image_mask(
    polygon_lat_lng: List[List[float]], height: int, width: int
) -> np.ndarray:
    """
    Map polygon (list of [lat, lng]) to image coordinate space.
    Assume polygon bbox spans full image; map proportionally.
    Returns binary mask (255 inside polygon, 0 outside).
    """
    if not polygon_lat_lng or len(polygon_lat_lng) < 3:
        raise ValueError("Polygon must have at least 3 points")
    
    arr = np.array(polygon_lat_lng, dtype=np.float64)
    lats = arr[:, 0]
    lngs = arr[:, 1]
    min_lat, max_lat = lats.min(), lats.max()
    min_lng, max_lng = lngs.min(), lngs.max()
    
    if max_lat - min_lat <= 1e-9:
        max_lat = min_lat + 1e-9
    if max_lng - min_lng <= 1e-9:
        max_lng = min_lng + 1e-9
    
    # Map lat/lng to image coords: x = lng, y = from top (lat max = 0)
    xs = (lngs - min_lng) / (max_lng - min_lng) * (width - 1)
    ys = (1.0 - (lats - min_lat) / (max_lat - min_lat)) * (height - 1)
    pts = np.column_stack([xs, ys]).astype(np.int32)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def clip_cultivated_mask_by_boundary(
    cultivated_mask: np.ndarray,
    polygon_lat_lng: List[List[float]],
) -> Tuple[np.ndarray, int, int]:
    """
    Clip cultivated mask by polygon boundary (lat/lng mapped to image).
    Returns (clipped_mask, cultivated_pixels_inside, total_pixels_inside).
    """
    h, w = cultivated_mask.shape[:2]
    polygon_mask = polygon_to_image_mask(polygon_lat_lng, h, w)
    total_inside = int(cv2.countNonZero(polygon_mask))
    if total_inside == 0:
        return cultivated_mask, int(cv2.countNonZero(cultivated_mask)), h * w
    
    clipped = cv2.bitwise_and(cultivated_mask, polygon_mask)
    cultivated_inside = int(cv2.countNonZero(clipped))
    return clipped, cultivated_inside, total_inside


def get_landuse_mask_image(
    img: np.ndarray,
    vegetation_mask: np.ndarray,
    farmable_mask: np.ndarray,
) -> np.ndarray:
    """
    Land use: vegetation (green), bare soil (tan), water (blue), roads (gray), buildings (brown).
    """
    h, w = img.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = (35, 35, 35)  # dark background
    water_mask = detect_water(img)
    road_mask = detect_roads(img)
    building_mask = detect_buildings(img, vegetation_mask)
    bare_only = (farmable_mask > 0) & (vegetation_mask == 0)
    building_bgr = (42, 42, 139)   # brown in BGR
    road_bgr = (100, 100, 100)     # gray
    vegetation_bgr = (0, 200, 0)   # green
    bare_soil_bgr = (42, 165, 200) # tan in BGR
    water_bgr = (255, 120, 0)      # blue in BGR
    out[building_mask > 0] = building_bgr
    out[road_mask > 0] = road_bgr
    out[bare_only] = bare_soil_bgr
    out[vegetation_mask > 0] = vegetation_bgr
    out[water_mask > 0] = water_bgr
    return out


def get_landuse_mask_png(
    img: np.ndarray,
    vegetation_mask: np.ndarray,
    farmable_mask: np.ndarray,
) -> bytes:
    """Encode land-use mask image as PNG bytes."""
    bgr = get_landuse_mask_image(img, vegetation_mask, farmable_mask)
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def get_overlay_mask(
    img: np.ndarray,
    vegetation_mask: np.ndarray,
    farmable_mask: np.ndarray,
) -> np.ndarray:
    """
    Overlay farmable land: tint ALL farmable pixels in green.
    This means both vegetated and non-green cultivable areas appear uniformly green.
    """
    overlay = img.copy().astype(np.float32)
    alpha = 0.35
    green_bgr = np.array([0, 180, 0], dtype=np.float32)
    overlay[farmable_mask > 0] = overlay[farmable_mask > 0] * (1 - alpha) + green_bgr * alpha
    return overlay.astype(np.uint8)


def get_polygons_from_cultivated_mask(cultivated_mask: np.ndarray) -> List[List[List[float]]]:
    """
    Extract polygon list from cultivated mask for JSON export.
    Uses findContours, approximates with approxPolyDP; coordinates in original image scale.
    Returns list of polygons, each polygon = list of [x, y] points.
    """
    contours, _ = cv2.findContours(cultivated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[List[float]]] = []
    min_area = 100  # skip tiny noise contours
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        epsilon = max(2.0, 0.005 * peri)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) < 3:
            continue
        coords = [[float(p[0][0]), float(p[0][1])] for p in approx]
        polygons.append(coords)
    return polygons


def overlay_polygons_on_image(img: np.ndarray, polygons: List[List[List[float]]]) -> np.ndarray:
    """Draw polygon outlines on image for visualization. Returns BGR image."""
    out = img.copy()
    green = (0, 255, 0)
    thickness = max(2, max(img.shape[:2]) // 400)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=green, thickness=thickness)
    return out


def get_binary_mask_png(binary_mask: np.ndarray) -> bytes:
    """Encode binary mask (0/255) as PNG bytes."""
    _, png = cv2.imencode(".png", binary_mask)
    return png.tobytes()


def get_overlay_mask_png(overlay_bgr: np.ndarray) -> bytes:
    """Encode overlay BGR image as PNG bytes."""
    _, png = cv2.imencode(".png", overlay_bgr)
    return png.tobytes()


def analyze_image(image_bytes: bytes) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, int, Tuple[int, int], Tuple[int, int]]:
    """
    Real OpenCV analysis: read image, ExG vegetation pipeline, farmable = vegetation + bare soil.
    Returns: (cultivated_percentage, stress_percentage, original_bgr_image, vegetation_mask,
             farmable_mask, removed_pixel_count, original_resolution, processing_resolution)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image: could not decode")

    vegetation_mask, vegetation_pixel_count, total_pixel_count, removed_pixel_count, original_resolution, processing_resolution = exg_vegetation_pipeline(img)
    cultivated_percentage = (vegetation_pixel_count / total_pixel_count) * 100.0 if total_pixel_count else 0.0

    farmable_mask = get_farmable_mask(img, vegetation_mask)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    yellow_pixels = int(cv2.countNonZero(yellow_mask))
    vegetation_pixels = vegetation_pixel_count + yellow_pixels
    if vegetation_pixels > 0:
        stress_percentage = (yellow_pixels / vegetation_pixels) * 100.0
    else:
        stress_percentage = 0.0

    cultivated_percentage = round(cultivated_percentage, 2)
    stress_percentage = round(stress_percentage, 2)
    return cultivated_percentage, stress_percentage, img, vegetation_mask, farmable_mask, removed_pixel_count, original_resolution, processing_resolution
