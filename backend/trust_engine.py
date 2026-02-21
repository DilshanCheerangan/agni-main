"""
AGNI - Real trust scoring and blur detection (OpenCV Laplacian).
No mock data; trust score and blur from actual metrics and image.
"""
import cv2
import numpy as np
from typing import Tuple

LAPLACIAN_BLUR_THRESHOLD = 100


def _detect_blur(image: np.ndarray) -> bool:
    """Blur detection: grayscale -> Laplacian variance. If variance < 100 then image is blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < LAPLACIAN_BLUR_THRESHOLD


def compute_trust_score(
    cultivated_percentage: float,
    stress_percentage: float,
    image: np.ndarray,
) -> Tuple[int, bool]:
    """
    Real trust score: start at 100, apply penalties, clamp 0-100.
    Real blur detection on image.
    Returns: (trust_score, image_blurry)
    """
    image_blurry = _detect_blur(image)

    trust_score = 100

    if cultivated_percentage < 40:
        trust_score -= 25
    if stress_percentage > 50:
        trust_score -= 30
    if image_blurry:
        trust_score -= 20

    trust_score = max(0, min(100, trust_score))
    return trust_score, image_blurry


def get_advisory(
    cultivated_percentage: float,
    stress_percentage: float,
    trust_score: int,
) -> str:
    """
    Real advisory from metrics and trust. No mock messages.
    """
    if cultivated_percentage > 70 and stress_percentage < 20:
        advisory = "Crop Healthy"
    elif 40 <= cultivated_percentage <= 70:
        advisory = "Irrigation Recommended"
    else:
        advisory = "Immediate Field Inspection Needed"

    if trust_score < 50:
        advisory += " (Low Data Confidence)"

    return advisory
