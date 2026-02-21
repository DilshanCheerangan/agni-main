"""
AGNI - Crop Recommendation Engine
Real rule-based crop suggestion using temperature, humidity, season, water availability,
soil (moisture, type, pH), and farmable space. Data from agronomic ranges and regional
practices (India/subtropical/temperate). No mock data; no external AI API required.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional

# --- Soil & environment: crop-specific preferences (crop name -> preferences) ---
# soil_types: clay, loam, sandy, silt, black_alluvial, red_laterite
# soil_ph: acidic (pH < 6.5), neutral (6.5–7.5), alkaline (> 7.5), any
# Crops not listed default to loam + any pH
CROP_SOIL_PREFERENCES: Dict[str, Dict[str, Any]] = {
    "Rice (Paddy)": {"soil_types": ["clay", "clay_loam", "silt", "black_alluvial"], "soil_ph": "acidic"},
    "Wheat": {"soil_types": ["loam", "clay_loam", "silt"], "soil_ph": "neutral"},
    "Maize (Corn)": {"soil_types": ["loam", "sandy_loam", "black_alluvial"], "soil_ph": "neutral"},
    "Barley": {"soil_types": ["loam", "sandy_loam", "silt"], "soil_ph": "neutral"},
    "Millet (Bajra)": {"soil_types": ["sandy", "sandy_loam", "loam"], "soil_ph": "any"},
    "Sorghum (Jowar)": {"soil_types": ["loam", "sandy_loam", "black_alluvial"], "soil_ph": "any"},
    "Chickpea (Chana)": {"soil_types": ["loam", "sandy_loam", "black_alluvial"], "soil_ph": "neutral"},
    "Lentil (Masoor)": {"soil_types": ["loam", "silt"], "soil_ph": "neutral"},
    "Groundnut (Peanut)": {"soil_types": ["sandy_loam", "loam", "sandy"], "soil_ph": "neutral"},
    "Potato": {"soil_types": ["loam", "sandy_loam"], "soil_ph": "acidic"},
    "Carrot": {"soil_types": ["sandy", "sandy_loam", "loam"], "soil_ph": "neutral"},
    "Sugarcane": {"soil_types": ["clay", "clay_loam", "black_alluvial"], "soil_ph": "neutral"},
    "Cotton": {"soil_types": ["black_alluvial", "loam", "clay_loam"], "soil_ph": "neutral"},
    "Jute": {"soil_types": ["alluvial", "clay", "silt"], "soil_ph": "neutral"},
    "Tea": {"soil_types": ["loam", "sandy_loam"], "soil_ph": "acidic"},
    "Coffee": {"soil_types": ["loam", "sandy_loam"], "soil_ph": "acidic"},
    "Turmeric": {"soil_types": ["loam", "clay_loam", "red_laterite"], "soil_ph": "acidic"},
    "Ginger": {"soil_types": ["loam", "sandy_loam", "red_laterite"], "soil_ph": "acidic"},
    "Mustard": {"soil_types": ["loam", "clay_loam", "silt"], "soil_ph": "neutral"},
    "Sunflower": {"soil_types": ["loam", "sandy_loam", "black_alluvial"], "soil_ph": "neutral"},
    "Sesame (Til)": {"soil_types": ["sandy_loam", "loam", "sandy"], "soil_ph": "any"},
}
# Normalize soil_type input to match keys (allow clay_loam, sandy_loam, etc.)
SOIL_TYPE_ALIASES = {
    "clay": "clay", "loam": "loam", "sandy": "sandy", "silt": "silt",
    "black": "black_alluvial", "black alluvial": "black_alluvial", "alluvial": "black_alluvial",
    "red": "red_laterite", "laterite": "red_laterite", "red laterite": "red_laterite",
    "clay_loam": "clay_loam", "sandy_loam": "sandy_loam",
}


# --- Season Detection ---

def get_season(latitude: float, month: int) -> str:
    """
    Determine agricultural season based on latitude and month.
    For tropical/subtropical regions (like India), uses Kharif/Rabi/Zaid system.
    For temperate regions, uses Spring/Summer/Fall/Winter.
    """
    is_southern = latitude < 0
    # Tropical zone: between -23.5 and 23.5 degrees
    is_tropical = -23.5 <= latitude <= 23.5
    # Subtropical: India-like regions
    is_subtropical = 8 <= abs(latitude) <= 35

    if is_subtropical and not is_southern:
        # Indian agricultural seasons
        if month in [6, 7, 8, 9, 10]:
            return "kharif"  # Monsoon season (June-October)
        elif month in [11, 12, 1, 2, 3]:
            return "rabi"    # Winter season (November-March)
        else:
            return "zaid"    # Summer season (April-May)
    elif is_tropical:
        if month in [5, 6, 7, 8, 9, 10]:
            return "wet"
        else:
            return "dry"
    else:
        # Temperate seasons (flip for southern hemisphere)
        if is_southern:
            month = (month + 6 - 1) % 12 + 1
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall"
        else:
            return "winter"


def get_season_display_name(season: str) -> str:
    """Human-readable season name."""
    names = {
        "kharif": "Kharif (Monsoon)",
        "rabi": "Rabi (Winter)",
        "zaid": "Zaid (Summer)",
        "wet": "Wet Season",
        "dry": "Dry Season",
        "spring": "Spring",
        "summer": "Summer",
        "fall": "Fall/Autumn",
        "winter": "Winter",
    }
    return names.get(season, season.title())


# --- Crop Database ---
# Each crop has: name, category, ideal temp range, ideal humidity range,
# water needs (low/medium/high), min space needed (% of land),
# suitable seasons, description, growing duration, emoji

CROP_DATABASE: List[Dict[str, Any]] = [
    # --- Cereals & Grains ---
    {
        "name": "Rice (Paddy)",
        "emoji": "🌾",
        "category": "Cereal",
        "temp_min": 20, "temp_max": 37,
        "humidity_min": 60, "humidity_max": 95,
        "water_need": "high",
        "min_space_pct": 15,
        "seasons": ["kharif", "wet", "summer"],
        "duration": "120-150 days",
        "description": "Staple crop, thrives in hot humid conditions with standing water."
    },
    {
        "name": "Wheat",
        "emoji": "🌾",
        "category": "Cereal",
        "temp_min": 10, "temp_max": 25,
        "humidity_min": 30, "humidity_max": 70,
        "water_need": "medium",
        "min_space_pct": 15,
        "seasons": ["rabi", "winter", "fall"],
        "duration": "120-150 days",
        "description": "Cool-season crop, needs well-drained soil with moderate water."
    },
    {
        "name": "Maize (Corn)",
        "emoji": "🌽",
        "category": "Cereal",
        "temp_min": 18, "temp_max": 35,
        "humidity_min": 40, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 10,
        "seasons": ["kharif", "zaid", "spring", "summer", "wet"],
        "duration": "90-120 days",
        "description": "Versatile crop, grows in warm weather with good drainage."
    },
    {
        "name": "Barley",
        "emoji": "🌾",
        "category": "Cereal",
        "temp_min": 5, "temp_max": 20,
        "humidity_min": 25, "humidity_max": 60,
        "water_need": "low",
        "min_space_pct": 10,
        "seasons": ["rabi", "winter", "fall"],
        "duration": "90-120 days",
        "description": "Hardy cool-season crop, drought-tolerant."
    },
    {
        "name": "Millet (Bajra)",
        "emoji": "🌾",
        "category": "Cereal",
        "temp_min": 25, "temp_max": 40,
        "humidity_min": 20, "humidity_max": 60,
        "water_need": "low",
        "min_space_pct": 10,
        "seasons": ["kharif", "summer", "dry", "wet"],
        "duration": "60-90 days",
        "description": "Drought-resistant, ideal for arid/semi-arid regions."
    },
    {
        "name": "Sorghum (Jowar)",
        "emoji": "🌾",
        "category": "Cereal",
        "temp_min": 25, "temp_max": 40,
        "humidity_min": 20, "humidity_max": 65,
        "water_need": "low",
        "min_space_pct": 10,
        "seasons": ["kharif", "rabi", "summer", "dry"],
        "duration": "90-120 days",
        "description": "Heat and drought tolerant, dual-purpose grain and fodder."
    },
    {
        "name": "Oats",
        "emoji": "🌾",
        "category": "Cereal",
        "temp_min": 5, "temp_max": 20,
        "humidity_min": 40, "humidity_max": 70,
        "water_need": "medium",
        "min_space_pct": 10,
        "seasons": ["rabi", "winter", "spring"],
        "duration": "100-130 days",
        "description": "Cool-season cereal, good for fodder and grain."
    },

    # --- Pulses & Legumes ---
    {
        "name": "Chickpea (Chana)",
        "emoji": "🫘",
        "category": "Pulse",
        "temp_min": 10, "temp_max": 30,
        "humidity_min": 30, "humidity_max": 60,
        "water_need": "low",
        "min_space_pct": 5,
        "seasons": ["rabi", "winter", "fall", "dry"],
        "duration": "90-120 days",
        "description": "Nitrogen-fixing legume, drought-tolerant, enriches soil."
    },
    {
        "name": "Lentil (Masoor)",
        "emoji": "🫘",
        "category": "Pulse",
        "temp_min": 10, "temp_max": 28,
        "humidity_min": 30, "humidity_max": 60,
        "water_need": "low",
        "min_space_pct": 5,
        "seasons": ["rabi", "winter", "fall"],
        "duration": "90-120 days",
        "description": "Cool-season pulse, minimal water needed."
    },
    {
        "name": "Pigeon Pea (Toor Dal)",
        "emoji": "🫘",
        "category": "Pulse",
        "temp_min": 20, "temp_max": 38,
        "humidity_min": 40, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 10,
        "seasons": ["kharif", "wet", "summer"],
        "duration": "150-270 days",
        "description": "Long-duration pulse, semi-arid tolerant."
    },
    {
        "name": "Mung Bean (Moong)",
        "emoji": "🫘",
        "category": "Pulse",
        "temp_min": 25, "temp_max": 38,
        "humidity_min": 40, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 5,
        "seasons": ["kharif", "zaid", "summer", "wet"],
        "duration": "60-75 days",
        "description": "Short-duration, fits between main crops."
    },
    {
        "name": "Black Gram (Urad)",
        "emoji": "🫘",
        "category": "Pulse",
        "temp_min": 25, "temp_max": 35,
        "humidity_min": 50, "humidity_max": 85,
        "water_need": "medium",
        "min_space_pct": 5,
        "seasons": ["kharif", "wet", "summer"],
        "duration": "70-90 days",
        "description": "Warm-season pulse, monsoon-friendly."
    },
    {
        "name": "Groundnut (Peanut)",
        "emoji": "🥜",
        "category": "Oilseed",
        "temp_min": 22, "temp_max": 35,
        "humidity_min": 40, "humidity_max": 75,
        "water_need": "medium",
        "min_space_pct": 10,
        "seasons": ["kharif", "zaid", "summer", "wet"],
        "duration": "100-130 days",
        "description": "Oilseed legume, enriches soil with nitrogen."
    },
    {
        "name": "Soybean",
        "emoji": "🫘",
        "category": "Oilseed",
        "temp_min": 20, "temp_max": 35,
        "humidity_min": 50, "humidity_max": 85,
        "water_need": "medium",
        "min_space_pct": 10,
        "seasons": ["kharif", "summer", "wet"],
        "duration": "90-120 days",
        "description": "High-protein oilseed, good for intercropping."
    },

    # --- Vegetables ---
    {
        "name": "Tomato",
        "emoji": "🍅",
        "category": "Vegetable",
        "temp_min": 18, "temp_max": 32,
        "humidity_min": 40, "humidity_max": 70,
        "water_need": "medium",
        "min_space_pct": 3,
        "seasons": ["rabi", "zaid", "spring", "summer", "fall", "dry"],
        "duration": "60-90 days",
        "description": "Year-round in warm climates, high market value."
    },
    {
        "name": "Potato",
        "emoji": "🥔",
        "category": "Vegetable",
        "temp_min": 10, "temp_max": 25,
        "humidity_min": 60, "humidity_max": 85,
        "water_need": "medium",
        "min_space_pct": 5,
        "seasons": ["rabi", "winter", "spring", "fall"],
        "duration": "75-120 days",
        "description": "Cool-season tuber, high yield per area."
    },
    {
        "name": "Onion",
        "emoji": "🧅",
        "category": "Vegetable",
        "temp_min": 12, "temp_max": 28,
        "humidity_min": 40, "humidity_max": 70,
        "water_need": "medium",
        "min_space_pct": 3,
        "seasons": ["rabi", "kharif", "winter", "fall", "spring"],
        "duration": "90-150 days",
        "description": "Essential kitchen staple, good storage life."
    },
    {
        "name": "Cauliflower",
        "emoji": "🥦",
        "category": "Vegetable",
        "temp_min": 10, "temp_max": 22,
        "humidity_min": 60, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 3,
        "seasons": ["rabi", "winter", "fall"],
        "duration": "60-90 days",
        "description": "Cool-season brassica, needs consistent moisture."
    },
    {
        "name": "Cabbage",
        "emoji": "🥬",
        "category": "Vegetable",
        "temp_min": 10, "temp_max": 24,
        "humidity_min": 60, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 3,
        "seasons": ["rabi", "winter", "fall", "spring"],
        "duration": "60-90 days",
        "description": "Hardy cool-season crop, high nutritional value."
    },
    {
        "name": "Spinach",
        "emoji": "🥬",
        "category": "Vegetable",
        "temp_min": 8, "temp_max": 22,
        "humidity_min": 50, "humidity_max": 75,
        "water_need": "medium",
        "min_space_pct": 2,
        "seasons": ["rabi", "winter", "fall", "spring"],
        "duration": "30-45 days",
        "description": "Fast-growing leafy green, multiple harvests possible."
    },
    {
        "name": "Okra (Bhindi)",
        "emoji": "🌿",
        "category": "Vegetable",
        "temp_min": 22, "temp_max": 38,
        "humidity_min": 50, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 3,
        "seasons": ["kharif", "zaid", "summer", "wet"],
        "duration": "45-65 days",
        "description": "Warm-season vegetable, continuous harvesting."
    },
    {
        "name": "Brinjal (Eggplant)",
        "emoji": "🍆",
        "category": "Vegetable",
        "temp_min": 20, "temp_max": 35,
        "humidity_min": 50, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 3,
        "seasons": ["kharif", "rabi", "summer", "wet", "spring"],
        "duration": "60-80 days",
        "description": "Year-round in tropical climates, versatile crop."
    },
    {
        "name": "Chilli Pepper",
        "emoji": "🌶️",
        "category": "Vegetable",
        "temp_min": 20, "temp_max": 35,
        "humidity_min": 40, "humidity_max": 70,
        "water_need": "medium",
        "min_space_pct": 3,
        "seasons": ["kharif", "rabi", "summer", "wet", "spring", "fall"],
        "duration": "60-120 days",
        "description": "High-value spice crop, good market demand."
    },
    {
        "name": "Cucumber",
        "emoji": "🥒",
        "category": "Vegetable",
        "temp_min": 18, "temp_max": 35,
        "humidity_min": 50, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 3,
        "seasons": ["kharif", "zaid", "summer", "wet", "spring"],
        "duration": "40-60 days",
        "description": "Fast-growing vine crop, needs adequate water."
    },
    {
        "name": "Pumpkin",
        "emoji": "🎃",
        "category": "Vegetable",
        "temp_min": 18, "temp_max": 35,
        "humidity_min": 50, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 5,
        "seasons": ["kharif", "zaid", "summer", "wet", "spring"],
        "duration": "90-120 days",
        "description": "Spreading vine, needs space but low maintenance."
    },
    {
        "name": "Carrot",
        "emoji": "🥕",
        "category": "Vegetable",
        "temp_min": 8, "temp_max": 24,
        "humidity_min": 50, "humidity_max": 75,
        "water_need": "medium",
        "min_space_pct": 2,
        "seasons": ["rabi", "winter", "fall", "spring"],
        "duration": "70-90 days",
        "description": "Root vegetable, cool-season, deep loose soil preferred."
    },
    {
        "name": "Radish",
        "emoji": "🌿",
        "category": "Vegetable",
        "temp_min": 10, "temp_max": 25,
        "humidity_min": 40, "humidity_max": 70,
        "water_need": "low",
        "min_space_pct": 2,
        "seasons": ["rabi", "winter", "fall", "spring"],
        "duration": "25-40 days",
        "description": "Very fast-growing root crop, good for intercropping."
    },
    {
        "name": "Bitter Gourd (Karela)",
        "emoji": "🌿",
        "category": "Vegetable",
        "temp_min": 24, "temp_max": 38,
        "humidity_min": 50, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 3,
        "seasons": ["kharif", "zaid", "summer", "wet"],
        "duration": "55-75 days",
        "description": "Medicinal value, warm-season vine crop."
    },
    {
        "name": "Bottle Gourd (Lauki)",
        "emoji": "🌿",
        "category": "Vegetable",
        "temp_min": 22, "temp_max": 38,
        "humidity_min": 50, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 5,
        "seasons": ["kharif", "zaid", "summer", "wet"],
        "duration": "60-80 days",
        "description": "Popular climbing vegetable, needs support structure."
    },
    {
        "name": "Green Peas",
        "emoji": "🫛",
        "category": "Vegetable",
        "temp_min": 8, "temp_max": 22,
        "humidity_min": 50, "humidity_max": 75,
        "water_need": "medium",
        "min_space_pct": 3,
        "seasons": ["rabi", "winter", "fall", "spring"],
        "duration": "60-90 days",
        "description": "Cool-season leguminous vegetable, nitrogen-fixing."
    },
    {
        "name": "Sweet Potato",
        "emoji": "🍠",
        "category": "Vegetable",
        "temp_min": 20, "temp_max": 35,
        "humidity_min": 50, "humidity_max": 85,
        "water_need": "medium",
        "min_space_pct": 5,
        "seasons": ["kharif", "summer", "wet", "spring"],
        "duration": "90-120 days",
        "description": "Nutritious tuber, drought-tolerant once established."
    },
    {
        "name": "Garlic",
        "emoji": "🧄",
        "category": "Vegetable",
        "temp_min": 10, "temp_max": 25,
        "humidity_min": 40, "humidity_max": 65,
        "water_need": "low",
        "min_space_pct": 2,
        "seasons": ["rabi", "winter", "fall"],
        "duration": "120-150 days",
        "description": "High-value spice, good storage and market demand."
    },
    {
        "name": "Lettuce",
        "emoji": "🥬",
        "category": "Vegetable",
        "temp_min": 8, "temp_max": 20,
        "humidity_min": 50, "humidity_max": 75,
        "water_need": "medium",
        "min_space_pct": 2,
        "seasons": ["rabi", "winter", "fall", "spring"],
        "duration": "30-60 days",
        "description": "Quick-growing salad green, premium pricing."
    },

    # --- Fruits ---
    {
        "name": "Watermelon",
        "emoji": "🍉",
        "category": "Fruit",
        "temp_min": 25, "temp_max": 40,
        "humidity_min": 40, "humidity_max": 70,
        "water_need": "high",
        "min_space_pct": 10,
        "seasons": ["zaid", "summer", "dry"],
        "duration": "70-90 days",
        "description": "Hot-season fruit, needs space and ample water."
    },
    {
        "name": "Muskmelon",
        "emoji": "🍈",
        "category": "Fruit",
        "temp_min": 24, "temp_max": 38,
        "humidity_min": 35, "humidity_max": 65,
        "water_need": "medium",
        "min_space_pct": 8,
        "seasons": ["zaid", "summer", "dry"],
        "duration": "60-90 days",
        "description": "Summer fruit, prefers dry hot conditions."
    },
    {
        "name": "Banana",
        "emoji": "🍌",
        "category": "Fruit",
        "temp_min": 22, "temp_max": 38,
        "humidity_min": 60, "humidity_max": 90,
        "water_need": "high",
        "min_space_pct": 10,
        "seasons": ["kharif", "wet", "summer", "spring"],
        "duration": "270-365 days",
        "description": "Tropical fruit, year-round in warm humid areas."
    },
    {
        "name": "Papaya",
        "emoji": "🌿",
        "category": "Fruit",
        "temp_min": 22, "temp_max": 38,
        "humidity_min": 50, "humidity_max": 85,
        "water_need": "medium",
        "min_space_pct": 5,
        "seasons": ["kharif", "wet", "summer", "spring"],
        "duration": "240-330 days",
        "description": "Tropical fruit, fast-growing, year-round fruiting."
    },
    {
        "name": "Strawberry",
        "emoji": "🍓",
        "category": "Fruit",
        "temp_min": 10, "temp_max": 25,
        "humidity_min": 50, "humidity_max": 75,
        "water_need": "medium",
        "min_space_pct": 2,
        "seasons": ["rabi", "winter", "fall", "spring"],
        "duration": "60-90 days",
        "description": "High-value fruit, cool-season, premium market."
    },

    # --- Cash Crops ---
    {
        "name": "Sugarcane",
        "emoji": "🌿",
        "category": "Cash Crop",
        "temp_min": 20, "temp_max": 40,
        "humidity_min": 60, "humidity_max": 90,
        "water_need": "high",
        "min_space_pct": 15,
        "seasons": ["kharif", "spring", "wet", "summer"],
        "duration": "300-365 days",
        "description": "Long-duration cash crop, needs lots of water."
    },
    {
        "name": "Cotton",
        "emoji": "🌿",
        "category": "Cash Crop",
        "temp_min": 22, "temp_max": 38,
        "humidity_min": 40, "humidity_max": 70,
        "water_need": "medium",
        "min_space_pct": 15,
        "seasons": ["kharif", "summer", "wet"],
        "duration": "150-180 days",
        "description": "Major fiber crop, warm-season."
    },
    {
        "name": "Jute",
        "emoji": "🌿",
        "category": "Cash Crop",
        "temp_min": 24, "temp_max": 38,
        "humidity_min": 70, "humidity_max": 95,
        "water_need": "high",
        "min_space_pct": 10,
        "seasons": ["kharif", "wet", "summer"],
        "duration": "100-150 days",
        "description": "Fiber crop, needs hot humid conditions."
    },
    {
        "name": "Tobacco",
        "emoji": "🍂",
        "category": "Cash Crop",
        "temp_min": 15, "temp_max": 35,
        "humidity_min": 40, "humidity_max": 70,
        "water_need": "medium",
        "min_space_pct": 5,
        "seasons": ["rabi", "winter", "spring"],
        "duration": "90-120 days",
        "description": "Cash crop, needs well-drained fertile soil."
    },
    {
        "name": "Tea",
        "emoji": "🍵",
        "category": "Cash Crop",
        "temp_min": 15, "temp_max": 30,
        "humidity_min": 70, "humidity_max": 95,
        "water_need": "high",
        "min_space_pct": 10,
        "seasons": ["kharif", "spring", "summer", "wet"],
        "duration": "Perennial",
        "description": "Plantation crop, needs hilly terrain with high rainfall."
    },
    {
        "name": "Coffee",
        "emoji": "☕",
        "category": "Cash Crop",
        "temp_min": 15, "temp_max": 28,
        "humidity_min": 60, "humidity_max": 90,
        "water_need": "medium",
        "min_space_pct": 10,
        "seasons": ["kharif", "wet", "spring"],
        "duration": "Perennial",
        "description": "Shade-grown plantation crop, high altitude preferred."
    },

    # --- Spices ---
    {
        "name": "Turmeric",
        "emoji": "🌿",
        "category": "Spice",
        "temp_min": 20, "temp_max": 35,
        "humidity_min": 60, "humidity_max": 90,
        "water_need": "high",
        "min_space_pct": 3,
        "seasons": ["kharif", "wet", "summer"],
        "duration": "210-270 days",
        "description": "Rhizome spice, needs warm humid conditions."
    },
    {
        "name": "Ginger",
        "emoji": "🌿",
        "category": "Spice",
        "temp_min": 20, "temp_max": 35,
        "humidity_min": 60, "humidity_max": 90,
        "water_need": "high",
        "min_space_pct": 3,
        "seasons": ["kharif", "wet", "summer"],
        "duration": "210-270 days",
        "description": "Underground rhizome, shade-tolerant."
    },
    {
        "name": "Coriander",
        "emoji": "🌿",
        "category": "Spice",
        "temp_min": 12, "temp_max": 28,
        "humidity_min": 40, "humidity_max": 65,
        "water_need": "low",
        "min_space_pct": 2,
        "seasons": ["rabi", "winter", "fall", "spring"],
        "duration": "40-60 days",
        "description": "Fast-growing herb, dual use (leaves + seeds)."
    },
    {
        "name": "Cumin (Jeera)",
        "emoji": "🌿",
        "category": "Spice",
        "temp_min": 15, "temp_max": 30,
        "humidity_min": 30, "humidity_max": 55,
        "water_need": "low",
        "min_space_pct": 2,
        "seasons": ["rabi", "winter", "dry"],
        "duration": "100-120 days",
        "description": "High-value spice, dry conditions preferred."
    },
    {
        "name": "Fenugreek (Methi)",
        "emoji": "🌿",
        "category": "Spice",
        "temp_min": 10, "temp_max": 28,
        "humidity_min": 40, "humidity_max": 65,
        "water_need": "low",
        "min_space_pct": 2,
        "seasons": ["rabi", "winter", "fall", "spring"],
        "duration": "30-60 days",
        "description": "Leafy herb + spice, very fast growing."
    },

    # --- Oilseeds ---
    {
        "name": "Mustard",
        "emoji": "🌼",
        "category": "Oilseed",
        "temp_min": 10, "temp_max": 25,
        "humidity_min": 30, "humidity_max": 60,
        "water_need": "low",
        "min_space_pct": 5,
        "seasons": ["rabi", "winter", "fall"],
        "duration": "90-120 days",
        "description": "Cool-season oilseed, drought-tolerant."
    },
    {
        "name": "Sunflower",
        "emoji": "🌻",
        "category": "Oilseed",
        "temp_min": 18, "temp_max": 35,
        "humidity_min": 35, "humidity_max": 65,
        "water_need": "medium",
        "min_space_pct": 5,
        "seasons": ["kharif", "rabi", "spring", "summer"],
        "duration": "80-100 days",
        "description": "Versatile oilseed, adapts to many climates."
    },
    {
        "name": "Sesame (Til)",
        "emoji": "🌿",
        "category": "Oilseed",
        "temp_min": 25, "temp_max": 40,
        "humidity_min": 30, "humidity_max": 60,
        "water_need": "low",
        "min_space_pct": 5,
        "seasons": ["kharif", "summer", "dry"],
        "duration": "80-100 days",
        "description": "Hot-season oilseed, very drought-tolerant."
    },
    {
        "name": "Flaxseed (Alsi)",
        "emoji": "🌿",
        "category": "Oilseed",
        "temp_min": 10, "temp_max": 25,
        "humidity_min": 40, "humidity_max": 65,
        "water_need": "low",
        "min_space_pct": 3,
        "seasons": ["rabi", "winter", "fall"],
        "duration": "90-120 days",
        "description": "Cool-season oilseed, health-food demand growing."
    },

    # --- Flowers ---
    {
        "name": "Marigold",
        "emoji": "🌼",
        "category": "Flower",
        "temp_min": 15, "temp_max": 35,
        "humidity_min": 40, "humidity_max": 70,
        "water_need": "low",
        "min_space_pct": 2,
        "seasons": ["kharif", "rabi", "zaid", "spring", "summer", "fall", "winter"],
        "duration": "45-60 days",
        "description": "High-demand flower, pest-repellent companion plant."
    },
    {
        "name": "Rose",
        "emoji": "🌹",
        "category": "Flower",
        "temp_min": 12, "temp_max": 30,
        "humidity_min": 50, "humidity_max": 75,
        "water_need": "medium",
        "min_space_pct": 2,
        "seasons": ["rabi", "winter", "spring", "fall"],
        "duration": "Perennial",
        "description": "Premium flower, year-round market demand."
    },
    {
        "name": "Jasmine",
        "emoji": "🌸",
        "category": "Flower",
        "temp_min": 18, "temp_max": 35,
        "humidity_min": 50, "humidity_max": 80,
        "water_need": "medium",
        "min_space_pct": 2,
        "seasons": ["kharif", "summer", "spring", "wet"],
        "duration": "Perennial",
        "description": "Fragrant flower, temple and perfume market demand."
    },
]


def _water_need_score(water_need: str, humidity: float, water_pct: float) -> float:
    """
    Score how well the water availability matches crop needs.
    water_pct: percentage of water detected in the image.
    Returns 0.0 to 1.0.
    """
    water_levels = {"low": 0, "medium": 1, "high": 2}
    crop_level = water_levels.get(water_need, 1)

    # Estimate available water from humidity + detected water bodies
    available = 0
    if humidity >= 70 or water_pct >= 10:
        available = 2  # High water
    elif humidity >= 45 or water_pct >= 3:
        available = 1  # Medium water
    else:
        available = 0  # Low water

    diff = abs(crop_level - available)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.6
    else:
        return 0.2


def _soil_moisture_level(value: Optional[Any]) -> Optional[int]:
    """Map user soil moisture input to 0=low, 1=medium, 2=high. None if not provided."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v < 30:
            return 0
        if v < 60:
            return 1
        return 2
    s = str(value).strip().lower()
    if s in ("low", "dry", "l"):
        return 0
    if s in ("medium", "moderate", "m", "mid"):
        return 1
    if s in ("high", "wet", "h", "moist"):
        return 2
    return None


def _soil_type_score(crop_name: str, user_soil_type: str) -> float:
    """Score 0.0–1.0: how well user soil type matches crop preference."""
    prefs = CROP_SOIL_PREFERENCES.get(crop_name)
    if not prefs:
        preferred = ["loam"]
    else:
        preferred = prefs.get("soil_types", ["loam"])
    raw = user_soil_type.strip().lower().replace(" ", "_")
    user_normalized = SOIL_TYPE_ALIASES.get(raw, raw)
    if user_normalized in preferred:
        return 1.0
    if user_normalized == "black_alluvial" and "alluvial" in preferred:
        return 1.0
    for p in preferred:
        if user_normalized in p or p in user_normalized:
            return 0.8
    return 0.3


def _soil_ph_score(crop_name: str, user_ph: str) -> float:
    """Score 0.0–1.0: how well user soil pH matches crop preference."""
    prefs = CROP_SOIL_PREFERENCES.get(crop_name)
    crop_ph = (prefs.get("soil_ph", "any") if prefs else "any")
    if crop_ph == "any":
        return 1.0
    u = str(user_ph).strip().lower()
    if u not in ("acidic", "neutral", "alkaline"):
        return 0.7
    if u == crop_ph:
        return 1.0
    if (crop_ph == "neutral" and u in ("acidic", "alkaline")) or (u == "neutral" and crop_ph != "neutral"):
        return 0.6
    return 0.3


def _rainfall_level(value: Optional[str]) -> Optional[int]:
    """Map rainfall_expected to 0=low, 1=medium, 2=high."""
    if not value:
        return None
    s = str(value).strip().lower()
    if s in ("low", "dry", "scanty", "l"):
        return 0
    if s in ("medium", "moderate", "normal", "m"):
        return 1
    if s in ("high", "heavy", "good", "h"):
        return 2
    return None


def recommend_crops(
    temperature: float,
    humidity: float,
    latitude: float,
    month: int,
    cultivated_pct: float = 50.0,
    water_pct: float = 5.0,
    farmable_space_pct: float = 50.0,
    soil_moisture: Optional[Any] = None,
    soil_type: Optional[str] = None,
    soil_ph: Optional[str] = None,
    rainfall_expected: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Recommend crops based on environmental and soil conditions (real agronomic data).

    Args:
        temperature: Current temperature in Celsius
        humidity: Current humidity percentage
        latitude: Location latitude
        month: Current month (1-12)
        cultivated_pct: Percentage already cultivated
        water_pct: Percentage of water detected
        farmable_space_pct: Percentage of land available for farming
        soil_moisture: Optional - "low"/"medium"/"high" or 0-100 (region's typical soil moisture)
        soil_type: Optional - e.g. "clay", "loam", "sandy", "black alluvial", "red laterite"
        soil_ph: Optional - "acidic", "neutral", "alkaline"
        rainfall_expected: Optional - "low", "medium", "high" (expected rainfall for the season)

    Returns:
        Dict with season info, recommended crops (sorted by match score), and summary.
    """
    season = get_season(latitude, month)
    season_display = get_season_display_name(season)
    available_space = max(0.0, farmable_space_pct - cultivated_pct)

    use_soil = soil_moisture is not None or (soil_type and soil_type.strip()) or (soil_ph and soil_ph.strip())
    use_rainfall = rainfall_expected and str(rainfall_expected).strip()
    # Weights: base 40+25+20+15; with soil 35+22+18+12+10; with rainfall extra +5 from temp/season
    if use_soil and use_rainfall:
        w_temp, w_season, w_water, w_space, w_soil, w_rain = 0.32, 0.18, 0.18, 0.12, 0.10, 0.10
    elif use_soil:
        w_temp, w_season, w_water, w_space, w_soil, w_rain = 0.35, 0.20, 0.18, 0.12, 0.15, 0.0
    elif use_rainfall:
        w_temp, w_season, w_water, w_space, w_soil, w_rain = 0.35, 0.20, 0.18, 0.12, 0.0, 0.15
    else:
        w_temp, w_season, w_water, w_space, w_soil, w_rain = 0.40, 0.25, 0.20, 0.15, 0.0, 0.0

    user_moisture_level = _soil_moisture_level(soil_moisture)
    user_rainfall_level = _rainfall_level(rainfall_expected) if use_rainfall else None

    scored_crops = []

    for crop in CROP_DATABASE:
        score = 0.0
        reasons = []
        warnings = []

        # 1. Temperature match
        temp_mid = (crop["temp_min"] + crop["temp_max"]) / 2
        temp_range = crop["temp_max"] - crop["temp_min"]
        if crop["temp_min"] <= temperature <= crop["temp_max"]:
            temp_closeness = 1.0 - abs(temperature - temp_mid) / (temp_range / 2)
            temp_score = 0.6 + 0.4 * temp_closeness
            reasons.append(f"Temperature {temperature}°C is ideal ({crop['temp_min']}–{crop['temp_max']}°C)")
        elif temperature < crop["temp_min"]:
            gap = crop["temp_min"] - temperature
            temp_score = 0.3 if gap <= 5 else 0.0
            warnings.append(f"Slightly cold" if gap <= 5 else f"Too cold ({temperature}°C, needs {crop['temp_min']}°C+)")
        else:
            gap = temperature - crop["temp_max"]
            temp_score = 0.3 if gap <= 5 else 0.0
            warnings.append(f"Slightly hot" if gap <= 5 else f"Too hot ({temperature}°C, max {crop['temp_max']}°C)")
        score += temp_score * w_temp

        # 2. Season match
        if season in crop["seasons"]:
            score += 1.0 * w_season
            reasons.append(f"Suitable for {season_display}")
        else:
            score += 0.2 * w_season
            warnings.append(f"Not typical for {season_display}")

        # 3. Water match
        water_score = _water_need_score(crop["water_need"], humidity, water_pct)
        score += water_score * w_water
        if water_score >= 0.8:
            reasons.append(f"Water availability matches ({crop['water_need']} need)")
        elif water_score <= 0.3:
            warnings.append(f"Water mismatch (crop needs {crop['water_need']})")

        # 4. Space match
        if available_space >= crop["min_space_pct"]:
            score += 1.0 * w_space
            reasons.append(f"Enough space ({available_space:.0f}% available, needs {crop['min_space_pct']}%)")
        elif available_space >= crop["min_space_pct"] * 0.5:
            score += 0.5 * w_space
            warnings.append(f"Limited space ({available_space:.0f}% vs {crop['min_space_pct']}% ideal)")
        else:
            warnings.append(f"Insufficient space ({available_space:.0f}%)")

        # 5. Soil match (moisture, type, pH) when provided
        if use_soil:
            soil_scores = []
            if user_moisture_level is not None:
                crop_water = {"low": 0, "medium": 1, "high": 2}.get(crop["water_need"], 1)
                diff = abs(user_moisture_level - crop_water)
                sm = 1.0 if diff == 0 else (0.6 if diff == 1 else 0.2)
                soil_scores.append(sm)
                if sm >= 0.8:
                    reasons.append("Soil moisture suits crop")
                elif sm <= 0.3:
                    warnings.append("Soil moisture may not suit crop")
            if soil_type and soil_type.strip():
                st = _soil_type_score(crop["name"], soil_type)
                soil_scores.append(st)
                if st >= 0.8:
                    reasons.append("Soil type suitable for region")
            if soil_ph and soil_ph.strip():
                sp = _soil_ph_score(crop["name"], soil_ph)
                soil_scores.append(sp)
                if sp >= 0.8:
                    reasons.append("Soil pH suitable")
            soil_score = sum(soil_scores) / len(soil_scores) if soil_scores else 0.7
            score += soil_score * w_soil

        # 6. Rainfall expected (when provided)
        if use_rainfall and user_rainfall_level is not None:
            crop_rain = {"low": 0, "medium": 1, "high": 2}.get(crop["water_need"], 1)
            diff = abs(user_rainfall_level - crop_rain)
            rain_score = 1.0 if diff == 0 else (0.6 if diff == 1 else 0.2)
            score += rain_score * w_rain
            if rain_score >= 0.8:
                reasons.append("Expected rainfall matches crop need")

        score = round(min(1.0, score), 3)

        scored_crops.append({
            "name": crop["name"],
            "emoji": crop["emoji"],
            "category": crop["category"],
            "score": score,
            "match_pct": int(round(score * 100)),
            "duration": crop["duration"],
            "water_need": crop["water_need"],
            "description": crop["description"],
            "reasons": reasons,
            "warnings": warnings,
            "temp_range": f"{crop['temp_min']}–{crop['temp_max']}°C",
        })

    # Sort by score descending
    scored_crops.sort(key=lambda c: c["score"], reverse=True)

    # Categorize
    highly_recommended = [c for c in scored_crops if c["score"] >= 0.7]
    moderately_suitable = [c for c in scored_crops if 0.45 <= c["score"] < 0.7]
    not_recommended = [c for c in scored_crops if c["score"] < 0.45]

    # Summary
    top_count = len(highly_recommended)
    if top_count >= 5:
        summary = f"Excellent conditions! {top_count} crops are highly suitable for your land."
    elif top_count >= 2:
        summary = f"Good conditions. {top_count} crops are well-matched for your environment."
    elif top_count >= 1:
        summary = f"Limited options, but {top_count} crop(s) can grow well in current conditions."
    else:
        summary = "Challenging conditions. Consider protected cultivation (greenhouse/polyhouse)."

    # One primary recommendation + sub-suggestions (alternatives)
    primary = highly_recommended[0] if highly_recommended else (moderately_suitable[0] if moderately_suitable else None)
    if highly_recommended:
        sub_suggestions = highly_recommended[1:6] + moderately_suitable[:3]
    else:
        sub_suggestions = moderately_suitable[1:6] if moderately_suitable else []
    sub_suggestions = sub_suggestions[:8]  # cap at 8 sub-suggestions

    out = {
        "season": season,
        "season_display": season_display,
        "temperature": temperature,
        "humidity": humidity,
        "latitude": latitude,
        "available_space_pct": round(available_space, 1),
        "water_detected_pct": round(water_pct, 1),
        "summary": summary,
        "primary_crop": primary,
        "sub_suggestions": sub_suggestions,
        "highly_recommended": highly_recommended[:10],
        "moderately_suitable": moderately_suitable[:10],
        "not_recommended_count": len(not_recommended),
        "total_crops_evaluated": len(CROP_DATABASE),
        "indicators_used": ["temperature", "humidity", "season", "water_availability", "space"],
    }
    if use_soil:
        if soil_moisture is not None:
            out["indicators_used"].append("soil_moisture")
        if soil_type and str(soil_type).strip():
            out["indicators_used"].append("soil_type")
        if soil_ph and str(soil_ph).strip():
            out["indicators_used"].append("soil_ph")
        out["soil_moisture"] = soil_moisture
        out["soil_type"] = soil_type
        out["soil_ph"] = soil_ph
    if use_rainfall:
        out["indicators_used"].append("rainfall_expected")
        out["rainfall_expected"] = rainfall_expected
    return out
