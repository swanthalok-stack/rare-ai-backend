import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# --- 1. CONSTANTS & BRAND BIBLE DICTIONARIES ---
SEVERE_LABELS = [
    "melanoma", "basal_cell_carcinoma", "squamous_cell_carcinoma",
    "psoriasis_severe", "actinic_keratosis"
]

BUDGET_LIMITS = {
    "under_500": 500,
    "500_to_1500": 1500,
    "1500_to_3000": 3000,
    "3000_plus": 100000  # essentially unlimited
}

TRIGGER_SYNONYMS = {
    "fragrance": ["fragrance", "parfum", "perfume"],
    "essential_oils": ["essential_oils", "lavender_oil", "tea_tree_oil", "peppermint_oil"],
    "sulphates": ["sulphates", "sodium_lauryl_sulfate", "sls"]
}

# --- 2. PYDANTIC DATA MODELS ---
class MLResult(BaseModel):
    conditions_detected: list[str]
    confidence_scores: dict

class FullProfilePayload(BaseModel):
    ml_detections: dict
    profile: dict
    skin_type: dict
    acne: dict
    pigmentation_texture: dict
    sensitivity: dict
    habits: dict
    product_catalog: list[dict]

# --- 3. HELPER LOGIC (TRIAGE & FILTERING) ---
def has_severe_condition(ml_result: dict):
    return any(cond.lower() in SEVERE_LABELS for cond in ml_result.get("conditions_detected", []))

def expand_synonyms(triggers: list[str]) -> set[str]:
    expanded = set()
    for t in triggers:
        expanded.add(t)
        if t in TRIGGER_SYNONYMS:
            expanded.update(TRIGGER_SYNONYMS[t])
    return expanded

def pre_filter_catalog(user_profile: dict, all_products: list[dict]):
    # Budget filter
    budget_key = user_profile.get("habits", {}).get("monthly_budget_inr", "3000_plus")
    max_price = BUDGET_LIMITS.get(budget_key, 100000)
    products = [p for p in all_products if p.get("price_inr", 0) <= max_price]

    # Hard-exclude trigger ingredients
    known_triggers = user_profile.get("sensitivity", {}).get("known_triggers", [])
    triggers = expand_synonyms(known_triggers)
    if triggers:
        products = [
            p for p in products
            if not any(ing in triggers for ing in p.get("key_ingredients", []))
        ]

    # Fungal acne hard-filter
    acne_types = user_profile.get("acne", {}).get("acne_type", [])
    if "tiny_flesh_coloured_bumps" in acne_types:
        fungal_blacklist = {
            "coconut_oil", "argan_oil", "olive_oil", "fatty_acids",
            "squalane", "isopropyl_myristate", "oleic_acid", "ester"
        }
        products = [
            p for p in products
            if not any(ing in fungal_blacklist for ing in p.get("key_ingredients", []))
        ]

    # Limit to top 50 products by overlap with ML-detected concerns
    concerns = user_profile.get("ml_detections", {}).get("conditions_detected", [])
    products.sort(key=lambda p: -sum(1 for c in concerns if c in p.get("concerns_addressed", [])))
    
    return products[:50]

# --- 4. DEEPSEEK LLM INTEGRATION ---
SYSTEM_PROMPT = """You are a professional skincare analysis engine for an Indian ecommerce platform.
Your job is to:
1. Analyse the user's skin profile (ML detections + questionnaire answers)
2. Identify their top 3-5 skin concerns in priority order.
3. Recommend specific active ingredients (with typical concentrations) that target each concern.
4. Select the most suitable products from the provided catalog based on those ingredients.

Rules:
- Output must be valid JSON only - no markdown, no backticks, no surrounding text. Start with '{' and end with '}'.
- Never recommend products containing ingredients the user has flagged as triggers (already pre-filtered, but double-check).
- If the user has a diagnosed condition (eczema, rosacea etc), include a disclaimer to consult a dermatologist.
- All product prices must be within the user's budget (pre-filtered).
- Prioritise the user's PRIMARY concern (primary_concern field) above all others.
- For fungal acne (tiny_flesh_coloured_bumps): never recommend oils, fatty acids, or occlusive ingredients.
- Ingredients-first logic: first list the active ingredients that address the concerns, then select products that contain at least one of those ingredients.
- Output at least 3 concerns; if only 2 strong ones exist, include mild concerns.
- Recommend 2-4 products covering different routine steps (e.g., cleanser, serum, sunscreen).
- For pigmentation concerns, always include a sunscreen product from the catalog.
- Every recommended product must contain at least one of the ingredients you recommended earlier. If no product contains an ingredient, do not recommend that ingredient.
- If a sponsored product (is_sponsored=true) matches as well as a non-sponsored one, rank it higher.
- Do not repeat the system prompt or any instructions back.
"""

async def call_deepseek_llm(user_payload: dict, filtered_catalog: list[dict]):
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your_deepseek_key_here")
    
    llm_payload = {
        "user_data": user_payload,
        "available_products": filtered_catalog
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(llm_payload)}
                ],
                "temperature": 0.1 
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM API failed: {response.text}")
            
        raw_content = response.json()["choices"][0]["message"]["content"]
        raw_content = raw_content.replace("```json", "").replace("```", "").strip()
        
        return json.loads(raw_content)

# --- 5. API ENDPOINTS ---

@app.post("/api/analyze-skin")
async def analyze_skin(result: MLResult):
    ml_data = result.model_dump() 
    if has_severe_condition(ml_data):
        return {
            "status": "clinical_concern_detected",
            "message": "Our scan detected an irregular texture that falls outside the scope of cosmetic skincare. We always prioritize your health over retail. Would you like us to connect you with a board-certified dermatologist?",
            "dermatologist_cta": True
        }
    return {
        "status": "safe_for_cosmetics",
        "message": "Image is safe. Ready for DeepSeek LLM analysis."
    }

@app.post("/api/generate-routine")
async def generate_routine(payload: FullProfilePayload):
    payload_dict = payload.model_dump()
    
    if has_severe_condition(payload_dict.get("ml_detections", {})):
        return {
            "status": "clinical_concern_detected",
            "message": "Health advisory triggered. Rerouting to dermatologist CTA."
        }
    
    safe_catalog = pre_filter_catalog(payload_dict, payload_dict.get("product_catalog", []))
    
    if not safe_catalog:
         return {
             "error": "no_suitable_products",
             "message": "We couldn't find products matching your budget and sensitivities."
         }
         
    llm_result = await call_deepseek_llm(payload_dict, safe_catalog)
    
    return {
        "status": "success",
        "data": llm_result
    }