from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import json

app = FastAPI()

# 1. Add Middleware FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Setup Model and Data
model = SentenceTransformer("all-MiniLM-L6-v2")

LEVEL_MAP = {
    "Not Started": 0, "0": 0,
    "Beginner": 1, "1": 1,
    "Intermediate": 2, "2": 2,
    "Advanced": 3, "3": 3,
    "Expert": 4, "4": 4
}

# This is the "Reverse" map that ensures we send "Expert" back to the UI, not "4"
REVERSE_LEVEL_MAP = {
    0: "Not Started",
    1: "Beginner",
    2: "Intermediate",
    3: "Advanced",
    4: "Expert"
}

with open("./esco_processing/curated_career_fields.json", "r") as f:
    FIELDS = json.load(f)

def build_user_semantic_text(profile: dict) -> str:
    skills = ", ".join([s["name"] for s in profile.get("skills", [])])
    interests = ", ".join(profile.get("interests", []))
    education = profile.get("background", {}).get("education_level", "")
    goal = profile.get("background", {}).get("career_goal", "")
    return f"Goal: {goal}. Skills: {skills}. Interests: {interests}. Education: {education}."

@app.post("/api/field-suitability")
def compute_field_suitability(profile: dict):
    user_text = build_user_semantic_text(profile)
    user_embedding = model.encode(user_text, convert_to_tensor=True)

    results = []
    for field in FIELDS:
        field_text = f"{field['field_name']}. {field['description']}"
        field_embedding = model.encode(field_text, convert_to_tensor=True)
        similarity = util.cos_sim(user_embedding, field_embedding).item()

        results.append({
            "id": field["field_id"],
            "field_name": field["field_name"],
            "description": field["description"],
            "score": round(similarity * 100, 2),
            "top_skills": field.get("priority_skills", [])[:5]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results[:3]}

# REVERSE_LEVEL_MAP = {v: k for k, v in LEVEL_MAP.items()}

@app.post("/api/skill-gaps")
def compute_skill_gaps(payload: dict):
    profile = payload.get("profile")
    field_id = payload.get("field_id")

    if not profile or not field_id:
        return {"error": "Missing profile or field_id"}, 400

    user_skills_raw = profile.get("skills", [])
    user_skills_mapped = {}
    
    for s in user_skills_raw:
        name = s.get("name", "").lower().strip()
        # FIX: Ensure we treat the level as a string to match the LEVEL_MAP keys
        raw_level = str(s.get("level", "1"))
        level_num = LEVEL_MAP.get(raw_level, 1)

        parts = name.replace("/", ",").replace("&", ",").split(",")
        for part in parts:
            clean_part = part.strip()
            if clean_part:
                user_skills_mapped[clean_part] = level_num
                # LOG CHECK: This should now show Mapped: html -> 4
                print(f"Mapped: {clean_part} -> {level_num}")

    field = next((f for f in FIELDS if f["field_id"] == field_id), None)
    if not field:
        return {"error": "Field not found"}, 404

    gaps = []
    strengths = []
    TARGET_VAL = 2 
    TARGET_LABEL = "Intermediate"

    for skill_name in field.get("priority_skills", []):
        skill_lower = skill_name.lower().strip()
        user_val = 0 
        
        # Check for direct or partial match
        if skill_lower in user_skills_mapped:
            user_val = user_skills_mapped[skill_lower]
        else:
            for u_skill, u_val in user_skills_mapped.items():
                if u_skill in skill_lower or skill_lower in u_skill:
                    user_val = u_val
                    break

        # This uses the Reverse Map to get the word "Expert"
        current_label = REVERSE_LEVEL_MAP.get(user_val, "Not Started")

        if user_val >= TARGET_VAL:
            strengths.append({
                "skill": skill_name,
                "current_level": current_label
            })
        else:
            gaps.append({
                "skill": skill_name,
                "required_level": TARGET_LABEL,
                "current_level": current_label,
                "gap_score": TARGET_VAL - user_val
            })
            
        print(f"SKILL: {skill_name} | USER_VAL: {user_val} | LABEL: {current_label}")

    total_skills = len(field.get("priority_skills", []))
    match_percentage = round((len(strengths) / total_skills) * 100) if total_skills > 0 else 0

    return {
        "field_name": field["field_name"],
        "match_percentage": match_percentage,
        "skill_gaps": gaps,
        "strengths": strengths
    }