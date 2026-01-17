import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import mysql.connector
from typing import Dict, List, Any, cast

# APP INIT
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL (SBERT for Semantic Ranking)
model = SentenceTransformer("all-MiniLM-L6-v2")

# DB CONNECTION
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="career_recommender"
    )

# LEVEL MAPS
LEVEL_MAP: Dict[str, int] = {
    "Not Started": 0, "Beginner": 1, "Intermediate": 2, "Advanced": 3, "Expert": 4,
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4
}

REVERSE_LEVEL_MAP: Dict[int, str] = {
    0: "Not Started", 1: "Beginner", 2: "Intermediate", 3: "Advanced", 4: "Expert"
}

@app.get("/api/skills")
def get_skills():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, skill_name FROM technical_skills")
    skills = cursor.fetchall()
    cursor.close()
    conn.close()
    return skills


# FIELD SUITABILITY
@app.post("/api/field-suitability")
def field_suitability(payload: Dict[str, Any]):
    profile = payload.get("profile", {})
    skills = ", ".join([s["name"] for s in profile.get("skills", [])])
    interests = ", ".join(profile.get("interests", []))
    goal = profile.get("background", {}).get("career_goal", "")

    user_text = f"Career goal: {goal}. Skills: {skills}. Interests: {interests}."
    user_embedding = model.encode(user_text, convert_to_tensor=True)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, role_name, description FROM career_roles")
    roles = cast(List[Dict[str, Any]], cursor.fetchall())

    results = []
    for role in roles:
        role_text = f"{role['role_name']} {role.get('description','')}"
        role_embedding = model.encode(role_text, convert_to_tensor=True)
        score = util.cos_sim(user_embedding, role_embedding).item()

        cursor.execute("""
            SELECT ts.skill_name FROM technical_skills ts
            JOIN role_technical_skills rts ON ts.id = rts.technical_skill_id
            WHERE rts.role_id = %s ORDER BY rts.relevance_score DESC LIMIT 5
        """, (role["id"],))
        skills_rows = cast(List[Dict[str, Any]], cursor.fetchall())
        top_skills = [str(r["skill_name"]) for r in skills_rows]

        results.append({
            "id": int(role["id"]),
            "field_name": role["role_name"].title(),
            "score": round(score * 100, 2),
            "top_skills": top_skills
        })

    cursor.close()
    conn.close()
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results[:3]}

# SKILL GAP ANALYSIS
@app.post("/api/skill-gaps")
def compute_skill_gaps(payload: Dict[str, Any]):
    profile = payload.get("profile", {})
    field_id = payload.get("field_id")
    if not profile or not field_id: return {"error": "Missing profile or field_id"}

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT id, role_name FROM career_roles WHERE id = %s", (field_id,))
    role_data = cast(Dict[str, Any], cursor.fetchone())
    if not role_data: return {"error": "Role not found"}

    # FIX: We now pull the relevance_score to ensure the "Expert Order" is preserved
    cursor.execute("""
        SELECT ts.skill_name, rts.relevance_score 
        FROM technical_skills ts
        JOIN role_technical_skills rts ON ts.id = rts.technical_skill_id
        WHERE rts.role_id = %s 
        ORDER BY rts.relevance_score DESC
    """, (field_id,))
    role_skills_data = cast(List[Dict[str, Any]], cursor.fetchall())

    user_skills = {str(s.get("name", "")).lower().strip(): LEVEL_MAP.get(str(s.get("level", "1")), 1) 
                   for s in profile.get("skills", [])}

    strengths, gaps = [], []
    TARGET = 2 

    for item in role_skills_data:
        skill = str(item["skill_name"])
        skill_l = skill.lower()
        
        # Check if user already knows this skill
        user_val = next((val for name, val in user_skills.items() if name in skill_l or skill_l in name), 0)
        label = REVERSE_LEVEL_MAP.get(user_val, "Not Started")

        if user_val >= TARGET:
            strengths.append({"skill": skill, "current_level": label})
        else:
            # Gaps are now appended in the ORDER of their relevance_score from the DB
            gaps.append({
                "skill": skill, 
                "required_level": "Intermediate", 
                "current_level": label, 
                "gap_score": TARGET - user_val
            })

    cursor.close()
    conn.close()
    return {
        "field_name": role_data["role_name"], 
        "match_percentage": round((len(strengths)/len(role_skills_data))*100, 2) if role_skills_data else 0, 
        "strengths": strengths, 
        "skill_gaps": gaps
    }

# COLLABORATIVE FILTERING LOGIC
def get_cf_scores(course_ids: List[int], cursor) -> Dict[int, float]:
    if not course_ids: return {}
    format_ids = ",".join(["%s"] * len(course_ids))
    query = f"""
        SELECT c2.course_id, COUNT(*) AS freq
        FROM user_course_interactions c1
        JOIN user_course_interactions c2 ON c1.user_id = c2.user_id
        WHERE c1.course_id IN ({format_ids}) AND c2.course_id NOT IN ({format_ids})
        GROUP BY c2.course_id
    """
    cursor.execute(query, tuple(course_ids + course_ids))
    rows = cursor.fetchall()
    return {int(r["course_id"]): float(r["freq"]) for r in rows}

# LEARNING PATH GENERATION
@app.post("/api/learning-path")
def generate_learning_path(payload: Dict[str, Any]):
    gaps = payload.get("gaps", [])
    profile_bg = payload.get("profile", {}).get("background", {})
    career_goal = payload.get("selected_role") or profile_bg.get("selected_role") or profile_bg.get("career_goal")
    
    if not career_goal:
        raise HTTPException(status_code=400, detail="Career goal is missing.")

    LEVEL_ORDER = ["Not Started", "Beginner", "Intermediate", "Advanced", "Expert"]
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    technical_target = f"Technical coding syllabus for {career_goal}. Focus on software tools."
    goal_embedding = model.encode(technical_target, convert_to_tensor=True)

    learning_path = []

    for gap in gaps:
        skill = gap["skill"]
        current_level = gap.get("current_level", "Beginner")
        target_level = gap.get("required_level", "Intermediate")

        try:
            levels = LEVEL_ORDER[LEVEL_ORDER.index(current_level): LEVEL_ORDER.index(target_level) + 1]
        except ValueError: continue

        steps = []
        for level in levels:
            regex_pattern = f"(^|[^a-zA-Z0-9]){re.escape(skill)}([^a-zA-Z0-9]|$)"
            
            cursor.execute("""
                SELECT * FROM courses 
                WHERE (primary_skill = %s OR course_title REGEXP %s) 
                AND difficulty = %s
            """, (skill, regex_pattern, level))
            
            candidates = cast(List[Dict[str, Any]], cursor.fetchall())
            if not candidates: continue

            # SBERT Semantic Ranking
            texts = [f"SYLLABUS: {c['course_title']}. {c['description']}" for c in candidates]
            embeddings = model.encode(texts, convert_to_tensor=True)
            semantic_scores = util.cos_sim(goal_embedding, embeddings)[0]

            # Collaborative Filtering (Popularity)
            course_ids = [int(c["id"]) for c in candidates]
            cf_data = get_cf_scores(course_ids, cursor)

            best_idx, max_score = -1, -1.0
            for i, c in enumerate(candidates):
                sem = semantic_scores[i].item()
                cf_norm = min(cf_data.get(int(c["id"]), 0.0) / 3.0, 1.0)
                final_score = (sem * 0.7) + (cf_norm * 0.3)
                
                if final_score > max_score:
                    max_score, best_idx = final_score, i

            if best_idx != -1:
                best = candidates[best_idx]
                steps.append({
                    "level": level,
                    "course_title": best["course_title"],
                    "provider": best["organization"],
                    "url": best["course_url"],
                    "match_score": round(max_score, 2),
                    "brief": f"{best['description'][:150]}...",
                    "reason": "Top rated" if cf_data.get(int(best["id"]), 0) > 2 else "Relevant"
                })

        if steps:
            learning_path.append({
                "skill": skill, 
                "current_level": current_level, 
                "target_level": target_level, 
                "steps": steps
            })

    cursor.close()
    conn.close()
    return {"career_goal": career_goal, "learning_path": learning_path}