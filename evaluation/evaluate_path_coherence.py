import requests
import numpy as np
from sentence_transformers import SentenceTransformer, util

API_URL = "http://127.0.0.1:8000"
model = SentenceTransformer("all-MiniLM-L6-v2")

DOMAINS = [
    {"role": "Software Engineer", "skill": "Java"},
    {"role": "Data Analyst", "skill": "Excel"},
    {"role": "AI Engineer", "skill": "Python"},
    {"role": "Web Developer", "skill": "HTML"},
    {"role": "Mobile Developer", "skill": "Kotlin"}
]

def compute_path_coherence():
    coherence_scores = []
    print(f"Evaluating Path Coherence...\n")

    for domain in DOMAINS:
        # Construct payload structure
        payload = {
            "profile": {
                "background": {"career_goal": domain["role"]},
                "skills": [{"name": domain["skill"], "level": "Beginner"}],
                "interests": [domain["role"]]
            }
        }

        # Field Suitability
        field_resp = requests.post(f"{API_URL}/api/field-suitability", json=payload).json()
        results = field_resp.get("results", [])
        if not results:
            print(f"FAILED at Field Suitability for {domain['role']}")
            continue
        
        field_id = results[0]["id"]
        field_name = results[0]["field_name"]

        # Skill Gaps
        gap_payload = {"profile": payload["profile"], "field_id": field_id}
        gap_resp = requests.post(f"{API_URL}/api/skill-gaps", json=gap_payload).json()
        gaps = gap_resp.get("skill_gaps", [])
        if not gaps:
            print(f"FAILED at Skill Gaps for {domain['role']}")
            continue

        # Learning Path
        path_payload = {"gaps": gaps, "profile": payload["profile"]}
        path_resp = requests.post(f"{API_URL}/api/learning-path", json=path_payload).json()
        path = path_resp.get("learning_path", [])
        
        if not path:
            print(f"FAILED at Learning Path (Empty) for {domain['role']}")
            continue

        # Calculation
        goal_embedding = model.encode(field_name, convert_to_tensor=True)
        for skill_group in path:
            for step in skill_group.get("steps", []):
                course_text = f"{step['course_title']} {step['brief']}"
                course_embedding = model.encode(course_text, convert_to_tensor=True)
                sim = util.cos_sim(goal_embedding, course_embedding).item()
                coherence_scores.append(sim)
                print(f"SUCCESS: {domain['role']} -> {step['course_title'][:30]} (Sim: {sim:.2f})")

    # Final Stats
    if not coherence_scores:
        print("\nCRITICAL ERROR: No scores were collected. Check if your FastAPI server is running.")
        return

    print("\n=========== LEARNING PATH COHERENCE RESULTS ===========")
    print(f"Average Path Coherence Score: {np.mean(coherence_scores):.2f}")
    print(f"Minimum Coherence Score: {np.min(coherence_scores):.2f}")
    print(f"Maximum Coherence Score: {np.max(coherence_scores):.2f}")
    print("======================================================")

if __name__ == "__main__":
    compute_path_coherence()