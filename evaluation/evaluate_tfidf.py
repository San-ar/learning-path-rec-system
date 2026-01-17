import mysql.connector
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, cast

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="career_recommender"
    )

# SYNTHETIC TEST CASES
DOMAINS = [
    {"val": "programming_software", "role": "Software Engineer", "skill": "Java"},
    {"val": "data_analysis", "role": "Data Analyst", "skill": "Excel"},
    {"val": "ai_ml", "role": "AI Engineer", "skill": "Python"},
    {"val": "web_technologies", "role": "Web Developer", "skill": "HTML"},
    {"val": "mobile_dev", "role": "Mobile Developer", "skill": "Kotlin"},
    {"val": "cloud_devops", "role": "Cloud Architect", "skill": "AWS"},
    {"val": "cybersecurity", "role": "Security Analyst", "skill": "Linux"},
    {"val": "databases_backend", "role": "Database Administrator", "skill": "SQL"},
    {"val": "networking_systems", "role": "Network Engineer", "skill": "TCP/IP"},
    {"val": "game_dev", "role": "Game Developer", "skill": "C#"},
    {"val": "uiux_frontend", "role": "UI/UX Designer", "skill": "Figma"},
]

def generate_test_cases():
    cases = []
    for d in DOMAINS:
        for _ in range(3):
            cases.append({
                "true_role": d["role"],
                "profile_text": (
                    f"{d['role']} career path. "
                    f"{d['role']} responsibilities and required skills. "
                    f"Technical skills include {d['skill']}. "
                    f"Experience in {d['val']} and related technologies. "
                    f"{d['skill']} development and practical usage."
                )
            })
    return cases

# TF-IDF BASELINE EVALUATION
def run_tfidf_baseline():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT role_name, description FROM career_roles")
    roles = cast(List[Dict[str, Any]], cursor.fetchall())

    role_texts = [
        f"{r.get('role_name', '')} {r.get('description', '')}"
        for r in roles
    ]

    role_names = [str(r.get("role_name", "")) for r in roles]

    test_cases = generate_test_cases()
    n = len(test_cases)

    top1 = 0
    top3 = 0
    mrr = 0.0

    # Added ngram_range to catch multi-word roles like "Software Engineer"
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(role_texts)

    for case in test_cases:
        user_vec = vectorizer.transform([case["profile_text"]])
        sims = cosine_similarity(user_vec, tfidf_matrix)[0]

        ranked_indices = np.argsort(sims)[::-1]
        
        # NORMALIZATION STEP
        true_role_norm = case["true_role"].lower().strip()
        
        # Get top 3 roles and normalize them for comparison
        ranked_roles = [role_names[i] for i in ranked_indices[:3]]
        ranked_roles_norm = [r.lower().strip() for r in ranked_roles]

        if ranked_roles_norm:
            # Check Top-1
            if ranked_roles_norm[0] == true_role_norm:
                top1 += 1
            
            # Check Top-3 and MRR
            if true_role_norm in ranked_roles_norm:
                top3 += 1
                rank = ranked_roles_norm.index(true_role_norm) + 1
                mrr += 1 / rank

    print("\n=========== TF-IDF BASELINE RESULTS ===========")
    print(f"Top-1 Accuracy: {top1 / n:.2f}")
    print(f"Top-3 Accuracy: {top3 / n:.2f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr / n:.2f}")
    print("=============================================")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    run_tfidf_baseline()