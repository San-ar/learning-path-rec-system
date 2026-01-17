import requests

API_URL = "http://127.0.0.1:8000"

# ROLE EQUIVALENCE (SEMANTIC GROUND TRUTH)
ROLE_EQUIVALENCE = {
    "Software Engineer": [
        "Software Engineer",
        "Backend Developer",
        "Application Developer"
    ],
    "Data Analyst": [
        "Data Analyst",
        "Business Intelligence Analyst"
    ],
    "AI Engineer": [
        "AI Engineer",
        "Artificial Intelligence Engineer",
        "Machine Learning Engineer",
        "ML Engineer"
    ],
    "Web Developer": [
        "Web Developer",
        "Frontend Developer",
        "Full Stack Developer"
    ],
    "Mobile Developer": [
        "Mobile Developer",
        "Android Developer",
        "iOS Developer"
    ],
    "Cloud Architect": [
        "Cloud Architect",
        "DevOps Engineer",
        "Cloud Engineer"
    ],
    "Security Analyst": [
        "Security Analyst",
        "Cybersecurity Analyst"
    ],
    "Database Administrator": [
        "Database Administrator",
        "Database Engineer"
    ],
    "Network Engineer": [
        "Network Engineer",
        "Systems Engineer"
    ],
    "Game Developer": [
        "Game Developer",
        "Game Engineer"
    ],
    "UI/UX Designer": [
        "UI/UX Designer",
        "Product Designer"
    ]
}

# DOMAIN-BASED SYNTHETIC TEST CASES
DOMAINS = [
    {"role": "Software Engineer", "skill": "Java", "interest": "programming_software"},
    {"role": "Data Analyst", "skill": "Excel", "interest": "data_analysis"},
    {"role": "AI Engineer", "skill": "Python", "interest": "ai_ml"},
    {"role": "Web Developer", "skill": "HTML", "interest": "web_technologies"},
    {"role": "Mobile Developer", "skill": "Kotlin", "interest": "mobile_dev"},
    {"role": "Cloud Architect", "skill": "AWS", "interest": "cloud_devops"},
    {"role": "Security Analyst", "skill": "Linux", "interest": "cybersecurity"},
    {"role": "Database Administrator", "skill": "SQL", "interest": "databases_backend"},
    {"role": "Network Engineer", "skill": "TCP/IP", "interest": "networking_systems"},
    {"role": "Game Developer", "skill": "C#", "interest": "game_dev"},
    {"role": "UI/UX Designer", "skill": "Figma", "interest": "uiux_frontend"}
]

# UTILITY FUNCTIONS
def is_semantically_correct(predicted: str, true_role: str) -> bool:
    return predicted in ROLE_EQUIVALENCE.get(true_role, [])

def generate_test_cases():
    cases = []
    for domain in DOMAINS:
        for level in ["Beginner", "Intermediate", "Intermediate"]:
            cases.append({
                "true_role": domain["role"],
                "payload": {
                    "profile": {
                        "background": {"career_goal": domain["role"]},
                        "skills": [{"name": domain["skill"], "level": level}],
                        "interests": [domain["interest"]],
                        "logistics": {"hours_per_week": 10}
                    }
                }
            })
    return cases

# EVALUATION PIPELINE
def run_evaluation():
    test_cases = generate_test_cases()
    n = len(test_cases)

    top1 = 0
    top3 = 0
    mrr_sum = 0
    prereq_violations = 0
    total_transitions = 0

    print(f"Running evaluation on {n} synthetic learner profiles...\n")

    for case in test_cases:
        # FIELD SUITABILITY EVALUATION
        fs_resp = requests.post(
            f"{API_URL}/api/field-suitability",
            json=case["payload"]
        )
        results = fs_resp.json().get("results", [])
        predicted_roles = [r["field_name"] for r in results]

        if predicted_roles:
            # Top-1
            if is_semantically_correct(predicted_roles[0], case["true_role"]):
                top1 += 1

            # Top-3 + MRR
            for rank, role in enumerate(predicted_roles[:3], start=1):
                if is_semantically_correct(role, case["true_role"]):
                    top3 += 1
                    mrr_sum += 1 / rank
                    break

        # PATH QUALITY EVALUATION
        if not results:
            continue

        gap_resp = requests.post(
            f"{API_URL}/api/skill-gaps",
            json={
                "profile": case["payload"]["profile"],
                "field_id": results[0]["id"]
            }
        )

        gaps = gap_resp.json().get("skill_gaps", [])

        path_resp = requests.post(
            f"{API_URL}/api/learning-path",
            json={
                "gaps": gaps,
                "career_goal": case["true_role"]
            }
        )

        path = path_resp.json().get("learning_path", [])

        level_order = {"Not Started": 0, "Beginner": 1, "Intermediate": 2, "Advanced": 3}

        for skill_group in path:
            steps = skill_group.get("steps", [])
            for i in range(len(steps) - 1):
                curr = level_order.get(steps[i]["level"], 0)
                nxt = level_order.get(steps[i + 1]["level"], 0)
                total_transitions += 1
                if curr > nxt:
                    prereq_violations += 1

    # FINAL METRICS
    print("=========== FINAL EVALUATION RESULTS ===========")
    print(f"Top-1 Accuracy: {top1 / n:.2f}")
    print(f"Top-3 Accuracy: {top3 / n:.2f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr_sum / n:.2f}")
    print(
        f"Prerequisite Violation Rate: "
        f"{(prereq_violations / total_transitions) if total_transitions else 0:.2f}"
    )
    print("===============================================")

if __name__ == "__main__":
    run_evaluation()
