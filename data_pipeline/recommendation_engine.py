import mysql.connector
from typing import cast, List, Dict, Tuple, Any

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="career_recommender"
)

cursor = db.cursor(dictionary=True)

def get_role_id(role_name: str) -> int | None:
    cursor.execute(
        "SELECT id FROM career_roles WHERE role_name = %s",
        (role_name.strip(),)
    )
    row = cursor.fetchone()
    return int(cast(Dict[str, Any], row)["id"]) if row else None


def get_role_technical_skills(role_id: int) -> List[Dict]:
    cursor.execute("""
        SELECT 
            ts.skill_name,
            rts.relevance_score
        FROM role_technical_skills rts
        JOIN technical_skills ts 
            ON rts.technical_skill_id = ts.id
        WHERE rts.role_id = %s
        ORDER BY rts.relevance_score DESC, ts.skill_name ASC
    """, (role_id,))
    return cast(List[Dict[str, Any]], cursor.fetchall())

# Recommendation Logic (Content-based Filtering)
def recommend_tools(
    target_role: str,
    user_known_tools: List[str] | None = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Main recommendation function
    """
    known_tools_list = user_known_tools or []
    known_set = {t.lower().strip() for t in known_tools_list}

    role_id = get_role_id(target_role)
    if not role_id:
        return {
            "success": False,
            "error": f"Role '{target_role}' not found."
        }

    role_tools = get_role_technical_skills(role_id)

    recommendations = []
    already_known = []

    for tool in role_tools:
        tool_name = str(tool["skill_name"])
        score = tool["relevance_score"]

        if tool_name.lower() in known_set:
            already_known.append(tool_name)
        else:
            recommendations.append({
                "tool": tool_name,
                "relevance_score": score
            })

    return {
        "success": True,
        "role": target_role,
        "known_tools": sorted(already_known),
        "recommended_tools": recommendations[:limit],
        "total_tools_for_role": len(role_tools),
        "missing_tools_count": len(recommendations)
    }