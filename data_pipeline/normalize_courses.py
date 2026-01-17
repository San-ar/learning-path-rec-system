import mysql.connector
from typing import List, Dict, Any, cast

KEY_SKILLS = [
    "Python", "JavaScript", "SQL", "HTML", "CSS",
    "React", "Node", "Java", "Git", "Data Analysis"
]

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="career_recommender"
)

cursor = db.cursor(dictionary=True)

cursor.execute("SELECT id, course_title FROM courses")

rows = cast(List[Dict[str, Any]], cursor.fetchall())

for row in rows:
    course_id = row["id"]
    text = f"{row['course_title']} {row.get('skills_covered','')}".lower()
    
    primary = None

    for skill in KEY_SKILLS:
        if skill.lower() in text:
            primary = skill
            break

    if primary:
        update_query = "UPDATE courses SET primary_skill = %s WHERE id = %s"
        cursor.execute(update_query, (primary, course_id))

db.commit()
cursor.close()
db.close()

print("Course primary skills normalized")
