import pandas as pd
import mysql.connector
import re
from typing import cast, Tuple, Any

# Database Connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="career_recommender"
)

# For looping over the records one by one.
cursor = db.cursor(buffered=True)

def clean_text(text):
    if not text: 
        return ""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

# Load the Core Files
try:
    occupations = pd.read_csv("./raw_data/occupations_en.csv")
    skills = pd.read_csv("./raw_data/skills_en.csv")
    relations = pd.read_csv("./raw_data/occupationSkillRelations_en.csv")
    print("CSV files loaded successfully.")
except Exception as e:
    print(f"Error loading CSVs: {e}")
    exit()

# Filter for Tech Roles (ISCO Group 25)
tech_occupation = occupations[occupations['iscoGroup'].astype(str).str.startswith('25')].copy()
print(f"Processing {len(tech_occupation)} ICT Professional roles.")

# Main Processing Loop
for _, job in tech_occupation.iterrows():
    role_name_clean = str(job['preferredLabel']).strip()
    
    # Insert the Job Role
    cursor.execute("""
        INSERT IGNORE INTO career_roles (esco_id, role_name, description)
        VALUES (%s, %s, %s)
    """, (job['conceptUri'], role_name_clean, job['description']))
    
    # Get Role ID
    cursor.execute("SELECT id FROM career_roles WHERE esco_id = %s", (job['conceptUri'],))
    role_result = cast(Tuple[Any, ...], cursor.fetchone())
    
    if role_result:
        role_id = role_result[0]
        job_uri = job['conceptUri']
        relevant_skills_uris = relations[
            (relations['occupationUri'] == job_uri) & 
            (relations['relationType'] == 'essential')
        ]['skillUri']

        job_skills = skills[skills['conceptUri'].isin(relevant_skills_uris)]
        
        for _, skill_row in job_skills.iterrows():
            skill_name_clean = str(skill_row['preferredLabel']).strip()
            cleaned_slug = clean_text(skill_name_clean)
            
            # Insert the General Skill
            cursor.execute("""
                INSERT IGNORE INTO general_skills (esco_skill_id, skill_name, cleaned_skill)
                VALUES (%s, %s, %s)
            """, (skill_row['conceptUri'], skill_name_clean, cleaned_slug))
            
            # Get Skill ID
            cursor.execute("SELECT id FROM general_skills WHERE esco_skill_id = %s", (skill_row['conceptUri'],))
            skill_result = cast(Tuple[Any, ...], cursor.fetchone())
            
            if skill_result:
                skill_id = skill_result[0]
                
                # Link Role to Skill
                cursor.execute("""
                    INSERT IGNORE INTO role_general_skills (role_id, general_skill_id, importance)
                    VALUES (%s, %s, %s)
                """, (role_id, skill_id, 1.0))

db.commit()
print("Success! Tech roles and essential skills have been cleaned and saved.")
cursor.close()
db.close()