import pandas as pd
import mysql.connector
import re
from typing import cast, Tuple, Any

# 1. Database Connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="career_recommender"
)

# FIXED: Added buffered=True to prevent "Unread result found" error
cursor = db.cursor(buffered=True)

def clean_text(text):
    if not text: 
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

# 2. Load the Core Files
try:
    # Using relative paths assuming you run this from inside data_pipeline folder
    occupations = pd.read_csv("./raw_data/occupations_en.csv")
    skills = pd.read_csv("./raw_data/skills_en.csv")
    relations = pd.read_csv("./raw_data/occupationSkillRelations_en.csv")
    print("CSV files loaded successfully.")
except Exception as e:
    print(f"Error loading CSVs: {e}")
    exit()

# 3. Filter for Tech Roles (ISCO Group 25)
tech_occupation = occupations[occupations['iscoGroup'].astype(str).str.startswith('25')].copy()
print(f"Processing {len(tech_occupation)} ICT Professional roles.")

# 4. Main Processing Loop
for _, job in tech_occupation.iterrows():
    # A. Insert the Job Role
    cursor.execute("""
        INSERT IGNORE INTO career_roles (esco_id, role_name, description)
        VALUES (%s, %s, %s)
    """, (job['conceptUri'], job['preferredLabel'], job['description']))
    
    # B. Fetch Role ID safely
    cursor.execute("SELECT id FROM career_roles WHERE esco_id = %s", (job['conceptUri'],))
    
    # Using cast to Tuple[Any, ...] tells Pylance that index [0] is valid
    role_result = cast(Tuple[Any, ...], cursor.fetchone())
    
    if role_result:
        role_id = role_result[0]

        # C. Get URI of essential skills for this job
        job_uri = job['conceptUri']
        relevant_skills_uris = relations[
            (relations['occupationUri'] == job_uri) & 
            (relations['relationType'] == 'essential')
        ]['skillUri']

        # D. Filter actual skill details
        job_skills = skills[skills['conceptUri'].isin(relevant_skills_uris)]
        
        for _, skill_row in job_skills.iterrows():
            cleaned = clean_text(str(skill_row['preferredLabel']))
            
            # E. Insert the General Skill
            cursor.execute("""
                INSERT IGNORE INTO general_skills (esco_skill_id, skill_name, cleaned_skill)
                VALUES (%s, %s, %s)
            """, (skill_row['conceptUri'], skill_row['preferredLabel'], cleaned))
            
            # F. Fetch Skill ID safely
            cursor.execute("SELECT id FROM general_skills WHERE esco_skill_id = %s", (skill_row['conceptUri'],))
            skill_result = cast(Tuple[Any, ...], cursor.fetchone())
            
            if skill_result:
                skill_id = skill_result[0]
                
                # G. Link Role to Skill
                cursor.execute("""
                    INSERT IGNORE INTO role_general_skills (role_id, general_skill_id, importance)
                    VALUES (%s, %s, %s)
                """, (role_id, skill_id, 1.0))

# 5. Save Changes and Close
db.commit()
print("Success! Tech roles and essential skills have been saved to the database.")

cursor.close()
db.close()