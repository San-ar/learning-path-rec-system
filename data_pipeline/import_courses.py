import mysql.connector
from datasets import load_dataset
from typing import Dict, Any, List, cast

def direct_import_all_courses():
    print("Fetching dataset from HuggingFace")
    dataset = load_dataset("azrai99/coursera-course-dataset", split='train')

    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="career_recommender"
    )
    cursor = db.cursor()

    print("Cleaning courses table.")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
    cursor.execute("TRUNCATE TABLE courses;")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")

    print(f"Importing {len(dataset)} courses...")
    count = 0

    for record in dataset:
        row = cast(Dict[str, Any], record)
        
        title = str(row.get('title', 'Unknown Course'))
        org = str(row.get('Organization', 'Unknown Organization'))
        skills = str(row.get('Skills', ''))
        desc = str(row.get('Description', ''))
        url = str(row.get('URL', ''))
        
        diff = str(row.get('Level', 'Beginner')).replace(' level', '')

        raw_rating = row.get('rating')
        try:
            # Check for strings or None before converting
            if raw_rating in [None, "Rating not found", "null"]:
                rating = 0.0
            else:
                rating = float(str(raw_rating))
        except (ValueError, TypeError):
            rating = 0.0

        # SQL Insertion
        sql = """
            INSERT INTO courses (course_title, organization, skills_covered, description, rating, difficulty, course_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        # Store Title + Skills in the 'skills_covered' column for easier API searching
        searchable_skills = f"{title}, {skills}"
        
        values = (title, org, searchable_skills, desc, rating, diff, url)

        try:
            cursor.execute(sql, values)
            count += 1
            if count % 200 == 0:
                print(f"Imported {count} rows...")
        except Exception as e:
            print(f"Skipping row '{title[:30]}': {e}")

    db.commit()
    print(f"\n Success! Total courses in DB: {count}")
    
    cursor.close()
    db.close()

if __name__ == "__main__":
    direct_import_all_courses()