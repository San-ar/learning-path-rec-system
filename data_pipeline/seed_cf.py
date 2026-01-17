import mysql.connector
import random

def seed_collaborative_filtering():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="career_recommender"
    )
    cursor = db.cursor(dictionary=True)

    print("--- Starting CF Seeding ---")

    # Get IDs of technical courses
    keywords = ['Python', 'JavaScript', 'React', 'SQL', 'Git', 'Node', 'API', 'Docker']
    
    technical_courses = []
    for kw in keywords:
        cursor.execute("SELECT id FROM courses WHERE course_title LIKE %s OR description LIKE %s", (f'%{kw}%', f'%{kw}%'))
        res = cursor.fetchall()

        if res:
            for r in res:
                course_id = r.get('id') if isinstance(r, dict) else r[0]
                technical_courses.append(course_id)

    if not technical_courses:
        print("No technical courses found to seed. Check your 'courses' table content!")
        return

    # Create fake users and interactions
    fake_user_ids = range(500, 550)
    interactions = []

    for user_id in fake_user_ids:
        # Fake users taking 3 to 6 technical courses
        num_courses = random.randint(3, 6)
        chosen_courses = random.sample(technical_courses, min(num_courses, len(technical_courses)))
        
        for course_id in chosen_courses:
            interactions.append((user_id, course_id))

    # Insert into database
    # IGNORE handles cases where the user/course combo might already exist
    query = "INSERT IGNORE INTO user_course_interactions (user_id, course_id) VALUES (%s, %s)"
    cursor.executemany(query, interactions)
    
    db.commit()
    print(f"Successfully inserted {cursor.rowcount} interactions for {len(fake_user_ids)} fake users.")
    
    cursor.close()
    db.close()

if __name__ == "__main__":
    seed_collaborative_filtering()