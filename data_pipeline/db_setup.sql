CREATE DATABASE career_recommender;
USE career_recommender;

CREATE TABLE career_roles (
  id INT AUTO_INCREMENT PRIMARY KEY,
  esco_id VARCHAR(100),
  role_name VARCHAR(255),
  description TEXT,
  source VARCHAR(20) DEFAULT 'ESCO'
);

CREATE TABLE technical_skills (
  id INT AUTO_INCREMENT PRIMARY KEY,
  onet_id VARCHAR(100),
  skill_name VARCHAR(255),
  category VARCHAR(100),
  cleaned_skill TEXT
);

SELECT 
    g.skill_name AS "General Skill (ESCO)", 
    t.skill_name AS "Technical Tool (O*NET)", 
    ROUND(gt.relevance_score, 2) AS "AI Confidence"
FROM general_to_technical gt
JOIN general_skills g ON gt.general_skill_id = g.id
JOIN technical_skills t ON gt.technical_skill_id = t.id
WHERE gt.relevance_score > 0.60
ORDER BY gt.relevance_score DESC
LIMIT 30;

SELECT 
    g.skill_name AS "General Skill", 
    t.skill_name AS "Technical Tool", 
    ROUND(gt.relevance_score, 2) AS "Score"
FROM general_to_technical gt
JOIN general_skills g ON gt.general_skill_id = g.id
JOIN technical_skills t ON gt.technical_skill_id = t.id
WHERE g.skill_name != t.skill_name
  AND gt.relevance_score BETWEEN 0.50 AND 0.95
ORDER BY gt.relevance_score DESC
LIMIT 20;

ALTER TABLE technical_skills ADD COLUMN soc_code VARCHAR(20);

CREATE TABLE role_technical_stats (
    role_id INT,
    technical_skill_id INT,
    frequency INT DEFAULT 0,
    PRIMARY KEY (role_id, technical_skill_id),
    FOREIGN KEY (role_id) REFERENCES career_roles(id),
    FOREIGN KEY (technical_skill_id) REFERENCES technical_skills(id)
);

SHOW VARIABLES LIKE 'secure_file_priv';
CREATE TABLE IF NOT EXISTS role_technical_skills (
    role_id INT,
    technical_skill_id INT,
    relevance_score FLOAT DEFAULT 0.95,
    PRIMARY KEY (role_id, technical_skill_id),
    FOREIGN KEY (role_id) REFERENCES career_roles(id) ON DELETE CASCADE,
    FOREIGN KEY (technical_skill_id) REFERENCES technical_skills(id) ON DELETE CASCADE
);
CREATE TABLE courses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    course_title VARCHAR(255),
    organization VARCHAR(255),
    skills_covered TEXT,
    rating FLOAT,
    difficulty VARCHAR(50),
    course_url VARCHAR(500)
);
ALTER TABLE courses MODIFY course_title TEXT;
ALTER TABLE courses MODIFY organization TEXT;
ALTER TABLE courses MODIFY skills_covered TEXT;
ALTER TABLE courses MODIFY course_url TEXT;
ALTER TABLE courses ADD COLUMN description LONGTEXT AFTER skills_covered;
ALTER TABLE courses ADD COLUMN primary_skill VARCHAR(100);