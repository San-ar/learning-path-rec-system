CREATE DATABASE career_recommender;
USE career_recommender;

CREATE TABLE career_roles (
  id INT AUTO_INCREMENT PRIMARY KEY,
  esco_id VARCHAR(100),
  role_name VARCHAR(255),
  description TEXT,
  source VARCHAR(20) DEFAULT 'ESCO'
);

CREATE TABLE general_skills (
  id INT AUTO_INCREMENT PRIMARY KEY,
  esco_skill_id VARCHAR(100),
  skill_name TEXT,
  cleaned_skill TEXT
);

CREATE TABLE role_general_skills (
  role_id INT,
  general_skill_id INT,
  importance FLOAT,
  PRIMARY KEY (role_id, general_skill_id),
  FOREIGN KEY (role_id) REFERENCES career_roles(id),
  FOREIGN KEY (general_skill_id) REFERENCES general_skills(id)
);


CREATE TABLE technical_skills (
  id INT AUTO_INCREMENT PRIMARY KEY,
  onet_id VARCHAR(100),
  skill_name VARCHAR(255),
  category VARCHAR(100),
  cleaned_skill TEXT
);

CREATE TABLE general_to_technical (
  general_skill_id INT,
  technical_skill_id INT,
  relevance_score FLOAT,
  PRIMARY KEY (general_skill_id, technical_skill_id),
  FOREIGN KEY (general_skill_id) REFERENCES general_skills(id),
  FOREIGN KEY (technical_skill_id) REFERENCES technical_skills(id)
);

-- Clear the database
SET FOREIGN_KEY_CHECKS = 0;

TRUNCATE TABLE role_general_skills;
TRUNCATE TABLE general_to_technical;
TRUNCATE TABLE career_roles;
TRUNCATE TABLE general_skills;
TRUNCATE TABLE technical_skills;

SET FOREIGN_KEY_CHECKS = 1;


-- 1. Check if we have roles
SELECT COUNT(*) as total_roles FROM career_roles;

-- 2. Check if we have skills
SELECT COUNT(*) as total_skills FROM general_skills;

-- 3. The "Ultimate Test": List a role and its essential skills
-- This proves the 'role_general_skills' table is working
SELECT r.role_name, s.skill_name 
FROM career_roles r
JOIN role_general_skills rgs ON r.id = rgs.role_id
JOIN general_skills s ON rgs.general_skill_id = s.id
WHERE r.role_name LIKE '%Developer%'
LIMIT 10;

-- Disable checks to allow truncation
SET FOREIGN_KEY_CHECKS = 0;

TRUNCATE TABLE technical_skills;

-- If you want to be extra safe, clear the mapping table too 
-- as it might point to old technical_skill IDs
TRUNCATE TABLE general_to_technical;

SET FOREIGN_KEY_CHECKS = 1;