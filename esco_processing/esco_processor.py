import pandas as pd
import json
from collections import defaultdict
import re

class ESCOProcessor:
    """
    Processes ESCO CSV files to create a hybrid skill taxonomy
    combining ESCO's semantic structure with specific tech stack mappings
    """
    
    def __init__(self):
        self.occupations = None
        self.skills = None
        self.relations = None
        
        # Tech stack mappings for common ESCO skills
        # This is where we bridge ESCO's abstract skills to concrete technologies
        self.tech_stack_mappings = {
            # Data & Analytics
            "data engineering": {
                "tech": ["Python", "SQL", "Apache Spark", "Apache Kafka", "Airflow", "dbt"],
                "use_case": "Building data pipelines and ETL processes"
            },
            "statistics": {
                "tech": ["Python", "R", "NumPy", "SciPy", "Pandas", "Statsmodels"],
                "use_case": "Statistical analysis and modeling"
            },
            "scientific computing": {
                "tech": ["Python", "NumPy", "SciPy", "Matplotlib", "Jupyter"],
                "use_case": "Numerical computation and scientific analysis"
            },
            "perform data analysis": {
                "tech": ["Python", "Pandas", "SQL", "Tableau", "Power BI", "Excel"],
                "use_case": "Analyzing and interpreting data"
            },
            "data mining": {
                "tech": ["Python", "scikit-learn", "Pandas", "SQL", "R"],
                "use_case": "Extracting patterns from large datasets"
            },
            
            # Machine Learning & AI
            "machine learning methods": {
                "tech": ["Python", "TensorFlow", "PyTorch", "scikit-learn", "Keras"],
                "use_case": "Building and training ML models"
            },
            "artificial intelligence": {
                "tech": ["Python", "TensorFlow", "PyTorch", "OpenAI API", "LangChain"],
                "use_case": "Developing AI systems and applications"
            },
            "deep learning": {
                "tech": ["Python", "TensorFlow", "PyTorch", "Keras", "CUDA"],
                "use_case": "Neural networks and deep learning"
            },
            "neural networks": {
                "tech": ["Python", "TensorFlow", "PyTorch", "Keras"],
                "use_case": "Building neural network architectures"
            },
            "natural language processing": {
                "tech": ["Python", "NLTK", "spaCy", "Transformers", "GPT", "BERT"],
                "use_case": "Text processing and language understanding"
            },
            "computer vision": {
                "tech": ["Python", "OpenCV", "TensorFlow", "PyTorch", "YOLO"],
                "use_case": "Image and video processing"
            },
            
            # Software Development
            "software development": {
                "tech": ["Python", "JavaScript", "Java", "C++", "Git", "GitHub"],
                "use_case": "Building software applications"
            },
            "object-oriented programming": {
                "tech": ["Python", "Java", "C++", "C#", "TypeScript"],
                "use_case": "OOP design patterns and principles"
            },
            "web development": {
                "tech": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Django", "Flask"],
                "use_case": "Creating web applications"
            },
            "mobile app development": {
                "tech": ["Swift", "Kotlin", "React Native", "Flutter", "iOS", "Android"],
                "use_case": "Building mobile applications"
            },
            "use software libraries": {
                "tech": ["Python", "npm", "pip", "Maven", "Gradle"],
                "use_case": "Leveraging existing code libraries"
            },
            "version control systems": {
                "tech": ["Git", "GitHub", "GitLab", "Bitbucket"],
                "use_case": "Code versioning and collaboration"
            },
            
            # Database & Backend
            "database management": {
                "tech": ["SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis"],
                "use_case": "Managing and optimizing databases"
            },
            "SQL": {
                "tech": ["PostgreSQL", "MySQL", "SQL Server", "Oracle", "SQLite"],
                "use_case": "Database querying and management"
            },
            "NoSQL": {
                "tech": ["MongoDB", "Cassandra", "Redis", "DynamoDB", "Neo4j"],
                "use_case": "Non-relational database systems"
            },
            
            # Cloud & DevOps
            "cloud computing": {
                "tech": ["AWS", "Azure", "Google Cloud", "Docker", "Kubernetes"],
                "use_case": "Cloud infrastructure and services"
            },
            "DevOps": {
                "tech": ["Docker", "Kubernetes", "Jenkins", "GitLab CI", "Terraform", "Ansible"],
                "use_case": "Automation and deployment"
            },
            "containerisation": {
                "tech": ["Docker", "Kubernetes", "Podman", "Container Registry"],
                "use_case": "Application containerization"
            },
            "infrastructure as code": {
                "tech": ["Terraform", "CloudFormation", "Ansible", "Pulumi"],
                "use_case": "Automated infrastructure provisioning"
            },
            
            # Security
            "cybersecurity": {
                "tech": ["Kali Linux", "Wireshark", "Metasploit", "Burp Suite", "Nmap"],
                "use_case": "Security testing and protection"
            },
            "network security": {
                "tech": ["Firewalls", "VPN", "IDS/IPS", "SSL/TLS", "Network Protocols"],
                "use_case": "Securing network infrastructure"
            },
            "cryptography": {
                "tech": ["OpenSSL", "Python cryptography", "AES", "RSA", "PKI"],
                "use_case": "Encryption and secure communications"
            },
            
            # Testing & Quality
            "software testing": {
                "tech": ["Selenium", "Jest", "PyTest", "JUnit", "Postman"],
                "use_case": "Testing and quality assurance"
            },
            "test automation": {
                "tech": ["Selenium", "Cypress", "Jest", "PyTest", "TestNG"],
                "use_case": "Automated testing frameworks"
            },
            
            # UI/UX
            "user interface design": {
                "tech": ["Figma", "Sketch", "Adobe XD", "HTML", "CSS", "React"],
                "use_case": "Designing user interfaces"
            },
            "user experience design": {
                "tech": ["Figma", "Adobe XD", "Miro", "UserTesting", "Hotjar"],
                "use_case": "UX research and design"
            },
        }
        
        # Common CS/IT fields we want to focus on
        self.target_occupations = [
            "machine learning engineer",
            "data scientist",
            "software developer",
            "full stack developer",
            "web developer",
            "mobile application developer",
            "cybersecurity specialist",
            "devops engineer",
            "cloud architect",
            "database administrator",
            "network engineer",
            "artificial intelligence engineer",
            "business intelligence analyst",
            "system administrator",
            "ui/ux designer"
        ]
    
    def load_esco_files(self, occupations_path, skills_path, relations_path):
        """Load ESCO CSV files"""
        print("Loading ESCO files...")
        
        self.occupations = pd.read_csv(occupations_path)
        self.skills = pd.read_csv(skills_path)
        self.relations = pd.read_csv(relations_path)
        
        print(f"✓ Loaded {len(self.occupations)} occupations")
        print(f"✓ Loaded {len(self.skills)} skills")
        print(f"✓ Loaded {len(self.relations)} relations")
    
    def filter_cs_it_occupations(self):
        """Filter occupations to CS/IT fields"""
        print("\nFiltering CS/IT occupations...")
        
        # ESCO uses ISCO codes - CS/IT jobs are typically in group 25 (ICT professionals)
        cs_it_occupations = self.occupations[
            (self.occupations['conceptUri'].str.contains('25', na=False)) |
            (self.occupations['preferredLabel'].str.lower().str.contains('software|data|developer|engineer|programmer|analyst|ict|computer|cyber|cloud|web|mobile', na=False))
        ]
        
        print(f"✓ Found {len(cs_it_occupations)} CS/IT occupations")
        return cs_it_occupations
    
    def get_skills_for_occupation(self, occupation_uri):
        """Get all skills required for a specific occupation"""
        # Filter relations for this occupation
        occupation_relations = self.relations[
            self.relations['occupationUri'] == occupation_uri
        ]
        
        # Get skill details
        skill_uris = occupation_relations['skillUri'].tolist()
        occupation_skills = self.skills[
            self.skills['conceptUri'].isin(skill_uris)
        ]
        
        return occupation_skills
    
    def map_skill_to_tech(self, skill_name):
        """Map ESCO skill to specific technologies"""
        skill_lower = skill_name.lower()
        
        # Direct match
        if skill_lower in self.tech_stack_mappings:
            return self.tech_stack_mappings[skill_lower]
        
        # Partial match
        for esco_skill, tech_info in self.tech_stack_mappings.items():
            if esco_skill in skill_lower or skill_lower in esco_skill:
                return tech_info
        
        # No mapping found - return generic programming
        return {
            "tech": ["Programming Fundamentals"],
            "use_case": "General technical skill"
        }
    
    def create_field_database(self, output_file='career_fields_database.json'):
        """Create the complete career fields database"""
        print("\n" + "="*50)
        print("Creating Career Fields Database")
        print("="*50)
        
        cs_it_occupations = self.filter_cs_it_occupations()
        
        fields_database = []
        
        for idx, occupation in cs_it_occupations.iterrows():
            occupation_name = occupation['preferredLabel']
            occupation_uri = occupation['conceptUri']
            
            print(f"\nProcessing: {occupation_name}")
            
            # Get skills for this occupation
            skills = self.get_skills_for_occupation(occupation_uri)
            
            if len(skills) == 0:
                print(f"  ⚠ No skills found, skipping...")
                continue
            
            # Process skills and map to tech
            esco_skills_list = []
            all_tech_stack = set()
            
            for _, skill in skills.iterrows():
                skill_name = skill['preferredLabel']
                tech_mapping = self.map_skill_to_tech(skill_name)
                
                esco_skills_list.append({
                    "skill_id": skill['conceptUri'],
                    "skill_name": skill_name,
                    "skill_type": skill.get('skillType', 'skill'),
                    "tech_stack": tech_mapping['tech'],
                    "use_case": tech_mapping['use_case']
                })
                
                all_tech_stack.update(tech_mapping['tech'])
            
            # Create field entry
            field_entry = {
                "field_id": occupation_uri.split('/')[-1],
                "field_name": occupation_name,
                "esco_uri": occupation_uri,
                "description": occupation.get('description', ''),
                "esco_skills": esco_skills_list[:15],  # Limit to top 15 skills
                "tech_stack_summary": sorted(list(all_tech_stack))[:20],  # Top 20 unique tech
                "total_skills": len(skills)
            }
            
            fields_database.append(field_entry)
            print(f"  ✓ Added {len(esco_skills_list)} skills")
            print(f"  ✓ Mapped to {len(all_tech_stack)} technologies")
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fields_database, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*50}")
        print(f"✓ Database saved to: {output_file}")
        print(f"✓ Total fields: {len(fields_database)}")
        print(f"{'='*50}")
        
        return fields_database
    
    def create_skills_database(self, output_file='skills_database.json'):
        """Create a standalone skills database with tech mappings"""
        print("\nCreating Skills Database...")
        
        skills_db = []
        
        for _, skill in self.skills.iterrows():
            skill_name = skill['preferredLabel']
            tech_mapping = self.map_skill_to_tech(skill_name)
            
            skills_db.append({
                "skill_id": skill['conceptUri'].split('/')[-1],
                "skill_name": skill_name,
                "esco_uri": skill['conceptUri'],
                "skill_type": skill.get('skillType', 'skill'),
                "description": skill.get('description', ''),
                "tech_stack": tech_mapping['tech'],
                "use_case": tech_mapping['use_case']
            })
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(skills_db, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Skills database saved to: {output_file}")
        print(f"✓ Total skills: {len(skills_db)}")
        
        return skills_db
    
    def create_curated_fields(self, output_file='curated_career_fields.json'):
        """
        Create a manually curated list of popular CS/IT fields
        This is what you'll use for your MVP/Phase 1
        """
        print("\nCreating Curated Career Fields for MVP...")
        
        curated_fields = [
            {
                "field_id": "ml_engineer",
                "field_name": "Machine Learning Engineer",
                "category": "AI & Data Science",
                "description": "Design, build, and deploy machine learning models and AI systems in production environments.",
                "avg_salary": 125000,
                "growth_outlook": "High Growth",
                "priority_skills": [
                    "Python", "TensorFlow", "PyTorch", "scikit-learn", 
                    "Statistics", "Machine Learning", "Deep Learning", "MLOps"
                ],
                "learning_topics": [
                    {
                        "topic": "Python for Machine Learning",
                        "description": "Master Python programming with NumPy, Pandas, and scikit-learn",
                        "duration_weeks": 6,
                        "suggested_courses": [
                            {
                                "title": "Python for Data Science and Machine Learning",
                                "provider": "Coursera",
                                "difficulty": "Beginner"
                            }
                        ]
                    },
                    {
                        "topic": "Machine Learning Foundations",
                        "description": "Learn supervised and unsupervised learning algorithms",
                        "duration_weeks": 12,
                        "suggested_courses": [
                            {
                                "title": "Machine Learning Specialization",
                                "provider": "Coursera - Stanford",
                                "difficulty": "Intermediate"
                            }
                        ]
                    },
                    {
                        "topic": "Deep Learning",
                        "description": "Neural networks, CNNs, RNNs, and transformers",
                        "duration_weeks": 12,
                        "suggested_courses": [
                            {
                                "title": "Deep Learning Specialization",
                                "provider": "Coursera - DeepLearning.AI",
                                "difficulty": "Advanced"
                            }
                        ]
                    }
                ]
            },
            {
                "field_id": "full_stack_dev",
                "field_name": "Full Stack Developer",
                "category": "Software Development",
                "description": "Build complete web applications managing both frontend and backend development.",
                "avg_salary": 95000,
                "growth_outlook": "Stable",
                "priority_skills": [
                    "JavaScript", "React", "Node.js", "HTML", "CSS", 
                    "SQL", "MongoDB", "Git", "RESTful APIs"
                ],
                "learning_topics": [
                    {
                        "topic": "Frontend Development",
                        "description": "HTML, CSS, JavaScript, React fundamentals",
                        "duration_weeks": 8,
                        "suggested_courses": [
                            {
                                "title": "Meta Front-End Developer",
                                "provider": "Coursera",
                                "difficulty": "Beginner"
                            }
                        ]
                    },
                    {
                        "topic": "Backend Development",
                        "description": "Node.js, Express, databases, and APIs",
                        "duration_weeks": 8,
                        "suggested_courses": [
                            {
                                "title": "Backend Development and APIs",
                                "provider": "freeCodeCamp",
                                "difficulty": "Intermediate"
                            }
                        ]
                    }
                ]
            },
            {
                "field_id": "data_scientist",
                "field_name": "Data Scientist",
                "category": "AI & Data Science",
                "description": "Analyze complex datasets, build predictive models, and derive actionable business insights.",
                "avg_salary": 115000,
                "growth_outlook": "High Growth",
                "priority_skills": [
                    "Python", "R", "SQL", "Statistics", "Machine Learning", 
                    "Data Visualization", "Pandas", "scikit-learn"
                ],
                "learning_topics": [
                    {
                        "topic": "Data Analysis with Python",
                        "description": "Pandas, NumPy, data cleaning and exploration",
                        "duration_weeks": 6,
                        "suggested_courses": [
                            {
                                "title": "Data Analysis with Python",
                                "provider": "Coursera - IBM",
                                "difficulty": "Beginner"
                            }
                        ]
                    },
                    {
                        "topic": "Statistical Analysis",
                        "description": "Probability, hypothesis testing, regression",
                        "duration_weeks": 8,
                        "suggested_courses": [
                            {
                                "title": "Statistics with Python Specialization",
                                "provider": "Coursera",
                                "difficulty": "Intermediate"
                            }
                        ]
                    }
                ]
            },
            {
                "field_id": "cybersecurity_analyst",
                "field_name": "Cybersecurity Analyst",
                "category": "Security",
                "description": "Protect systems and networks from cyber threats through monitoring, analysis, and response.",
                "avg_salary": 105000,
                "growth_outlook": "High Demand",
                "priority_skills": [
                    "Network Security", "Ethical Hacking", "Cryptography", 
                    "SIEM Tools", "Penetration Testing", "Linux", "Python"
                ],
                "learning_topics": [
                    {
                        "topic": "Network Security Fundamentals",
                        "description": "TCP/IP, firewalls, VPNs, network protocols",
                        "duration_weeks": 6,
                        "suggested_courses": [
                            {
                                "title": "Introduction to Cybersecurity",
                                "provider": "Coursera - NYU",
                                "difficulty": "Beginner"
                            }
                        ]
                    },
                    {
                        "topic": "Ethical Hacking",
                        "description": "Penetration testing and vulnerability assessment",
                        "duration_weeks": 12,
                        "suggested_courses": [
                            {
                                "title": "The Complete Ethical Hacking Course",
                                "provider": "Udemy",
                                "difficulty": "Intermediate"
                            }
                        ]
                    }
                ]
            },
            {
                "field_id": "devops_engineer",
                "field_name": "DevOps Engineer",
                "category": "Infrastructure",
                "description": "Automate and optimize software development lifecycle through CI/CD and infrastructure management.",
                "avg_salary": 110000,
                "growth_outlook": "High Demand",
                "priority_skills": [
                    "Docker", "Kubernetes", "Jenkins", "AWS", "Terraform", 
                    "Linux", "Git", "Python", "CI/CD"
                ],
                "learning_topics": [
                    {
                        "topic": "Linux System Administration",
                        "description": "Command line, shell scripting, system management",
                        "duration_weeks": 6,
                        "suggested_courses": [
                            {
                                "title": "Linux Fundamentals",
                                "provider": "Coursera",
                                "difficulty": "Beginner"
                            }
                        ]
                    },
                    {
                        "topic": "Container Orchestration",
                        "description": "Docker, Kubernetes, container management",
                        "duration_weeks": 8,
                        "suggested_courses": [
                            {
                                "title": "Docker and Kubernetes Complete Course",
                                "provider": "Udemy",
                                "difficulty": "Intermediate"
                            }
                        ]
                    }
                ]
            }
        ]
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(curated_fields, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Curated fields saved to: {output_file}")
        print(f"✓ Total curated fields: {len(curated_fields)}")
        
        return curated_fields
    
    def generate_summary_report(self):
        """Generate a summary report of the processed data"""
        print("\n" + "="*50)
        print("ESCO PROCESSING SUMMARY")
        print("="*50)
        print(f"Total Occupations: {len(self.occupations)}")
        print(f"Total Skills: {len(self.skills)}")
        print(f"Total Relations: {len(self.relations)}")
        print(f"Tech Stack Mappings: {len(self.tech_stack_mappings)}")
        print("="*50)


# Main execution function
def main():
    """
    Main function to process ESCO data
    """
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║     ESCO Career Fields Processor for LPRS            ║
    ║     Converting ESCO taxonomy to actionable data      ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Initialize processor
    processor = ESCOProcessor()
    
    # STEP 1: Load ESCO files
    # Replace these paths with your actual file locations
    processor.load_esco_files(
        occupations_path='occupations_en.csv',
        skills_path='skills_en.csv',
        relations_path='occupationSkillRelations_en.csv'
    )
    
    # STEP 2: Create full career fields database (from ESCO)
    print("\n[1/3] Creating full career fields database...")
    fields_db = processor.create_field_database('career_fields_database.json')
    
    # STEP 3: Create skills database
    print("\n[2/3] Creating skills database...")
    skills_db = processor.create_skills_database('skills_database.json')
    
    # STEP 4: Create curated fields for MVP (recommended for Phase 1)
    print("\n[3/3] Creating curated fields for MVP...")
    curated_db = processor.create_curated_fields('curated_career_fields.json')
    
    # STEP 5: Generate summary
    processor.generate_summary_report()
    
    print("\n✅ Processing complete!")
    print("\nGenerated files:")
    print("  1. career_fields_database.json    - Full ESCO-based database")
    print("  2. skills_database.json           - All skills with tech mappings")
    print("  3. curated_career_fields.json     - MVP-ready 5 fields (USE THIS!)")
    print("\n💡 For Phase 1, use 'curated_career_fields.json'")


if __name__ == "__main__":
    main()