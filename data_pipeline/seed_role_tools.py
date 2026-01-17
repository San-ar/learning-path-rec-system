import mysql.connector
from typing import cast, Tuple, List, Set, Any, Dict

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="career_recommender"
)
cursor = db.cursor(buffered=True)

print("MANUAL TECHNICAL SKILLS MAPPING")
print("Based on industry certifications and job market analysis")

# COMPLETE ROLE → TECHNICAL SKILLS MAPPING
ROLE_TECH_MAPPING = {
    # ========== SECURITY & CYBERSECURITY ==========
    "ICT security administrator": [
        "Wireshark", "Nmap", "Splunk", "Windows Server", "Linux", 
        "Active Directory", "Firewall", "VPN", "SIEM", "PowerShell"
    ],
    "digital forensics expert": [
        "EnCase", "FTK", "Autopsy", "Volatility", "Wireshark", 
        "Python", "Linux", "X-Ways Forensics", "Sleuth Kit", "Forensic Toolkit"
    ],
    "cyber incident responder": [
        "Splunk", "SIEM", "Wireshark", "Volatility", "Python", 
        "EDR", "IDS/IPS", "Threat Intelligence", "Firewall", "Nmap"
    ],
    "ethical hacker": [
        "Metasploit", "Burp Suite", "Nmap", "Wireshark", "Kali Linux", 
        "Python", "SQL Injection", "John the Ripper", "Aircrack-ng", "Nessus"
    ],
    "chief ICT security officer": [
        "ISO 27001", "NIST Framework", "Risk Management", "GRC Software", 
        "SIEM", "Splunk", "Security Architecture", "Compliance Tools", "PowerShell"
    ],
    "cybersecurity risk manager": [
        "Risk Assessment Tools", "ISO 27001", "NIST Framework", "GRC Software",
        "Qualys", "Nessus", "Splunk", "Python", "Excel"
    ],
    "embedded systems security engineer": [
        "C", "C++", "ARM Assembly", "JTAG", "IDA Pro", 
        "Ghidra", "Hardware Security", "Cryptography", "RTOS", "Embedded Linux"
    ],
    "ICT disaster recovery analyst": [
        "Veeam", "Disaster Recovery Planning", "VMware", 
        "Azure Site Recovery", "AWS Backup", "PowerShell", "Linux", "Backup Software"
    ],
    "ICT resilience manager": [
        "Business Continuity", "Risk Management", "ITIL", "ServiceNow",
        "Nagios", "SolarWinds", "Excel", "PowerPoint", "Monitoring Tools"
    ],
    "IT auditor": [
        "ACL", "IDEA", "SQL", "Excel", "Audit Frameworks", 
        "COBIT", "ISO Standards", "Risk Assessment", "Python", "Power BI"
    ],
    "ICT auditor manager": [
        "ACL", "IDEA", "SQL", "Excel", "COBIT", 
        "ISO 27001", "Risk Assessment", "Audit Software", "Power BI"
    ],
    
    # ========== CLOUD & INFRASTRUCTURE ==========
    "cloud architect": [
        "AWS", "Azure", "Google Cloud", "Terraform", "Kubernetes", 
        "Docker", "CloudFormation", "Python", "Jenkins", "Networking"
    ],
    "cloud engineer": [
        "AWS", "Azure", "Docker", "Kubernetes", "Terraform", 
        "Python", "Linux", "CI/CD", "Jenkins", "Ansible"
    ],
    "cloud software developer": [
        "AWS", "Azure", "Python", "Java", "Docker", 
        "Kubernetes", "REST API", "Microservices", "Git", "CI/CD"
    ],
    "cloud DevOps engineer": [
        "Docker", "Kubernetes", "Jenkins", "Terraform", "Ansible", 
        "AWS", "Azure", "Git", "Python", "CI/CD"
    ],
    "cloud identity manager": [
        "Azure AD", "Okta", "SAML", "OAuth", "IAM", 
        "SSO", "MFA", "LDAP", "PowerShell", "Security Protocols"
    ],
    
    # ========== DATA & AI ==========
    "data scientist": [
        "Python", "R", "SQL", "Pandas", "NumPy", 
        "scikit-learn", "TensorFlow", "Jupyter", "Tableau", "Excel"
    ],
    "data engineer": [
        "Python", "SQL", "Apache Spark", "Hadoop", "Airflow", 
        "ETL", "AWS", "Snowflake", "PostgreSQL", "Kafka"
    ],
    "data analyst": [
        "SQL", "Python", "Excel", "Tableau", "Power BI", 
        "R", "Statistics", "Data Visualization", "Pandas", "Google Analytics"
    ],
    "data warehouse designer": [
        "SQL", "Snowflake", "Redshift", "ETL", "Data Modeling", 
        "Python", "Talend", "SSIS", "PostgreSQL", "dbt"
    ],
    "data quality specialist": [
        "SQL", "Python", "Data Profiling", "Talend", "Informatica", 
        "Excel", "ETL", "Data Validation", "Pandas", "Great Expectations"
    ],
    "artificial intelligence engineer": [
        "Python", "TensorFlow", "PyTorch", "scikit-learn", 
        "Deep Learning", "NLP", "Computer Vision", "Jupyter", "Git"
    ],
    "computer vision engineer": [
        "Python", "OpenCV", "TensorFlow", "PyTorch", "YOLO", 
        "Deep Learning", "Image Processing", "Jupyter", "NumPy", "Git"
    ],
    "knowledge engineer": [
        "Python", "NLP", "Knowledge Graphs", "Ontologies", "SPARQL", 
        "Machine Learning", "Neo4j", "RDF", "Protégé", "SQL"
    ],
    
    # ========== SOFTWARE DEVELOPMENT ==========
    "software developer": [
        "Python", "Java", "JavaScript", "Git", "SQL", 
        "REST API", "HTML", "CSS", "Docker", "CI/CD"
    ],
    "software architect": [
        "UML", "Design Patterns", "Microservices", "REST API", "Java", 
        "Python", "Cloud Architecture", "Docker", "Kubernetes", "Git"
    ],
    "software analyst": [
        "UML", "Requirements Analysis", "JIRA", "Agile", "SQL", 
        "Documentation", "Git", "Business Analysis", "Visio"
    ],
    "ICT application developer": [
        "Java", "Python", "JavaScript", "SQL", "Git", 
        "REST API", "HTML", "CSS", "React", "Docker"
    ],
    "ICT system developer": [
        "Python", "Java", "C#", "SQL", "Linux", 
        "System Architecture", "API Development", "Git", "Docker", "CI/CD"
    ],
    "embedded systems software developer": [
        "C", "C++", "Embedded C", "RTOS", "Microcontrollers", 
        "ARM", "Debugging Tools", "JTAG", "Git", "Assembly"
    ],
    "embedded system designer": [
        "C", "C++", "RTOS", "Microcontrollers", "Circuit Design", 
        "ARM", "PCB Design", "Embedded Linux", "FPGA", "CAD Tools"
    ],
    "numerical tool and process control programmer": [
        "MATLAB", "Python", "C++", "LabVIEW", "PLC Programming", 
        "SCADA", "Control Systems", "Simulink", "Git", "Industrial Automation"
    ],
    "industrial mobile devices software developer": [
        "Android", "iOS", "Java", "Kotlin", "Swift", 
        "Industrial Protocols", "Bluetooth", "RFID", "Git", "Mobile Development"
    ],
    
    # ========== WEB & MOBILE ==========
    "web developer": [
        "JavaScript", "HTML", "CSS", "React", "Node.js", 
        "Git", "SQL", "REST API", "TypeScript", "Webpack"
    ],
    "mobile application developer": [
        "React Native", "Flutter", "Swift", "Kotlin", "Java", 
        "Android Studio", "Xcode", "Git", "REST API", "Firebase"
    ],
    "user interface developer": [
        "HTML", "CSS", "JavaScript", "React", "Vue.js", 
        "Figma", "Responsive Design", "Git", "Webpack", "SASS"
    ],
    "user interface designer": [
        "Figma", "Adobe XD", "Sketch", "HTML", "CSS", 
        "Prototyping", "Wireframing", "User Research", "Responsive Design", "InVision"
    ],
    "user experience analyst": [
        "Figma", "User Research", "Usability Testing", "Analytics", "Hotjar", 
        "Google Analytics", "A/B Testing", "Wireframing", "Personas", "Journey Mapping"
    ],
    "web content manager": [
        "WordPress", "CMS", "HTML", "CSS", "SEO", 
        "Google Analytics", "Content Strategy", "Copywriting", "Photoshop", "Social Media"
    ],
    "search engine optimisation expert": [
        "Google Analytics", "SEMrush", "Ahrefs", "Google Search Console", "SEO Tools", 
        "Keyword Research", "HTML", "JavaScript", "Python", "Data Analysis"
    ],
    
    # ========== BLOCKCHAIN & EMERGING TECH ==========
    "blockchain developer": [
        "Solidity", "Ethereum", "Web3.js", "Smart Contracts", "JavaScript", 
        "Truffle", "Hardhat", "Git", "Cryptography", "Node.js"
    ],
    "blockchain architect": [
        "Ethereum", "Hyperledger", "Solidity", "Smart Contracts", "Consensus Algorithms", 
        "Distributed Systems", "Cryptography", "Architecture Design", "Web3", "Security"
    ],
    "IoT developer": [
        "Python", "C", "Arduino", "Raspberry Pi", "MQTT", 
        "IoT Protocols", "Embedded Systems", "Cloud Platforms", "Sensors", "Linux"
    ],
    
    # ========== GAMES ==========
    "digital games developer": [
        "Unity", "C#", "Unreal Engine", "C++", "Game Physics", 
        "3D Modeling", "Git", "Blender", "Animation", "Game Design"
    ],
    "digital games tester": [
        "Test Plans", "Bug Tracking", "JIRA", "Game Testing", "QA Methodologies", 
        "Automation", "Performance Testing", "Unity", "Documentation", "Excel"
    ],
    
    # ========== DATABASE ==========
    "database administrator": [
        "SQL", "MySQL", "PostgreSQL", "Oracle", "MongoDB", 
        "Backup & Recovery", "Performance Tuning", "Linux", "PowerShell", "Replication"
    ],
    "database designer": [
        "SQL", "ER Diagrams", "Data Modeling", "PostgreSQL", "MySQL", 
        "Normalization", "Database Design", "UML", "NoSQL", "Performance Optimization"
    ],
    "database developer": [
        "SQL", "Stored Procedures", "T-SQL", "PL/SQL", "MySQL", 
        "PostgreSQL", "Query Optimization", "ETL", "Git", "Python"
    ],
    "database integrator": [
        "SQL", "ETL", "Data Integration", "APIs", "Python", 
        "SSIS", "Talend", "Kafka", "REST API", "Data Migration"
    ],
    
    # ========== NETWORK ==========
    "ICT network administrator": [
        "Cisco", "Routing", "Switching", "Firewall", "VPN", 
        "Network Monitoring", "Windows Server", "Linux", "TCP/IP", "DNS"
    ],
    "ICT network engineer": [
        "Cisco", "Routing", "Switching", "BGP", "OSPF", 
        "Network Design", "Firewall", "VPN", "Wireshark", "Python"
    ],
    "ICT network architect": [
        "Network Design", "Cisco", "SD-WAN", "Cloud Networking", "Security Architecture", 
        "Load Balancing", "BGP", "OSPF", "Network Automation", "Python"
    ],
    
    # ========== SYSTEM & INTEGRATION ==========
    "ICT system administrator": [
        "Windows Server", "Linux", "Active Directory", "PowerShell", "Bash", 
        "VMware", "Backup Solutions", "Monitoring", "Networking", "Security"
    ],
    "ICT system analyst": [
        "Requirements Analysis", "UML", "Systems Design", "SQL", "JIRA", 
        "Business Process", "Documentation", "Agile", "Visio", "Git"
    ],
    "ICT system architect": [
        "System Design", "Enterprise Architecture", "UML", "Cloud Architecture", "Microservices", 
        "API Design", "Security Architecture", "TOGAF", "Documentation", "DevOps"
    ],
    "system configurator": [
        "Configuration Management", "Ansible", "Puppet", "Chef", "PowerShell", 
        "Linux", "Windows Server", "Scripting", "Git", "Documentation"
    ],
    "integration engineer": [
        "API Development", "REST", "SOAP", "Microservices", "Python", 
        "Java", "Integration Platforms", "Kafka", "MuleSoft", "Git"
    ],
    "ICT system integration consultant": [
        "Integration Architecture", "APIs", "Middleware", "MuleSoft", "ESB", 
        "System Design", "Python", "Java", "Project Management", "Documentation"
    ],
    "enterprise architect": [
        "TOGAF", "Enterprise Architecture", "Business Process", "UML", "Cloud Architecture", 
        "Strategic Planning", "Microservices", "API Design", "Documentation", "Governance"
    ],
    
    # ========== TESTING & QA ==========
    "software tester": [
        "Test Automation", "Selenium", "JIRA", "Test Plans", "Bug Tracking", 
        "Manual Testing", "API Testing", "SQL", "Python", "Postman"
    ],
    "ICT test analyst": [
        "Test Planning", "Test Cases", "Selenium", "JIRA", "SQL", 
        "Test Management", "Automation", "Defect Tracking", "API Testing", "Performance Testing"
    ],
    "ICT usability tester": [
        "Usability Testing", "User Research", "Test Plans", "Analytics", "Heuristic Evaluation", 
        "A/B Testing", "Wireframing", "Documentation", "Survey Tools", "Video Recording"
    ],
    "ICT accessibility tester": [
        "WCAG", "Screen Readers", "Accessibility Testing", "HTML", "ARIA", 
        "JAWS", "NVDA", "axe DevTools", "Lighthouse", "Manual Testing"
    ],
    "ICT system tester": [
        "System Testing", "Integration Testing", "Test Plans", "JIRA", "SQL", 
        "Automation", "Performance Testing", "Load Testing", "Bug Tracking", "Documentation"
    ],
    "ICT integration tester": [
        "Integration Testing", "API Testing", "Postman", "SOAP UI", "Test Automation", 
        "CI/CD", "Jenkins", "SQL", "Python", "REST API"
    ],
    "ICT quality assurance manager": [
        "QA Methodologies", "Test Management", "JIRA", "Test Automation", "Selenium", 
        "Performance Testing", "Metrics", "Agile", "Risk Management", "Leadership"
    ],
    
    # ========== BUSINESS & MANAGEMENT ==========
    "ICT business analyst": [
        "Requirements Analysis", "Business Process", "JIRA", "SQL", "UML", 
        "Agile", "Stakeholder Management", "Documentation", "Excel", "Visio"
    ],
    "ICT business analysis manager": [
        "Business Analysis", "Requirements Management", "Stakeholder Management", "Agile", "JIRA", 
        "Strategic Planning", "Process Improvement", "Documentation", "Leadership", "Excel"
    ],
    "ICT consultant": [
        "IT Strategy", "Business Analysis", "Project Management", "Cloud Solutions", "Documentation", 
        "Stakeholder Management", "Change Management", "Technical Writing", "Excel", "PowerPoint"
    ],
    "ICT research consultant": [
        "Research Methodologies", "Data Analysis", "Python", "R", "Technical Writing", 
        "Statistics", "Literature Review", "Excel", "Survey Design", "Presentation"
    ],
    "green ICT consultant": [
        "Sustainability Analysis", "Energy Efficiency", "IT Asset Management", "Carbon Footprint", "Lifecycle Assessment", 
        "Green IT Standards", "Data Center Optimization", "Excel", "Reporting", "Audit Tools"
    ],
    
    # ========== CONFIGURATION & CHANGE ==========
    "ICT application configurator": [
        "Application Configuration", "Scripting", "PowerShell", "Bash", "XML", 
        "Configuration Management", "Documentation", "Testing", "Git", "YAML"
    ],
    "ICT change and configuration manager": [
        "Change Management", "ITIL", "ServiceNow", "Configuration Management", "Git", 
        "Automation", "Documentation", "Risk Assessment", "Compliance", "Process Management"
    ],
    "ICT capacity planner": [
        "Capacity Planning", "Performance Monitoring", "Forecasting", "Excel", "SQL", 
        "System Monitoring", "SolarWinds", "Nagios", "Resource Management", "Data Analysis"
    ],
    
    # ========== COMPUTER SCIENCE ==========
    "computer scientist": [
        "Python", "Java", "C++", "Algorithms", "Data Structures", 
        "Machine Learning", "Research", "Mathematics", "Git", "LaTeX"
    ]
}

# Validation (Normalized)
print(f"\nValidating mappings...")
cursor.execute("SELECT role_name FROM career_roles")
# Normalize strings from DB to avoid hidden whitespace mismatch
raw_db_rows = cast(List[Tuple[str]], cursor.fetchall())
db_roles: Set[str] = {str(row[0]).strip() for row in raw_db_rows}

# Standardize dictionary keys
mapped_roles: Set[str] = {k.strip() for k in ROLE_TECH_MAPPING.keys()}

missing = db_roles - mapped_roles
extra = mapped_roles - db_roles

print(f"Roles in database: {len(db_roles)}")
print(f"Roles mapped: {len(mapped_roles)}")

if not missing and not extra:
    print("All 75 roles matched perfectly!")
else:
    if missing:
        print(f"Missing in Script ({len(missing)})")
    if extra:
        print(f"Extra in Script ({len(extra)})")

# PART 4: Execution
response = input("\nType 'yes' to clear old data and insert new mappings: ").strip().lower()
if response != 'yes':
    print("Aborted.")
    cursor.close()
    db.close()
    exit()

print("\nInserting mappings...")
cursor.execute("TRUNCATE TABLE role_technical_skills")

for role_name, tech_skills in ROLE_TECH_MAPPING.items():
    # Get ID for current role
    cursor.execute("SELECT id FROM career_roles WHERE role_name = %s", (role_name.strip(),))
    res = cast(Tuple[int], cursor.fetchone())
    if not res: continue
    role_id = res[0]

    for tool in tech_skills:
        # Insert/Update tool in technical_skills
        cursor.execute("""
            INSERT IGNORE INTO technical_skills (onet_id, skill_name, category, cleaned_skill)
            VALUES (%s, %s, %s, %s)
        """, (f"manual_{tool.lower().replace(' ', '_')}", tool, "Expert Curated", tool.lower()))

        # Get the tool's ID
        cursor.execute("SELECT id FROM technical_skills WHERE skill_name = %s", (tool,))
        skill_res = cast(Tuple[int], cursor.fetchone())
        if not skill_res: continue
        skill_id = skill_res[0]

        # Create the direct mapping (Role -> Tool)
        cursor.execute("""
            INSERT IGNORE INTO role_technical_skills (role_id, technical_skill_id, relevance_score)
            VALUES (%s, %s, %s)
        """, (role_id, skill_id, 0.95))

db.commit()
print("Hybrid Mapping Complete.")

# Test Verification
print("\nVerification Results:")
for test_role in ["web developer", "data scientist", "ethical hacker"]:
    print(f"\n{test_role.upper()}:")
    cursor.execute("""
        SELECT ts.skill_name 
        FROM technical_skills ts
        JOIN role_technical_skills rts ON ts.id = rts.technical_skill_id
        JOIN career_roles cr ON rts.role_id = cr.id
        WHERE cr.role_name = %s
    """, (test_role,))
    for (skill,) in cursor.fetchall():
        print(f"  - {skill}")

cursor.close()
db.close()