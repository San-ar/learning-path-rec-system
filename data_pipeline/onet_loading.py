import pandas as pd
import mysql.connector
import re
import hashlib
from typing import cast, Tuple, Any

# -----------------------------
# 1. Database Connection
# -----------------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="career_recommender"
)

# buffered=True prevents "Unread result found" issues
cursor = db.cursor(buffered=True)

# -----------------------------
# 2. Text Cleaning Utilities
# -----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def make_onet_id(soc_code: str, skill_name: str) -> str:
    """
    O*NET does not provide unique IDs per tool,
    so we generate a stable hash-based ID.
    """
    return hashlib.md5(f"{soc_code}:{skill_name}".encode()).hexdigest()

# -----------------------------
# 3. Load O*NET Technology Skills
# -----------------------------
try:
    onet_data = pd.read_csv(
        "./raw_data/Technology Skills.txt",
        sep="\t",
        encoding="utf-8",
        on_bad_lines="skip"
    )
    print("O*NET Technology Skills loaded successfully (utf-8).")
except UnicodeDecodeError:
    onet_data = pd.read_csv(
        "./raw_data/Technology Skills.txt",
        sep="\t",
        encoding="latin1",
        on_bad_lines="skip"
    )
    print("O*NET Technology Skills loaded successfully (latin1).")

# -----------------------------
# 4. Filter Keywords (CS / IT only)
# -----------------------------
tech_keywords = [
    "software", "programming", "database", "cloud",
    "machine learning", "data", "framework",
    "platform", "api", "network", "security",
    "devops", "web", "mobile"
]

# -----------------------------
# 5. Insert Technical Skills
# -----------------------------
inserted = 0

for _, row in onet_data.iterrows():
    skill_name = str(row.get("Example", "")).strip()
    category = str(row.get("Commodity Title", "")).strip()
    soc_code = str(row.get("O*NET-SOC Code", "")).strip()

    # Basic validation
    if not skill_name or skill_name.lower() == "nan":
        continue

    # Filter non-IT tools
    if not any(k in category.lower() for k in tech_keywords):
        continue

    cleaned_skill = clean_text(skill_name)
    onet_id = make_onet_id(soc_code, skill_name)

    cursor.execute(
        """
        INSERT IGNORE INTO technical_skills
            (onet_id, skill_name, category, cleaned_skill)
        VALUES (%s, %s, %s, %s)
        """,
        (onet_id, skill_name, category, cleaned_skill)
    )

    inserted += 1

# -----------------------------
# 6. Commit and Verify
# -----------------------------
db.commit()

cursor.execute("SELECT COUNT(*) FROM technical_skills")
result = cast(Tuple[Any, ...], cursor.fetchone())

if result:
    print(f"Success! Total technical skills in database: {result[0]}")
    print(f"Inserted in this run: {inserted}")

cursor.close()
db.close()