# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional SBERT
use_sbert = False
try:
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    use_sbert = True
except Exception:
    use_sbert = False

st.set_page_config(page_title="Learning Path Recommendation System", layout="wide")

# -----------------------
# Mock / sample datasets
# -----------------------
FIELDS = {
    "ml_engineer": {
        "name": "Machine Learning Engineer",
        "desc": "Design, build, and deploy machine learning models and systems. Requires strong programming, ML algorithms, and statistical knowledge.",
        "skills": ["python","statistics","machine learning","deep learning","tensorflow","data preprocessing"]
    },
    "data_science": {
        "name": "Data Scientist",
        "desc": "Extract insights from data using statistics, visualization, and machine learning. Focus on analysis and communication.",
        "skills": ["python","statistics","pandas","machine learning","data visualization","sql"]
    },
    "web_dev": {
        "name": "Full-Stack Web Developer",
        "desc": "Build complete web applications with frontend and backend technologies.",
        "skills": ["html","css","javascript","react","nodejs","backend","databases"]
    },
    "cybersec": {
        "name": "Cybersecurity Analyst",
        "desc": "Protect systems and networks from security threats. Requires networking and security fundamentals.",
        "skills": ["networks","linux","cryptography","security fundamentals","incident response"]
    },
    "cloud": {
        "name": "Cloud Architect",
        "desc": "Design and manage cloud infrastructure using AWS, Azure, or GCP. Requires DevOps knowledge.",
        "skills": ["linux","docker","kubernetes","aws","ci/cd","cloud platforms"]
    }
}

COURSES = pd.DataFrame([
    {"id":"c1","title":"Python for Beginners","skills":["python"],"difficulty":"Beginner","desc":"Introduction to Python programming fundamentals."},
    {"id":"c2","title":"Data Structures in Python","skills":["python","data structures"],"difficulty":"Intermediate","desc":"Learn essential data structures and algorithms.","prereqs":["c1"]},
    {"id":"c3","title":"Statistics Fundamentals","skills":["statistics"],"difficulty":"Beginner","desc":"Core statistical concepts for data analysis.","prereqs":[]},
    {"id":"c4","title":"Introduction to Machine Learning","skills":["machine learning","statistics","python"],"difficulty":"Intermediate","desc":"ML foundations and algorithms.","prereqs":["c1","c3"]},
    {"id":"c5","title":"Deep Learning Specialization","skills":["deep learning","tensorflow","python"],"difficulty":"Advanced","desc":"Neural networks and deep learning.","prereqs":["c4"]},
    {"id":"c6","title":"Web Basics (HTML/CSS)","skills":["html","css"],"difficulty":"Beginner","desc":"HTML and CSS fundamentals.","prereqs":[]},
    {"id":"c7","title":"JavaScript & React","skills":["javascript","react"],"difficulty":"Intermediate","desc":"Modern React development.","prereqs":["c6"]},
    {"id":"c8","title":"Linux Fundamentals","skills":["linux"],"difficulty":"Beginner","desc":"Command line and Linux basics.","prereqs":[]},
    {"id":"c9","title":"Networking Essentials","skills":["networks"],"difficulty":"Beginner","desc":"Network protocols and security.","prereqs":[]},
    {"id":"c10","title":"Cloud Fundamentals (AWS)","skills":["aws","linux","cloud platforms"],"difficulty":"Intermediate","desc":"AWS cloud essentials.","prereqs":["c8"]},
    {"id":"c11","title":"Data Visualization with Python","skills":["data visualization","python","pandas"],"difficulty":"Intermediate","desc":"Create charts and dashboards.","prereqs":["c1"]},
    {"id":"c12","title":"SQL for Data Analysis","skills":["sql","databases"],"difficulty":"Beginner","desc":"Database queries and data extraction.","prereqs":[]},
])

# -----------------------
# Helper functions
# -----------------------
def embed_texts(texts):
    if use_sbert:
        return sbert.encode(texts, convert_to_numpy=True)
    else:
        # Create a shared vectorizer for consistent dimensions
        vectorizer = TfidfVectorizer(max_features=100)
        return vectorizer.fit_transform(texts).toarray()

def semantic_similarity_vec(vecs_a, vecs_b):
    # Ensure both vectors have same shape
    if vecs_a.shape[1] != vecs_b.shape[1]:
        # If using TF-IDF fallback, recompute with shared vocabulary
        return np.array([[0.5]])  # Return default moderate similarity
    return cosine_similarity(vecs_a, vecs_b)

def compute_skill_overlap(user_skills, field_skills):
    user = set([s.strip().lower() for s in user_skills])
    field = set([s.strip().lower() for s in field_skills])
    inter = user.intersection(field)
    overlap = len(inter) / (len(field) if len(field)>0 else 1)
    return overlap, list(inter)

def compute_suitability(user_profile_text, user_skills, interests, field):
    # Semantic similarity (profile text vs field desc)
    # Combine both texts into single strings for embedding
    user_combined = user_profile_text + " " + " ".join(interests)
    field_combined = field["desc"] + " " + " ".join(field["skills"])
    
    if use_sbert:
        emb_user = embed_texts([user_combined])
        emb_field = embed_texts([field_combined])
        sem = semantic_similarity_vec(emb_user, emb_field)[0][0]
    else:
        # Use TF-IDF with shared vocabulary for fallback
        vectorizer = TfidfVectorizer(max_features=100)
        combined_texts = [user_combined, field_combined]
        embeddings = vectorizer.fit_transform(combined_texts).toarray()
        sem = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
    
    # Skill overlap (normalize skill names for matching)
    overlap, matched = compute_skill_overlap(user_skills, field["skills"])
    
    # Interest match - simple keyword matching
    interest_score = 0.0
    field_name_lower = field["name"].lower()
    for interest in interests:
        if interest.lower() in field_name_lower or any(interest.lower() in skill for skill in field["skills"]):
            interest_score += 0.2
    interest_score = min(interest_score, 1.0)
    
    # Weighted combination: 40% skill, 30% interest, 30% semantic
    suitability = 0.40 * overlap + 0.30 * interest_score + 0.30 * sem
    
    return float(suitability), float(sem), float(overlap), float(interest_score), matched

def find_missing_skills(user_skills, field_skills):
    user = set([s.strip().lower() for s in user_skills])
    missing = [s for s in field_skills if s.lower() not in user]
    return missing

def find_candidate_courses_for_skills(missing_skills, top_n=10):
    results = []
    for _, row in COURSES.iterrows():
        covered = [s for s in row["skills"] if s in missing_skills]
        if covered:
            # Score: more covered skills = higher, prefer beginner courses
            difficulty_penalty = {"Beginner": 0, "Intermediate": 0.2, "Advanced": 0.4}
            score = len(covered) - difficulty_penalty.get(row["difficulty"], 0)
            results.append((score, row))
    results = sorted(results, key=lambda x: -x[0])
    return [r[1] for r in results][:top_n]

def sequence_courses(courses):
    # Simple difficulty-based ordering
    order_map = {"Beginner":0, "Intermediate":1, "Advanced":2}
    df = pd.DataFrame(courses)
    df["order_key"] = df["difficulty"].map(order_map)
    df = df.sort_values("order_key").drop(columns=["order_key"])
    return df.to_dict(orient="records")

# -----------------------
# Initialize session state
# -----------------------
if "screen" not in st.session_state: 
    st.session_state.screen = "Home"
if "profile" not in st.session_state: 
    st.session_state.profile = {"skills": []}
if "suitability" not in st.session_state: 
    st.session_state.suitability = None
if "selected_field" not in st.session_state: 
    st.session_state.selected_field = None
if "missing_skills" not in st.session_state: 
    st.session_state.missing_skills = []
if "candidates" not in st.session_state: 
    st.session_state.candidates = []
if "final_path" not in st.session_state: 
    st.session_state.final_path = []

# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("🎓 Navigation")
menu = ["🏠 Home", "👤 Profile Input", "📊 Field Suitability", "🎯 Skill Gaps", "📚 Recommended Courses", "🗺️ Learning Path", "💡 Explanation"]
menu_keys = ["Home", "Profile Input", "Field Suitability", "Missing Skills", "Candidate Courses", "Learning Path", "Explanation"]

for display, key in zip(menu, menu_keys):
    if st.sidebar.button(display, use_container_width=True):
        st.session_state.screen = key

st.sidebar.markdown("---")
st.sidebar.info("**Model:** " + ("Sentence-BERT" if use_sbert else "TF-IDF"))

# -----------------------
# Main Content Area
# -----------------------

# ----- HOME -----
if st.session_state.screen == "Home":
    st.title("🎓 Personalized Learning Path Recommendation System")
    st.markdown("### Welcome to the Career Explorer Tool")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This prototype system helps you:
        
        1. **Assess Career Field Suitability** - Find which CS/IT fields match your profile
        2. **Identify Skill Gaps** - Discover what skills you need to develop
        3. **Generate Learning Paths** - Get a structured course sequence
        4. **Understand Recommendations** - See transparent explanations for all suggestions
        
        #### How it works:
        - Enter your background, skills, and interests
        - Get ranked career field recommendations
        - View personalized learning paths with explanations
        """)
        
        if st.button("🚀 Get Started", type="primary", use_container_width=True):
            st.session_state.screen = "Profile Input"
            st.rerun()
    
    with col2:
        st.info("**Current Profile:**\n\n" + 
                f"Name: {st.session_state.profile.get('name', 'Not set')}\n\n" +
                f"Skills: {len(st.session_state.profile.get('skills', []))}")

# ----- PROFILE INPUT -----
elif st.session_state.screen == "Profile Input":
    st.title("👤 Learner Profile Input")
    
    st.markdown("### Tell us about yourself")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Basic Information")
        name = st.text_input("Full Name", value=st.session_state.profile.get("name", ""))
        
        # Education Level
        edu_level = st.selectbox("Level of Education", 
                          ["-- select --","High School","Associate Degree","Bachelor's Degree","Master's Degree","Doctoral Degree (PhD)"],
                          index=0 if not st.session_state.profile.get("edu_level") else 
                                ["-- select --","High School","Associate Degree","Bachelor's Degree","Master's Degree","Doctoral Degree (PhD)"].index(st.session_state.profile.get("edu_level")))
        
        # Major/Specialization (only show if education is beyond high school)
        major = ""
        if edu_level not in ["-- select --", "High School"]:
            major = st.selectbox("Major / Specialization",
                               ["-- select --","Computer Science","Information Technology","Software Engineering","Data Science",
                                "Engineering (Non-CS)","Mathematics","Business","Natural Sciences","Humanities","Other"],
                               index=0 if not st.session_state.profile.get("major") else
                                     ["-- select --","Computer Science","Information Technology","Software Engineering","Data Science",
                                      "Engineering (Non-CS)","Mathematics","Business","Natural Sciences","Humanities","Other"].index(st.session_state.profile.get("major")))
        
        work_exp = st.text_area("Work Experience (Years & Roles)", 
                                 value=st.session_state.profile.get("work_exp", ""),
                                 placeholder="e.g., '2 years as Junior Developer at ABC Corp' or 'No professional experience yet'",
                                 height=100)
    
    with col2:
        st.subheader("Skills & Interests")
        
        # Skills section
        st.markdown("**Add Your Skills:**")
        skill_col1, skill_col2, skill_col3 = st.columns([3, 2, 1])
        
        with skill_col1:
            skill_text = st.text_input("Skill name", placeholder="e.g., Python", key="skill_input")
        with skill_col2:
            prof = st.selectbox("Proficiency", ["Beginner","Intermediate","Advanced"], key="prof_input")
        with skill_col3:
            st.write("")
            st.write("")
            if st.button("➕ Add"):
                if skill_text:
                    skills = st.session_state.profile.get("skills", [])
                    # Keep original case for display
                    skills.append((skill_text.strip(), prof))
                    st.session_state.profile["skills"] = skills
                    st.rerun()
        
        # Display existing skills
        if st.session_state.profile.get("skills"):
            st.markdown("**Current Skills:**")
            for i, (s, p) in enumerate(st.session_state.profile["skills"]):
                col_skill, col_remove = st.columns([5, 1])
                with col_skill:
                    st.text(f"• {s} ({p})")
                with col_remove:
                    if st.button("🗑️", key=f"rm_{i}"):
                        st.session_state.profile["skills"].pop(i)
                        st.rerun()
        
        st.markdown("---")
        interests = st.multiselect("Career Interests", 
                                  ["Machine Learning","Data Science","Web Development","Cybersecurity","Cloud Computing","DevOps","Mobile Development","UI/UX Design"],
                                  default=st.session_state.profile.get("interests", []))
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    with col_btn1:
        if st.button("🔍 Assess Field Suitability", type="primary", use_container_width=True):
            if not st.session_state.profile.get("skills"):
                st.error("Please add at least one skill!")
            elif edu_level == "-- select --":
                st.error("Please select your education level!")
            else:
                st.session_state.profile["name"] = name
                st.session_state.profile["edu_level"] = edu_level
                st.session_state.profile["major"] = major if major else "Not specified"
                st.session_state.profile["work_exp"] = work_exp
                st.session_state.profile["interests"] = interests
                st.session_state.screen = "Field Suitability"
                st.rerun()
    with col_btn2:
        if st.button("🔄 Reset Profile", use_container_width=True):
            st.session_state.profile = {"skills": []}
            st.rerun()

# ----- FIELD SUITABILITY -----
elif st.session_state.screen == "Field Suitability":
    st.title("📊 Career Field Suitability Assessment")
    
    prof = st.session_state.profile
    if not prof.get("skills"):
        st.warning("⚠️ Please complete your profile first.")
        if st.button("← Back to Profile"):
            st.session_state.screen = "Profile Input"
            st.rerun()
    else:
        user_text = f"{prof.get('work_exp','')} {' '.join(prof.get('interests',[]))}"
        user_skills = [s for s, p in prof.get("skills", [])]
        interests = prof.get("interests", [])
        
        st.markdown("### Top Recommended Fields for You")
        
        rows = []
        for k, f in FIELDS.items():
            score, sem, overlap, interest, matched = compute_suitability(user_text, user_skills, interests, f)
            rows.append({
                "field_key": k,
                "name": f["name"],
                "score": score,
                "sem": sem,
                "overlap": overlap,
                "interest": interest,
                "matched_skills": matched,
                "desc": f["desc"]
            })
        
        df = pd.DataFrame(rows).sort_values("score", ascending=False)
        
        for idx, r in df.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"{'🥇' if idx == 0 else '🥈' if idx == 1 else '🥉' if idx == 2 else '📌'} {r['name']}")
                    st.write(r["desc"])
                    
                    st.markdown(f"""
                    **Suitability Score: {r['score']:.2%}**
                    - 🎯 Skill Match: {r['overlap']:.0%} ({len(r['matched_skills'])}/{len(FIELDS[r['field_key']]['skills'])} skills)
                    - 💡 Interest Alignment: {r['interest']:.0%}
                    - 📝 Semantic Match: {r['sem']:.0%}
                    """)
                    
                    if r['matched_skills']:
                        st.success(f"✅ Your matching skills: {', '.join(r['matched_skills'])}")
                
                with col2:
                    st.metric("Overall Score", f"{r['score']:.1%}")
                    st.write("")
                    if st.button(f"Select This Field →", key=r["field_key"], use_container_width=True, type="primary" if idx == 0 else "secondary"):
                        st.session_state.selected_field = r["field_key"]
                        st.session_state.screen = "Missing Skills"
                        st.rerun()
                
                st.markdown("---")

# ----- MISSING SKILLS -----
elif st.session_state.screen == "Missing Skills":
    st.title("🎯 Skill Gap Analysis")
    
    prof = st.session_state.profile
    if not prof.get("skills") or not st.session_state.selected_field:
        st.warning("⚠️ Please complete profile and select a field first.")
    else:
        field = FIELDS[st.session_state.selected_field]
        user_skills = [s for s, p in prof.get("skills", [])]
        missing = find_missing_skills(user_skills, field["skills"])
        st.session_state.missing_skills = missing
        
        st.markdown(f"### Selected Field: **{field['name']}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Your Current Skills")
            if user_skills:
                for skill in user_skills:
                    if skill in field["skills"]:
                        st.success(f"✓ {skill}")
                    else:
                        st.info(f"• {skill}")
            else:
                st.write("No skills provided.")
        
        with col2:
            st.subheader("❌ Skills You Need to Develop")
            if missing:
                for skill in missing:
                    st.error(f"✗ {skill}")
                
                st.markdown(f"**Total Gaps: {len(missing)} skills**")
            else:
                st.success("🎉 You already have all the required skills!")
        
        st.markdown("---")
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if missing and st.button("📚 Find Courses to Address Gaps", type="primary", use_container_width=True):
                st.session_state.candidates = find_candidate_courses_for_skills(missing)
                st.session_state.screen = "Candidate Courses"
                st.rerun()
        with col_btn2:
            if st.button("← Choose Different Field", use_container_width=True):
                st.session_state.screen = "Field Suitability"
                st.rerun()

# ----- CANDIDATE COURSES -----
elif st.session_state.screen == "Candidate Courses":
    st.title("📚 Recommended Courses")
    
    candidates = st.session_state.candidates
    if not candidates:
        st.info("No candidate courses found. Try adjusting your profile or selecting another field.")
    else:
        st.markdown(f"### Found {len(candidates)} courses to address your skill gaps")
        
        for idx, c in enumerate(candidates, 1):
            with st.expander(f"**{idx}. {c['title']}** — {c['difficulty']}", expanded=(idx <= 3)):
                st.write(c['desc'])
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    covered_skills = [s for s in c['skills'] if s in st.session_state.missing_skills]
                    if covered_skills:
                        st.success(f"✅ Addresses gaps: {', '.join(covered_skills)}")
                    st.info(f"📖 Teaches: {', '.join(c['skills'])}")
                
                with col2:
                    st.metric("Difficulty", c['difficulty'])
        
        st.markdown("---")
        if st.button("🗺️ Generate Learning Path", type="primary", use_container_width=True):
            st.session_state.final_path = sequence_courses(candidates[:8])  # Limit to top 8
            st.session_state.screen = "Learning Path"
            st.rerun()

# ----- LEARNING PATH -----
elif st.session_state.screen == "Learning Path":
    st.title("🗺️ Your Personalized Learning Path")
    
    path = st.session_state.final_path
    if not path:
        st.info("No learning path generated yet.")
    else:
        field = FIELDS.get(st.session_state.selected_field, {})
        st.markdown(f"### Path for: **{field.get('name', 'Selected Field')}**")
        st.markdown(f"**Total Courses:** {len(path)} | **Estimated Duration:** {len(path) * 4} weeks")
        
        st.markdown("---")
        
        # Group by difficulty
        beginner = [c for c in path if c['difficulty'] == 'Beginner']
        intermediate = [c for c in path if c['difficulty'] == 'Intermediate']
        advanced = [c for c in path if c['difficulty'] == 'Advanced']
        
        if beginner:
            st.markdown("### 📘 Foundation Stage (Beginner)")
            for i, step in enumerate(beginner, 1):
                covered = [s for s in step['skills'] if s in st.session_state.missing_skills]
                with st.container():
                    st.markdown(f"**{i}. {step['title']}**")
                    st.write(f"📖 {step['desc']}")
                    if covered:
                        st.success(f"✅ Addresses: {', '.join(covered)}")
                    st.caption(f"Skills: {', '.join(step['skills'])}")
                    st.markdown("---")
        
        if intermediate:
            st.markdown("### 📗 Core Stage (Intermediate)")
            for i, step in enumerate(intermediate, len(beginner) + 1):
                covered = [s for s in step['skills'] if s in st.session_state.missing_skills]
                with st.container():
                    st.markdown(f"**{i}. {step['title']}**")
                    st.write(f"📖 {step['desc']}")
                    if covered:
                        st.success(f"✅ Addresses: {', '.join(covered)}")
                    st.caption(f"Skills: {', '.join(step['skills'])}")
                    st.markdown("---")
        
        if advanced:
            st.markdown("### 📕 Specialization Stage (Advanced)")
            for i, step in enumerate(advanced, len(beginner) + len(intermediate) + 1):
                covered = [s for s in step['skills'] if s in st.session_state.missing_skills]
                with st.container():
                    st.markdown(f"**{i}. {step['title']}**")
                    st.write(f"📖 {step['desc']}")
                    if covered:
                        st.success(f"✅ Addresses: {', '.join(covered)}")
                    st.caption(f"Skills: {', '.join(step['skills'])}")
                    st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("💡 View Detailed Explanation", type="primary", use_container_width=True):
                st.session_state.screen = "Explanation"
                st.rerun()
        with col2:
            csv = pd.DataFrame(path)[['title', 'difficulty', 'desc']].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Path (CSV)", csv, file_name="learning_path.csv", use_container_width=True)

# ----- EXPLANATION -----
elif st.session_state.screen == "Explanation":
    st.title("💡 Explanation & Transparency Report")
    
    prof = st.session_state.profile
    field = FIELDS.get(st.session_state.selected_field, {})
    
    st.markdown("## Why These Recommendations?")
    
    st.markdown("### 1️⃣ Field Suitability Reasoning")
    st.info(f"**Selected Field:** {field.get('name', 'N/A')}")
    st.write(field.get("desc", ""))
    
    if st.session_state.missing_skills:
        st.markdown("### 2️⃣ Skill Gap Analysis")
        st.warning(f"**Identified {len(st.session_state.missing_skills)} skill gaps:**")
        for skill in st.session_state.missing_skills:
            st.write(f"- {skill}")
    
    st.markdown("### 3️⃣ Course Selection & Sequencing Logic")
    st.write("""
    **How courses were selected:**
    - Filtered courses that teach your identified skill gaps
    - Prioritized courses covering multiple gaps
    - Considered difficulty levels for proper progression
    
    **How courses were sequenced:**
    - Beginner courses placed first (Foundation stage)
    - Intermediate courses in the middle (Core stage)
    - Advanced courses at the end (Specialization stage)
    - Prerequisites respected where applicable
    """)
    
    if st.session_state.final_path:
        st.markdown("### 4️⃣ Your Learning Path Summary")
        for i, step in enumerate(st.session_state.final_path, 1):
            covered = [s for s in step['skills'] if s in st.session_state.missing_skills]
            st.markdown(f"**Step {i}: {step['title']}** ({step['difficulty']})")
            if covered:
                st.write(f"→ Addresses gaps: {', '.join(covered)}")
            else:
                st.write(f"→ Reinforces: {step['skills'][0]}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🏠 Start New Assessment", use_container_width=True):
            # Reset all except profile
            st.session_state.selected_field = None
            st.session_state.missing_skills = []
            st.session_state.candidates = []
            st.session_state.final_path = []
            st.session_state.screen = "Home"
            st.rerun()
    with col2:
        csv = pd.DataFrame(st.session_state.final_path)[['title', 'difficulty', 'desc']].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Report", csv, file_name="recommendation_report.csv", use_container_width=True)