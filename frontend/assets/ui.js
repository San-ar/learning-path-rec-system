// Global State
let OFFICIAL_SKILL_DATABASE = [];

const MAJOR_OPTIONS = `
    <option value="">Select major</option>
    <option value="Computer Science">Computer Science</option>
    <option value="Software Engineering">Software Engineering</option>
    <option value="Data Science">Data Science</option>
    <option value="Information Technology">Information Technology</option>
    <option value="Cybersecurity">Cybersecurity</option>
    <option value="Business Administration">Business Administration</option>
    <option value="Mechanical Engineering">Mechanical Engineering</option>
    <option value="Electrical Engineering">Electrical Engineering</option>
    <option value="Mathematics">Mathematics</option>
    <option value="Graphic Design">Graphic Design</option>
    <option value="Other (CS Related)">Other (CS Related)</option>
    <option value="Other (Non-CS)">Other (Non-CS)</option>
`;

// Fetch skills from MySQL via FastAPI
async function loadSkillsFromDB() {
    try {
        const response = await fetch('http://127.0.0.1:8000/api/skills');
        const data = await response.json();
        OFFICIAL_SKILL_DATABASE = data;
        return data;
    } catch (err) {
        console.error("Critical Error: Could not connect to backend skills API.", err);
        return [];
    }
}

function getProfile() {
    const saved = localStorage.getItem("learner_profile");
    return saved ? JSON.parse(saved) : { 
        background: {}, 
        skills: [], 
        interests: [] 
    };
}

// SAVE by merging with what is already there
function saveProfile(updatedSectionOrFullProfile, data = null) {
    const currentProfile = getProfile();
    
    if (typeof updatedSectionOrFullProfile === 'string') {
        currentProfile[updatedSectionOrFullProfile] = data;
    } else {
        Object.assign(currentProfile, updatedSectionOrFullProfile);
    }
    
    localStorage.setItem("learner_profile", JSON.stringify(currentProfile));
}

// Toggles dynamic fields based on selected Current Status
function toggleStatusFields() {
    const status = document.getElementById('current_status').value;
    const container = document.getElementById('status-fields');
    
    container.innerHTML = '';
    
    if (!status) {
        container.classList.add('hidden');
        return;
    }

    // Ensure hidden class is removed when a status is selected
    container.classList.remove('hidden');

    if (status === 'student') {
        renderStudentFields(container);
    } else if (['employed_tech', 'freelance', 'entrepreneur'].includes(status)) {
        renderProfessionalFields(container, true);
    } else if (['employed_other', 'career_change', 'job_seeking'].includes(status)) {
        renderProfessionalFields(container, false);
    }
}

function renderStudentFields(container) {
    const currentYear = new Date().getFullYear();
    let yearOptions = '';
    for (let i = 0; i < 10; i++) {
        yearOptions += `<option value="${currentYear + i}">${currentYear + i}</option>`;
    }

    container.innerHTML = `
        <div class="grid md:grid-cols-2 gap-6 p-6 bg-indigo-50 rounded-xl border border-indigo-100 animate-fade-in">
            <div class="md:col-span-2">
                <h4 class="text-indigo-900 font-bold text-sm uppercase tracking-wider mb-2">Current Academic Details</h4>
            </div>
            <div>
                <label class="block text-gray-700 font-semibold mb-2">Degree Type</label>
                <select class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500">
                    <option value="bachelors">Bachelor's Degree</option>
                    <option value="masters">Master's Degree</option>
                    <option value="diploma">Diploma / Associate</option>
                    <option value="phd">PhD / Doctorate</option>
                </select>
            </div>
            <div>
                <label class="block text-gray-700 font-semibold mb-2">Expected Graduation</label>
                <select class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500">
                    ${yearOptions}
                </select>
            </div>
            <div class="md:col-span-2">
                <label class="block text-gray-700 font-semibold mb-2">Major / Subject of Study</label>
                <select class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500">
                    ${MAJOR_OPTIONS}
                </select>
            </div>
        </div>`;
}

function renderProfessionalFields(container, isTech) {
    container.innerHTML = `
        <div class="grid md:grid-cols-2 gap-6 p-6 bg-gray-50 rounded-xl border border-gray-200">
            <div>
                <label class="block text-gray-700 font-semibold mb-2">${isTech ? 'Current Role' : 'Current Industry'}</label>
                <input type="text" placeholder="e.g., Senior Developer" class="w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-indigo-500">
            </div>
            <div>
                <label class="block text-gray-700 font-semibold mb-2">Years of Experience</label>
                <input type="number" min="0" placeholder="e.g., 2" class="w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-indigo-500">
            </div>
        </div>`;
}

// BACKGROUND LOGIC
function saveStep1() {
    const profile = getProfile();
    profile.background = {
        education_level: document.getElementById('education_level').value,
        completion_year: document.getElementById('completion_year').value,
        field_of_study: document.getElementById('field_of_study').value,
        current_status: document.getElementById('current_status').value,
        career_goal: document.getElementById('career_goal').value,
        status_details: Array.from(document.querySelectorAll('#status-fields input, #status-fields select')).map(i => i.value),
        degrees: Array.from(document.querySelectorAll('.degree-row')).map(row => ({
            type: row.querySelector('.degree-type').value,
            major: row.querySelector('.degree-major').value,
            year: row.querySelector('.degree-year').value
        })),
        experience: Array.from(document.querySelectorAll('.experience-row')).map(row => ({
            role: row.querySelector('.exp-role').value,
            company: row.querySelector('.exp-company').value,
            duration: row.querySelector('.exp-duration').value
        }))
    };
    saveProfile(profile);
    window.location.href = "profile-step2.html";
}

//SKILLS LOGIC (DB CONNECTED)
async function handleSkillSearch(query) {
    const resultsDiv = document.getElementById('skill-results');
    const searchTerm = query.toLowerCase().trim();

    if (!searchTerm) {
        resultsDiv.classList.add('hidden');
        return;
    }

    if (OFFICIAL_SKILL_DATABASE.length === 0) {
        await loadSkillsFromDB();
    }

    const matches = OFFICIAL_SKILL_DATABASE.filter(skill =>
        skill.skill_name.toLowerCase().includes(searchTerm)
    );

    const uniqueNames = [...new Set(matches.map(s => s.skill_name))].slice(0, 5);

    if (uniqueNames.length > 0) {
        resultsDiv.innerHTML = uniqueNames.map(skillName => `
            <div onclick="selectSkill('${skillName.replace(/'/g, "\\'")}')" 
                 class="px-4 py-3 hover:bg-indigo-50 cursor-pointer border-b last:border-0 flex justify-between items-center">
                <span class="font-medium text-gray-800">${skillName}</span>
                <span class="text-[10px] bg-green-100 text-green-700 px-2 py-1 rounded font-bold uppercase">Database Match</span>
            </div>
        `).join('');
        resultsDiv.classList.remove('hidden');
    } else {
        resultsDiv.innerHTML = `
            <div onclick="addSkillManual()" class="px-4 py-3 hover:bg-orange-50 cursor-pointer text-orange-600 text-sm font-medium">
                + Add "${query}" as custom skill
            </div>`;
        resultsDiv.classList.remove('hidden');
    }
}

// UPDATE SELECTSKILL TO RENDER AND SAVE
function selectSkill(name) {
    renderSkillItem(name, "Verified Skill");
    document.getElementById('skill-search').value = '';
    document.getElementById('skill-results').classList.add('hidden');
    saveStep2DataOnly(); 
}

// TO RENDER EXISTING DATA ON LOAD
function renderSkillsList() {
    const profile = getProfile();
    const list = document.getElementById('skills-list');
    if (!list) return;
    
    list.innerHTML = ''; // Clear "No skills added yet" message
    if (profile.skills && profile.skills.length > 0) {
        profile.skills.forEach(skill => {
            renderSkillItem(skill.name, "Saved Skill", skill.level);
        });
    }
}

// UPDATE RENDERSKILLITEM TO HANDLE LEVELS
function renderSkillItem(name, label, savedLevel = 1) {
    const list = document.getElementById('skills-list');
    
    // Unique check
    const existing = Array.from(list.querySelectorAll('.skill-name-text')).map(el => el.innerText.toLowerCase());
    if (existing.includes(name.toLowerCase())) return;

    const skillDiv = document.createElement('div');
    skillDiv.className = "skill-card flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl shadow-sm mb-3 animate-fade-in";
    
    skillDiv.innerHTML = `
        <div class="flex-1">
            <div class="font-bold text-gray-900 skill-name-text">${name}</div>
            <div class="text-[10px] font-black uppercase tracking-widest text-indigo-500">${label}</div>
        </div>
        <div class="flex items-center gap-4">
            <select onchange="saveStep2DataOnly()" class="skill-level-select px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm font-bold text-gray-700">
                <option value="1" ${savedLevel == 1 ? 'selected' : ''}>Beginner</option>
                <option value="2" ${savedLevel == 2 ? 'selected' : ''}>Intermediate</option>
                <option value="3" ${savedLevel == 3 ? 'selected' : ''}>Advanced</option>
                <option value="4" ${savedLevel == 4 ? 'selected' : ''}>Expert</option>
            </select>
            <button onclick="this.closest('.skill-card').remove(); saveStep2DataOnly();" class="text-gray-300 hover:text-red-500 p-2">
                ✕
            </button>
        </div>
    `;
    list.appendChild(skillDiv);
}

// Ensure "Back" button also saves current state
function saveAndGoBack() {
    saveStep2DataOnly();
    window.location.href = "profile-step1.html";
}

// Final Next button
function saveStep2() {
    saveStep2DataOnly();
    window.location.href = "profile-step3.html";
}

// HELPER TO CAPTURE CURRENT UI AND PUT IN LOCALSTORAGE
function saveStep2DataOnly() {
    const skills = [];
    document.querySelectorAll(".skill-card").forEach(card => {
        const name = card.querySelector(".skill-name-text")?.innerText;
        const level = card.querySelector(".skill-level-select")?.value;
        if (name && level) {
            skills.push({ name: name, level: parseInt(level) });
        }
    });
    saveProfile('skills', skills);
}
// SUBMISSION & API CONNECT
function generateSemanticContext() {
    const profile = getProfile();
    const bg = profile.background;
    let context = `Learner background: ${bg.field_of_study}. Goal: ${bg.career_goal}. `;
    if (profile.interests) context += `Interests: ${profile.interests.join(', ')}.`;
    return context;
}

async function submitProfile() {
    const profile = getProfile();
    
    // Capture Interests
    profile.interests = Array.from(document.querySelectorAll('input[name="interests[]"]:checked')).map(cb => cb.value);

    // Capture Logistics
    profile.logistics = {
        hours_per_week: document.querySelector('input[name="hours_per_week"]').value,
        learning_style: document.querySelector('select[name="learning_style"]').value
    };

    saveProfile(profile);

    // Prepare API Payload
    const payload = {
        profile: {
            skills: profile.skills,
            background: profile.background,
            interests: profile.interests,
            semantic_context: generateSemanticContext()
        }
    };

    try {
        const response = await fetch('http://127.0.0.1:8000/api/field-suitability', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const results = await response.json();
        localStorage.setItem("last_results", JSON.stringify(results));
        window.location.href = "../results/field-results.html";
    } catch (err) {
        alert("API Error: Make sure FastAPI server is running!");
    }
}

// SHARED UI HELPERS
function initializeNavbar(config) {
    const placeholder = document.getElementById('navbar-placeholder');
    if (!placeholder) return;
    
    // Detects if in a subfolder like /results/ or /profile/
    const depth = window.location.pathname.split('/').filter(p => p).length;
    const prefix = depth > 1 ? '../' : './';
    
    fetch(`${prefix}components/navbar.html`)
        .then(r => r.text())
        .then(html => { placeholder.innerHTML = html; })
        .catch(e => console.error("Navbar failed to load", e));
}

// Add Dynamic Rows (Degrees & Experience)
function addDegreeRow(savedData = null) {
    const list = document.getElementById('degrees-list');
    const div = document.createElement('div');
    div.className = "degree-row grid md:grid-cols-3 gap-3 p-3 bg-white border rounded-lg relative mb-3 animate-fade-in";
    
    const yearOptions = Array.from({length: 30}, (_, i) => 2026 - i).map(y => `<option value="${y}">${y}</option>`).join('');
    
    div.innerHTML = `
        <select class="degree-type px-3 py-2 border rounded-md">
            <option value="bachelors">Bachelor's</option>
            <option value="masters">Master's</option>
            <option value="diploma">Diploma</option>
            <option value="phd">PhD</option>
        </select>
        <select class="degree-major px-3 py-2 border rounded-md">
            ${MAJOR_OPTIONS}
        </select>
        <div class="flex gap-2">
            <select class="degree-year px-3 py-2 border rounded-md flex-1">${yearOptions}</select>
            <button type="button" onclick="this.closest('.degree-row').remove()" class="text-red-500">✕</button>
        </div>
    `;
    list.appendChild(div);

    if (savedData) {
        div.querySelector('.degree-type').value = savedData.type;
        div.querySelector('.degree-major').value = savedData.major;
        div.querySelector('.degree-year').value = savedData.year;
    }
}

function addExperience(savedData = null) {
    const list = document.getElementById('experience-list');
    const row = document.createElement('div');
    row.className = 'experience-row grid md:grid-cols-3 gap-4 p-3 border rounded-lg mb-2 relative animate-fade-in';
    row.innerHTML = `
        <input type="text" placeholder="Job Title" class="exp-role px-4 py-3 border rounded-lg text-sm" value="${savedData?.role || ''}">
        <input type="text" placeholder="Company" class="exp-company px-4 py-3 border rounded-lg text-sm" value="${savedData?.company || ''}">
        <div class="flex gap-2">
            <input type="text" placeholder="Duration" class="exp-duration flex-1 px-4 py-3 border rounded-lg text-sm" value="${savedData?.duration || ''}">
            <button type="button" onclick="this.parentElement.parentElement.remove()" class="text-red-500">✕</button>
        </div>
    `;
    list.appendChild(row);
}