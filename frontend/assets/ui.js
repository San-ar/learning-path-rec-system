const SKILL_DATABASE = [
    { name: "Python", category: "Programming" },
    { name: "JavaScript", category: "Programming" },
    { name: "Java", category: "Programming" },
    { name: "React", category: "Web Development" },
    { name: "Node.js", category: "Web Development" },
    { name: "SQL", category: "Data" },
    { name: "Machine Learning", category: "Data Science" },
    { name: "Data Visualization", category: "Data Science" },
    { name: "Data Analytics", category: "Data Science" },
    { name: "HTML", category: "Web Design" },
    { name: "CSS", category: "Web Design" },
    { name: "Docker", category: "DevOps" },
    { name: "AWS", category: "Cloud Computing" },
    { name: "Git", category: "Tools" }
];

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

/**
 * Toggles dynamic fields based on selected Current Status
 */
function toggleStatusFields() {
    const status = document.getElementById('current_status').value;
    const container = document.getElementById('status-fields');
    container.innerHTML = '';
    container.classList.toggle('hidden', !status);

    if (status === 'student') {
        renderStudentFields(container);
    } else if (['employed_tech', 'freelance'].includes(status)) {
        renderProfessionalFields(container, true);
    } else if (['employed_other', 'career_change'].includes(status)) {
        renderProfessionalFields(container, false);
    }
}

// Update your saveStep1() to use these specific classes
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

function saveStep2() {
    const profile = getProfile();
    const skills = [];

    document.querySelectorAll("#skills-list > div").forEach(skillDiv => {
        const name = skillDiv.querySelector(".font-medium")?.innerText;
        const level = skillDiv.querySelector("select")?.value;

        if (name && level) {
            skills.push({ name, level: parseInt(level) });
        }
    });

    profile.skills = skills;
    saveProfile(profile);

    window.location.href = "profile-step3.html";
}

// Update your submit function to save this context
function submitProfile() {
    const profile = getProfile();
    
    // 1. Capture Technical Interests
    const selectedInterests = [];
    document.querySelectorAll('input[name="interests[]"]:checked').forEach(cb => {
        selectedInterests.push(cb.value);
    });
    profile.interests = selectedInterests;

    // 2. Capture Logistics
    profile.logistics = {
        hours_per_week: document.querySelector('input[name="hours_per_week"]').value,
        learning_style: document.querySelector('select[name="learning_style"]').value
    };

    // 3. Save and Move to results
    saveProfile(profile);
    window.location.href = "../results/field-results.html";
}

// Helper functions
function collectSkills() {
    return [
        { name: "Python", level: 2 },
        { name: "JavaScript", level: 1 }
    ];
}

function collectInterests() {
    return ["data_science", "artificial_intelligence"];
}


/**
 * Templates for dynamic sections
 */
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

/**
 * Logic for adding "Previous Roles" (Dynamic section)
 */
function addPreviousRole() {
    const list = document.getElementById('prev-roles-list');
    const newRole = document.createElement('div');
    newRole.className = 'grid md:grid-cols-3 gap-4 p-3 bg-gray-50 rounded-lg relative group';
    newRole.innerHTML = `
        <input type="text" placeholder="Role Title" class="px-3 py-2 border rounded-md text-sm">
        <input type="text" placeholder="Company" class="px-3 py-2 border rounded-md text-sm">
        <div class="flex gap-2">
            <input type="text" placeholder="Years" class="flex-1 px-3 py-2 border rounded-md text-sm">
            <button type="button" onclick="this.parentElement.parentElement.remove()" class="text-gray-400 hover:text-red-500">✕</button>
        </div>
    `;
    list.appendChild(newRole);
}

/**
 * Logic for adding "Previous Degrees" (Static section)
 */
function addDegree() {
    const list = document.getElementById('degrees-list');
    // We create a wrapper div for the new row
    const row = document.createElement('div');
    row.className = 'grid md:grid-cols-2 gap-4 mt-4';
    row.innerHTML = `
        <input type="text" placeholder="Degree" class="px-4 py-3 border rounded-lg">
        <div class="flex gap-2">
            <input type="text" placeholder="Institution & Year" class="flex-1 px-4 py-3 border rounded-lg">
            <button type="button" onclick="this.parentElement.parentElement.remove()" class="text-red-500 px-2">✕</button>
        </div>
    `;
    list.appendChild(row);
}

/**
 * Logic for adding "Other Work Experience"
 * Supports both fresh clicks and loading saved data
 */
function addExperience(savedData = null) {
    const list = document.getElementById('experience-list');
    if (!list) return;

    const row = document.createElement('div');
    // Ensure this class matches what saveStep1() looks for
    row.className = 'experience-row grid md:grid-cols-3 gap-4 p-3 border-b border-gray-200 relative animate-fade-in';
    
    row.innerHTML = `
        <input type="text" placeholder="Job Title" class="exp-role px-4 py-3 border rounded-lg text-sm" 
            value="${savedData?.role || ''}">
        <input type="text" placeholder="Company/Field" class="exp-company px-4 py-3 border rounded-lg text-sm" 
            value="${savedData?.company || ''}">
        <div class="flex gap-2">
            <input type="text" placeholder="Duration" class="exp-duration flex-1 px-4 py-3 border rounded-lg text-sm" 
                value="${savedData?.duration || ''}">
            <button type="button" onclick="this.parentElement.parentElement.remove()" class="text-red-500 px-2">✕</button>
        </div>
    `;
    list.appendChild(row);
}

/**
 * Injects the shared navbar and configures the back button and logo redirect.
 * @param {Object} backConfig - { url: string, label: string }
 */
function initializeNavbar(backConfig) {
    const isInsideFolder = window.location.pathname.includes('/profile/');
    const componentPath = isInsideFolder ? '../components/navbar.html' : './components/navbar.html';
    fetch(componentPath).then(r => r.text()).then(data => {
        document.getElementById('navbar-placeholder').innerHTML = data;
    });
}

// Profile State Management
function getProfile() {
    return JSON.parse(localStorage.getItem("learner_profile")) || { background: {}, skills: [], interests: [] };
}

function saveProfile(profile) {
    localStorage.setItem("learner_profile", JSON.stringify(profile));
}

/**
 * Adds a skill manually from the search input
 */
function addSkillManual() {
    const input = document.getElementById('skill-search');
    const skillName = input.value.trim();

    if (!skillName) return;

    // Prevent duplicates
    const existingSkills = Array.from(document.querySelectorAll('#skills-list .font-medium'))
        .map(el => el.innerText.toLowerCase());

    if (existingSkills.includes(skillName.toLowerCase())) {
        alert("Skill already added!");
        input.value = '';
        return;
    }

    renderSkillItem(skillName, "User Added");
    input.value = ''; // Clear input
}

/**
 * Creates the HTML for a skill row
 */
function renderSkillItem(name, category = "Custom Skill") {
    const list = document.getElementById('skills-list');
    
    // Duplicate check
    const existing = Array.from(list.querySelectorAll('.skill-name')).map(el => el.innerText.toLowerCase());
    if (existing.includes(name.toLowerCase())) return;

    const skillDiv = document.createElement('div');
    // Added 'skill-card' class here
    skillDiv.className = "skill-card flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl shadow-sm animate-fade-in mb-3";
    
    skillDiv.innerHTML = `
        <div class="flex-1">
            <div class="font-bold text-gray-900 skill-name">${name}</div>
            <div class="text-[10px] font-black uppercase tracking-widest text-indigo-500">${category}</div>
        </div>
        <div class="flex items-center gap-4">
            <select class="px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm font-bold text-gray-700">
                <option value="1">Beginner</option>
                <option value="2">Intermediate</option>
                <option value="3">Advanced</option>
                <option value="4" selected>Expert</option>
            </select>
            <button onclick="this.closest('.skill-card').remove()" class="text-gray-300 hover:text-red-500 transition p-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
                </svg>
            </button>
        </div>
    `;
    list.appendChild(skillDiv);
}

function saveStep2() {
    const profile = getProfile();
    const skills = [];

    // Select all skill cards
    document.querySelectorAll("#skills-list > div").forEach(skillDiv => {
        // Updated selector to match the actual HTML being rendered
        const name = skillDiv.querySelector(".font-bold")?.innerText; 
        const category = skillDiv.querySelector(".text-indigo-500")?.innerText;
        const level = skillDiv.querySelector("select")?.value;

        if (name && level) {
            skills.push({
                name: name,
                category: category || "Skill",
                level: parseInt(level)
            });
        }
    });

    profile.skills = skills;
    saveProfile(profile);

    window.location.href = "profile-step3.html";
}

/**
 * Filters the SKILL_DATABASE based on user input and shows a dropdown
 */
function handleSkillSearch(query) {
    const resultsDiv = document.getElementById('skill-results');
    const searchTerm = query.toLowerCase().trim();

    if (!searchTerm) {
        resultsDiv.classList.add('hidden');
        return;
    }

    const matches = SKILL_DATABASE.filter(skill =>
        skill.name.toLowerCase().includes(searchTerm)
    ).slice(0, 5); // Limit to top 5 matches

    if (matches.length > 0) {
        resultsDiv.innerHTML = matches.map(skill => `
            <div onclick="selectSkill('${skill.name}', '${skill.category}')" 
                 class="px-4 py-3 hover:bg-indigo-50 cursor-pointer border-b last:border-0 flex justify-between items-center">
                <span class="font-medium text-gray-800">${skill.name}</span>
                <span class="text-[10px] bg-gray-100 px-2 py-1 rounded text-gray-500 uppercase">${skill.category}</span>
            </div>
        `).join('');
        resultsDiv.classList.remove('hidden');
    } else {
        resultsDiv.innerHTML = `
            <div onclick="addSkillManual()" class="px-4 py-3 hover:bg-orange-50 cursor-pointer text-orange-600 text-sm font-medium">
                + Add "${query}" as a custom skill
            </div>`;
        resultsDiv.classList.remove('hidden');
    }
}

/**
 * Handles selecting a skill from the autocomplete dropdown
 */
function selectSkill(name, category) {
    renderSkillItem(name, category);
    document.getElementById('skill-search').value = '';
    document.getElementById('skill-results').classList.add('hidden');
}

function saveAndGoBack() {
    // 1. Run the same logic as saveStep2 to capture current list
    const profile = getProfile();
    const skills = [];

    document.querySelectorAll("#skills-list > div").forEach(skillDiv => {
        const name = skillDiv.querySelector(".skill-name")?.innerText;
        const level = skillDiv.querySelector("select")?.value;

        if (name && level) {
            skills.push({ name: name, level: parseInt(level) });
        }
    });

    profile.skills = skills;
    saveProfile(profile);

    // 2. Now navigate back
    window.location.href = "profile-step1.html";
}

const MAJOR_DATABASE = ["Computer Science", "Software Engineering", "Data Science", "Information Technology", "Business Administration", "Mechanical Engineering", "Psychology", "Graphic Design"];

// 1. Major Autocomplete
function handleMajorSearch(query) {
    const results = document.getElementById('major-results');
    const matches = MAJOR_DATABASE.filter(m => m.toLowerCase().includes(query.toLowerCase())).slice(0, 5);
    
    if (query && matches.length > 0) {
        results.innerHTML = matches.map(m => `<div onclick="selectMajor('${m}')" class="px-4 py-2 hover:bg-indigo-50 cursor-pointer">${m}</div>`).join('');
        results.classList.remove('hidden');
    } else {
        results.classList.add('hidden');
    }
}

function selectMajor(val) {
    document.getElementById('field_of_study').value = val;
    document.getElementById('major-results').classList.add('hidden');
}

// 2. Structured Degree Rows
function addDegreeRow(savedData = null) {
    const list = document.getElementById('degrees-list');
    const div = document.createElement('div');
    div.className = "degree-row grid md:grid-cols-3 gap-3 p-3 bg-white border rounded-lg relative mb-3";
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
            <button onclick="this.closest('.degree-row').remove()" class="text-red-500">✕</button>
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
    row.className = 'experience-row grid md:grid-cols-3 gap-4 p-3 border-b mb-2 relative';
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

function generateSemanticContext() {
    const profile = getProfile();
    const bg = profile.background;
    
    // Construct a rich string for SBERT to analyze
    let context = `A learner with a background in ${bg.field_of_study || 'general studies'}. `;
    context += `Currently a ${bg.current_status.replace('_', ' ')} at the ${bg.education_level} level. `;
    
    if (bg.career_goal) {
        context += `Their primary career objective is to ${bg.career_goal}. `;
    }

    if (profile.interests && profile.interests.length > 0) {
        context += `They are specifically interested in ${profile.interests.join(', ')}.`;
    }

    return context;
}