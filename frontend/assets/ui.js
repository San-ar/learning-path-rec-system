/**
 * Toggles dynamic fields based on selected Current Status
 */
function toggleStatusFields() {
    const status = document.getElementById('current_status').value;
    const container = document.getElementById('status-fields');
    
    container.innerHTML = '';
    container.classList.add('hidden');

    if (!status) return;

    container.classList.remove('hidden');
    
    // Animation
    container.style.opacity = '0';
    setTimeout(() => {
        container.style.transition = 'opacity 0.3s ease';
        container.style.opacity = '1';
    }, 10);

    if (status === 'student') {
        renderStudentFields(container);
    } else if (['employed_tech', 'freelance'].includes(status)) {
        renderProfessionalFields(container, true);
    } else if (['employed_other', 'career_change'].includes(status)) {
        renderProfessionalFields(container, false);
    }
}

/**
 * Templates for dynamic sections
 */
function renderStudentFields(container) {
    const currentYear = new Date().getFullYear();
    let options = '';
    for (let i = 0; i < 10; i++) {
        options += `<option value="${currentYear + i}">${currentYear + i}</option>`;
    }

    container.innerHTML = `
        <div>
            <label class="block text-gray-700 font-semibold mb-2">Expected Graduation Year</label>
            <select class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500">
                <option value="">Select year</option>
                ${options}
            </select>
        </div>`;
}

function renderProfessionalFields(container, isTech) {
    container.innerHTML = `
        <div class="grid md:grid-cols-2 gap-6">
            <div>
                <label class="block text-gray-700 font-semibold mb-2">${isTech ? 'Current Role' : 'Current Industry / Role'}</label>
                <input type="text" placeholder="e.g., Senior Developer" class="w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-indigo-500">
            </div>
            <div>
                <label class="block text-gray-700 font-semibold mb-2">Years of Experience</label>
                <input type="number" min="0" placeholder="e.g., 2" class="w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-indigo-500">
            </div>
        </div>
        <div class="mt-6">
            <label class="block text-gray-700 font-semibold mb-2">Previous Roles <span class="text-gray-400 text-sm font-normal">(Optional)</span></label>
            <div id="prev-roles-list" class="space-y-3">
                </div>
            <button type="button" onclick="addPreviousRole()" class="mt-3 text-indigo-600 font-medium hover:underline text-sm">
                + Add a previous role
            </button>
        </div>
    `;
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
 */
function addExperience() {
    const list = document.getElementById('experience-list');
    const row = document.createElement('div');
    row.className = 'grid md:grid-cols-3 gap-4 p-3 border-b border-gray-200 relative animate-fade-in';
    row.innerHTML = `
        <input type="text" placeholder="Job Title" class="px-4 py-3 border rounded-lg text-sm">
        <input type="text" placeholder="Company/Field" class="px-4 py-3 border rounded-lg text-sm">
        <div class="flex gap-2">
            <input type="text" placeholder="Duration (e.g. 2 yrs)" class="flex-1 px-4 py-3 border rounded-lg text-sm">
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
    const isInsideFolder = window.location.pathname.includes('/profile/') || 
                           window.location.pathname.includes('/admin/') ||
                           window.location.pathname.includes('/results/');
    
    const componentPath = isInsideFolder ? '../components/navbar.html' : './components/navbar.html';
    const dashboardPath = isInsideFolder ? '../dashboard.html' : './dashboard.html';

    fetch(componentPath)
        .then(response => {
            if (!response.ok) throw new Error('Navbar component not found');
            return response.text();
        })
        .then(data => {
            document.getElementById('navbar-placeholder').innerHTML = data;

            // Setup Logo Redirect
            const logoLink = document.getElementById('nav-logo');
            if (logoLink) logoLink.href = dashboardPath;

            // Setup Back Button
            const backLink = document.getElementById('nav-back-link');
            if (backLink && backConfig && backConfig.url) {
                backLink.href = backConfig.url;
                backLink.classList.remove('hidden');
                backLink.title = `Back to ${backConfig.label}`;
            }
        })
        .catch(err => console.error('Navbar Error:', err));
}