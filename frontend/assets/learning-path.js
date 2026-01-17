document.addEventListener("DOMContentLoaded", loadLearningPath);
async function loadLearningPath() {
    const profile = JSON.parse(localStorage.getItem("learner_profile"));
    const skillGaps = JSON.parse(localStorage.getItem("last_skill_gaps"));

    if (!profile || !skillGaps) {
        alert("No gap analysis found.");
        return;
    }
    try {
        const response = await fetch("http://127.0.0.1:8000/api/learning-path", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_id: 1, 
                gaps: skillGaps,
                profile: profile, // Pass the whole profile
                selected_role: profile.background.selected_role
            })
        });

        const data = await response.json();
        
        // Professional title returned by the API
        document.getElementById('role-title').innerText = `${toTitleCase(data.career_goal)} Path`;

        updateHeaderStats(data.learning_path);
        renderPath(data.learning_path);

    } catch (err) {
        console.error("Path error:", err);
    }
}

function updateHeaderStats(pathData) {
    const totalSkills = pathData.length;
    // Estimate 1.5 months per skill gap
    const estimatedMonths = Math.max(3, totalSkills * 1.5);

    document.getElementById("time-estimate").innerText = `~${estimatedMonths} Months Total`;
    document.getElementById("spec-count").innerText = `${totalSkills} Skill Specializations`;
}

function renderPath(pathData) {
    const container = document.getElementById("path-container");
    container.innerHTML = "";

    pathData.forEach((skillGroup, index) => {
        const phaseNumber = (index + 1).toString().padStart(2, '0');

        let skillHtml = `
            <div class="relative z-10 mb-16 animate-fade-in">
                <div class="flex items-center mb-8">
                    <div class="w-12 h-12 md:w-16 md:h-16 rounded-2xl bg-indigo-600 text-white flex items-center justify-center shadow-lg ring-4 ring-white">
                        <span class="text-xl font-black">${phaseNumber}</span>
                    </div>
                    <div class="ml-6">
                        <h3 class="text-xl font-bold text-gray-900">${skillGroup.skill} Mastery</h3>
                        <p class="text-sm text-gray-500 font-medium">Target Level: ${skillGroup.target_level}</p>
                    </div>
                </div>
                <div class="ml-6 md:ml-16 space-y-6">
                    ${skillGroup.steps.map(step => `
                        <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 hover:shadow-md transition-all border-l-4 border-l-indigo-500">
                            <div class="flex flex-col md:flex-row justify-between gap-6">
                                <div class="flex-1">
                                    <h4 class="text-lg font-bold text-gray-800">${step.course_title}</h4>
                                    <p class="text-xs text-gray-500 mb-3 uppercase font-bold tracking-tight">${step.provider} • ${step.level}</p>
                                    <p class="text-sm text-gray-600 mb-4">${step.brief}</p>
                                    <div class="flex items-center gap-4 text-xs text-gray-400 font-bold">
                                        <span class="text-indigo-600">${step.reason}</span>
                                        <span class="bg-green-50 text-green-700 px-2 py-0.5 rounded">Match: ${Math.round(step.match_score * 100)}%</span>
                                    </div>
                                </div>
                                <div class="flex items-center">
                                    <a href="${step.url}" target="_blank" class="w-full md:w-auto text-center bg-indigo-600 text-white px-6 py-2.5 rounded-xl 
                                    text-sm font-bold hover:bg-indigo-700 transition shadow-sm">Explore Course</a>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', skillHtml);
    });
}

function toTitleCase(str) {
  return str
    .toLowerCase()
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

