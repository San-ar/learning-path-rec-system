document.addEventListener("DOMContentLoaded", loadSkillGaps);

async function loadSkillGaps() {
    const params = new URLSearchParams(window.location.search);
    const fieldId = params.get("field");
    const profile = JSON.parse(localStorage.getItem("learner_profile"));

    if (!profile || !fieldId) {
        alert("Missing profile or field selection.");
        return;
    }

    try {
        const response = await fetch("http://127.0.0.1:8000/api/skill-gaps", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                profile: profile,
                field_id: fieldId
            })
        });

        const data = await response.json();
        renderSkillGaps(data);
    } catch (err) {
        console.error("Skill gap error:", err);
    }
}

function renderSkillGaps(data) {
    document.getElementById("field-name").innerText = data.field_name;

    // --- 1. HANDLE STRENGTHS (EXPERT/MET SKILLS) ---
    const strengthList = document.getElementById("strength-list");
    strengthList.innerHTML = "";
    
    if (!data.strengths || data.strengths.length === 0) {
        strengthList.innerHTML = `<p class="p-6 text-gray-400 italic">No skills currently match the target requirements.</p>`;
    } else {
        data.strengths.forEach(s => {
            // LEVEL COLOR LOGIC: Indigo for Expert, Green for others
            const levelColorClass = (s.current_level === "Expert") ? "text-indigo-600" : "text-green-600";
            
            strengthList.innerHTML += `
                <div class="p-4 hover:bg-gray-50 transition-colors flex justify-between items-center">
                    <div>
                        <h4 class="font-bold text-gray-800">${s.skill}</h4>
                        <p class="text-xs ${levelColorClass} font-semibold uppercase tracking-wider">${s.current_level}</p>
                    </div>
                    <span class="text-green-500 bg-green-50 px-3 py-1 rounded-full text-xs font-bold">Requirement Met</span>
                </div>
            `;
        });
    }

    // --- 2. HANDLE GAPS (AREAS TO DEVELOP) ---
    const gapList = document.getElementById("gap-list");
    gapList.innerHTML = "";
    
    if (!data.skill_gaps || data.skill_gaps.length === 0) {
        gapList.innerHTML = `<p class="p-6 text-green-600 font-bold italic text-center">You have no major gaps! Ready for transition. 🎉</p>`;
    } else {
        data.skill_gaps.forEach(gap => {
            gapList.innerHTML += `
                <div class="p-4 hover:bg-gray-50 transition-colors flex justify-between items-center">
                    <div>
                        <h4 class="font-bold text-gray-800">${gap.skill}</h4>
                        <p class="text-xs text-gray-500 italic">Needs: ${gap.required_level}</p>
                    </div>
                    <div class="text-right">
                        <span class="block text-sm font-bold text-orange-600">${gap.current_level}</span>
                        <span class="text-[10px] text-gray-400 uppercase tracking-tighter">Current Level</span>
                    </div>
                </div>
            `;
        });
    }

    // --- 3. DYNAMIC PERCENTAGE & SVG ANIMATION ---
    const percentage = data.match_percentage !== undefined ? data.match_percentage : 0;

    // Update the text in the center of the circle
    const percentageText = document.querySelector('.text-2xl.font-black');
    if (percentageText) {
        percentageText.innerText = `${percentage}%`;
    }

    // NEW: Update the Readiness Description Text
    const readinessText = document.getElementById('readiness-text');
    if (readinessText) {
        readinessText.innerText = `${percentage}%`;
        
        // Optional: Change the description based on the score
        const descriptionPara = readinessText.parentElement;
        if (percentage >= 80) {
            readinessText.className = "font-bold text-green-600";
            descriptionPara.innerHTML = `Based on your profile, you are <span id="readiness-text" class="font-bold text-green-600">${percentage}%</span> ready for this role. You're a strong candidate!`;
        } else if (percentage >= 50) {
            readinessText.className = "font-bold text-indigo-600";
            descriptionPara.innerHTML = `Based on your profile, you are <span id="readiness-text" class="font-bold text-indigo-600">${percentage}%</span> ready. You're on the right track!`;
        } else {
            readinessText.className = "font-bold text-orange-600";
            descriptionPara.innerHTML = `Based on your profile, you are <span id="readiness-text" class="font-bold text-orange-600">${percentage}%</span> ready. Use the learning path below to bridge the gaps.`;
        }
    }

    // Update the SVG Circle Dash-Offset
    const circle = document.querySelector('circle.text-indigo-600');
    if (circle) {
        const circumference = 364.4; 
        const offset = circumference - (percentage / 100) * circumference;
        
        circle.style.transition = "stroke-dashoffset 1s ease-out";
        circle.style.strokeDashoffset = offset;
    }
}