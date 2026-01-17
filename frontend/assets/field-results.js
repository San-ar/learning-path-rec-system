initializeNavbar({ url: '../dashboard.html', label: 'Dashboard' });

document.addEventListener("DOMContentLoaded", loadFieldResults);

async function loadFieldResults() {
  const grid = document.getElementById("results-grid");
  const profile = JSON.parse(localStorage.getItem("learner_profile"));

  if (!profile) {
    window.location.href = "../profile/profile-step1.html";
    return;
  }

  try {
    // Show a loading state while waiting for SBERT
    grid.innerHTML = '<div class="col-span-full text-center py-10">Analyzing your profile...</div>';

    const response = await fetch("http://127.0.0.1:8000/api/field-suitability", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: profile })
    });

    const data = await response.json();

    if (data && data.results) {
      renderResults(data.results);
    }
  } catch (err) {
    grid.innerHTML = '<div class="col-span-full text-center text-red-500">Error connecting to server.</div>';
  }
}

function renderResults(fields) {
  const grid = document.getElementById("results-grid");
  grid.innerHTML = ""; // CLEAN THE GRID FIRST

  // Take only the first 3
  const topThree = fields.slice(0, 3);

  topThree.forEach((field, index) => {
    grid.innerHTML += createFieldCard(field, index);
  });
}

function createFieldCard(field, index) {
  const config = [
    { badge: "🏆 Best Match", color: "emerald" },
    { badge: "Strong Match", color: "blue" },
    { badge: "Steady Match", color: "purple" }
  ][index];

  const formattedName = field.field_name
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  return `
      <div class="relative group animate-fade-in">
        <div class="absolute -inset-0.5 bg-gradient-to-r from-${config.color}-500 to-${config.color}-600 rounded-2xl blur opacity-20"></div>
        <div class="relative bg-white rounded-2xl shadow-sm border p-6 flex flex-col h-full">
          <div class="flex justify-between items-start mb-4">
            <span class="bg-${config.color}-100 text-${config.color}-700 px-3 py-1 rounded-full text-xs font-bold uppercase">${config.badge}</span>
            <div class="text-right">
              <span class="block text-3xl font-black text-${config.color}-600">${Math.round(field.score)}%</span>
              <span class="text-[10px] text-gray-400 font-bold uppercase">Score</span>
            </div>
          </div>
          <h3 class="text-xl font-bold text-gray-800 mb-2">${formattedName}</h3>
          <div class="space-y-4 mt-auto">
            <div class="text-[10px] font-black text-gray-400 uppercase tracking-widest">Key Requirements</div>
            <div class="flex flex-wrap gap-2">
              ${field.top_skills.map(s => `<span class="bg-gray-100 text-gray-600 px-2.5 py-1 rounded-md text-xs font-medium">${s}</span>`).join("")}
            </div>
            <a href="skill-gaps.html?field=${field.id}" class="block w-full text-center bg-${config.color}-600 text-white py-2.5 rounded-xl font-bold 
            text-sm hover:bg-${config.color}-700 transition">Explore Path →</a>
          </div>
        </div>
      </div>`;
}