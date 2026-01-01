initializeNavbar({ url: '../dashboard.html', label: 'Dashboard' });

document.addEventListener("DOMContentLoaded", loadFieldResults);

// In field-results.js
async function loadFieldResults() {
  const profile = JSON.parse(localStorage.getItem("learner_profile"));

  if (!profile) {
    alert("Profile not found. Please complete your profile first.");
    window.location.href = "../profile/profile-step1.html";
    return;
  }

  try {
    const response = await fetch("http://127.0.0.1:8000/api/field-suitability", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile)
    });

    const data = await response.json();

    // IMPORTANT: Access the 'results' key from your Python dictionary
    if (data && data.results) {
      renderResults(data.results);
    } else {
      console.error("Unexpected data format from server:", data);
    }

  } catch (err) {
    console.error("Field suitability error:", err);
  }
}

function renderResults(fields) {
  const grid = document.getElementById("results-grid");
  grid.innerHTML = "";

  fields.forEach((field, index) => {
    grid.innerHTML += createFieldCard(field, index);
  });
}

function createFieldCard(field, index) {
  const badge =
    index === 0 ? "🏆 Best Match" :
      index === 1 ? "Strong Match" :
        "Steady Match";

  const color =
    index === 0 ? "emerald" :
      index === 1 ? "blue" :
        "purple";

  return `
      <div class="relative group">
        <div class="absolute -inset-0.5 bg-gradient-to-r from-${color}-500 to-${color}-600 rounded-2xl blur opacity-20"></div>
        <div class="relative bg-white rounded-2xl shadow-sm border p-6 flex flex-col h-full">
          <div class="flex justify-between items-start mb-4">
            <span class="bg-${color}-100 text-${color}-700 px-3 py-1 rounded-full text-xs font-bold uppercase">
              ${badge}
            </span>
            <div class="text-right">
              <span class="block text-3xl font-black text-${color}-600">
                ${Math.round(field.score)}%
              </span>
              <span class="text-[10px] text-gray-400 font-bold uppercase">Score</span>
            </div>
          </div>

          <h3 class="text-xl font-bold text-gray-800 mb-2">${field.field_name}</h3>
          <p class="text-sm text-gray-600 mb-6 flex-grow leading-relaxed">
            ${field.description}
          </p>

          <div class="space-y-4">
            <div class="flex flex-wrap gap-2">
              ${field.top_skills.map(skill =>
    `<span class="bg-gray-100 text-gray-600 px-2.5 py-1 rounded-md text-xs font-medium">${skill}</span>`
  ).join("")}
            </div>

            <a href="skill-gaps.html?field=${encodeURIComponent(field.id)}"
               class="block w-full text-center bg-${color}-600 text-white py-2.5 rounded-xl font-bold text-sm hover:bg-${color}-700 transition">
              Explore Path →
            </a>
          </div>
        </div>
      </div>
    `;
}

async function calculateSemanticMatches() {
  // 1. Initialize the embedding pipeline
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  // 2. Prepare the Learner Context
  const profile = getProfile();
  const userText = profile.semantic_context;

  // 3. Define ESCO Fields (In a real project, this would be a larger JSON)
  const careerFields = [
    { id: 1, title: "Data Scientist", desc: "Analyzing complex data to find patterns and build predictive models." },
    { id: 2, title: "Web Developer", desc: "Building and maintaining websites using HTML, CSS, and JavaScript." },
    { id: 3, title: "Cybersecurity Analyst", desc: "Protecting networks and data from digital attacks and unauthorized access." }
  ];

  // 4. Generate User Embedding
  const userOutput = await extractor(userText, { pooling: 'mean', normalize: true });
  const userVector = userOutput.data;

  // 5. Calculate Similarities
  const results = await Promise.all(careerFields.map(async (field) => {
    const fieldOutput = await extractor(field.desc, { pooling: 'mean', normalize: true });
    const fieldVector = fieldOutput.data;

    const score = cosine_similarity(userVector, fieldVector);
    return {
      ...field,
      matchPercentage: Math.round(score * 100)
    };
  }));

  // 6. Sort and Render
  renderResults(results.sort((a, b) => b.matchPercentage - a.matchPercentage));
}

async function processResultsPage() {
  // 1. Run SBERT to find the best career field
  const topCareer = await calculateSemanticMatches();

  // 2. Identify missing skills
  const missingSkills = identifyGaps(topCareer);

  // 3. PLACE THE PRIORITIZATION BLOCK HERE
  // This organizes the gaps so the user isn't overwhelmed
  const prioritizedGaps = prioritizeSkillGaps(missingSkills, topCareer.relevanceWeight);

  // 4. Render the UI using the prioritized data
  renderSkillGapUI(prioritizedGaps);
  renderCourseRecommendations(prioritizedGaps);
}