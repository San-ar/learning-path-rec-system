// function goToStep(stepNumber) {
//     for (let i = 1; i <= 3; i++) {
//         const step = document.getElementById(`wizard-step-${i}`);
//         const circle = document.getElementById(`step${i}-circle`);

//         if (!step || !circle) continue;

//         step.style.display = i === stepNumber ? "block" : "none";

//         if (i <= stepNumber) {
//             circle.className =
//                 "progress-step w-10 h-10 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold";
//         } else {
//             circle.className =
//                 "progress-step w-10 h-10 rounded-full bg-gray-300 text-gray-600 flex items-center justify-center font-bold";
//         }
//     }
// }