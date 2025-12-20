from fastapi import FastAPI, Request, Form, responses
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# This list holds your data in the computer's memory while the script runs
career_fields = [
    {"name": "Machine Learning Engineer", "category": "AI & Data Science"},
    {"name": "Cybersecurity Analyst", "category": "Security"}
]

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/admin/dashboard.html")
async def admin_dashboard(request: Request):
    # This sends the list to the HTML file
    return templates.TemplateResponse("admin.html", {
        "request": request, 
        "careers": career_fields,
        "total": len(career_fields)
    })

@app.post("/admin/inject-skill")
async def inject_skill(skill_name: str = Form(...), category: str = Form(...)):
    # Adds the new skill to our list
    career_fields.append({"name": skill_name, "category": category})
    # Redirects back to the dashboard to show the update
    return responses.RedirectResponse(url="/admin/dashboard.html", status_code=303)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)