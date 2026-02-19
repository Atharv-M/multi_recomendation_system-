import logging
from fastapi import FastAPI
from app.routes import recommend, search
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import Request


logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Movie Recommendation System",
    description="Hybrid recommendation system using popularity, content-based and collaborative filtering",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")


app.include_router(recommend.router)
app.include_router(search.router)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    from app.config import SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "supabase_url": SUPABASE_PROJECT_URL,
            "supabase_key": SUPABASE_ANON_KEY
        }
    )

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    from app.config import SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "supabase_url": SUPABASE_PROJECT_URL,
            "supabase_key": SUPABASE_ANON_KEY
        }
    )

@app.get("/health")
def health_check():
    return {"status": "OK", "message": "Movie Recommender API is running"}
