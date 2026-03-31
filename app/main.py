import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.routes import recommend, search, feedback
import joblib
from src.config import LINKS_PKL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models ONCE at startup — shared across ALL requests."""
    from app.dependencies import get_hybrid_model, get_links_df
    logger.info("🚀 Server starting — pre-loading all models into RAM...")
    get_hybrid_model()   # loads SVD + NCF + Popularity + Content into RAM once
    get_links_df()       # loads links dataframe into RAM once
    logger.info("✅ All models loaded. Server ready to serve requests.")
    yield
    logger.info("🛑 Server shutting down.")


app = FastAPI(
    title="Movie Recommendation System",
    description="Hybrid recommendation system using popularity, content-based and collaborative filtering",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

app.include_router(recommend.router)
app.include_router(search.router)
app.include_router(feedback.router)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    from app.config import SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "supabase_url": SUPABASE_PROJECT_URL,
            "supabase_key": SUPABASE_ANON_KEY
        }
    )

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    from app.config import SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "supabase_url": SUPABASE_PROJECT_URL,
            "supabase_key": SUPABASE_ANON_KEY
        }
    )

@app.get("/health")
def health_check():
    return {"status": "OK", "message": "Movie Recommender API is running"}
