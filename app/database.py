from supabase import create_client, Client
from app.config import SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY

# Initialize the Supabase client
try:
    if not SUPABASE_PROJECT_URL or not SUPABASE_ANON_KEY:
        raise ValueError("Missing Supabase credentials in environment config.")
        
    supabase: Client = create_client(SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY)
    print("Supabase client initialized successfully.")
    
except Exception as e:
    print(f"Failed to initialize Supabase client: {e}")
    supabase = None
