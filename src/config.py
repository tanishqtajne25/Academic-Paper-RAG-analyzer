import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 5
    
    # GROQ CONFIG
    GROQ_API_KEY: str
    MODEL_NAME: str = "llama-3.3-70b-versatile"    
    # LOCAL EMBEDDINGS (Keep this local!)
    EMBEDDING_MODEL_NAME: str = "llama3.1" 

    class Config:
        env_file = ".env"

settings = Settings()