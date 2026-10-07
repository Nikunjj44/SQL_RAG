"""
Two engines:
- admin (readonly=False): used ONLY by load_data.py
- readonly (default):     used by everything the agent touches



"""
from functools import lru_cache

from sqlalchemy import create_engine, text

from config import settings


## engine - manages a pool of database engines
## lru_cache -- ensures that the same engine is returned everytime

@lru_cache
def get_engine(readonly: bool = True):
    url = settings.READONLY_DATABASE_URL if readonly else settings.DATABASE_URL
    return create_engine(url, pool_pre_ping=True)


if __name__ == "__main__":
    with get_engine(readonly=False).connect() as conn:
        print(conn.execute(text("SELECT version()")).scalar())