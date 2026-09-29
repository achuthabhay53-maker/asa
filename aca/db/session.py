"""Engine + session factory."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aca.config import settings

engine = create_engine(
    settings().database_url,
    pool_pre_ping=True,
    pool_recycle=300,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
