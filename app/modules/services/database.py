# app/modules/video_processing/database.py

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from fastapi import Depends

DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


# --- Crear tablas automáticamente en el startup de FastAPI ---
def create_database():
    """
    Inicializa todas las tablas definidas en los modelos SQLAlchemy.
    LLamar esta función desde el evento startup de FastAPI.
    """
    Base.metadata.create_all(bind=engine)


# --- Dependency para obtener una sesión de DB ---
def get_db() -> Generator[Session]:
    """
    Función usada por FastAPI con Depends(),
    devuelve una sesión usable y la cierra automáticamente.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
