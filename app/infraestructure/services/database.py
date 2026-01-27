from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from contextlib import contextmanager

from app.utils.routes import DATABASE_DIR

DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}, pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# --- Crear tablas automáticamente en el startup de FastAPI ---
def create_database():
    """
    Inicializa todas las tablas definidas en los modelos SQLAlchemy.
    LLamar esta función desde el evento startup de FastAPI.
    """
    Base.metadata.create_all(bind=engine)


def create_temporary_database(match_id: int):
    db_path = DATABASE_DIR / f"temp_db_{match_id}.sqlite"
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    return db, engine, db_path




@contextmanager
def get_db_session(match_id: int):
    """
    Context manager para manejar sesiones de base de datos de forma segura.
    Crea una base de datos temporal por cada frame.
    """
    db = None
    try:
        # Crear base de datos temporal para este match
        db, _, _ = create_temporary_database(match_id)
        yield db
        db.commit()
    except Exception:
        if db:
            db.rollback()
        raise
    finally:
        if db:
            db.close()


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