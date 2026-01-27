from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.pool import QueuePool
from fastapi import Depends

DATABASE_URL = "sqlite:///:memory:"

# Configuración mejorada para SQLite
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30  # Timeout de 30 segundos para operaciones
    },
    pool_pre_ping=True,
    # Usar QueuePool para mejor manejo de conexiones
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600  # Reciclar conexiones cada hora
)

# Eventos para mejorar el rendimiento de SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Configura pragmas de SQLite para mejor rendimiento y recuperación de bloqueos"""
    cursor = dbapi_conn.cursor()
    # WAL mode para mejor concurrencia
    cursor.execute("PRAGMA journal_mode=WAL")
    # Timeout más agresivo para evitar deadlocks
    cursor.execute("PRAGMA busy_timeout=30000")
    # Sincronización normal (no FULL para mejor velocidad)
    cursor.execute("PRAGMA synchronous=NORMAL")
    # Tamaño de caché más grande
    cursor.execute("PRAGMA cache_size=10000")
    cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Evitar recargas innecesarias
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
