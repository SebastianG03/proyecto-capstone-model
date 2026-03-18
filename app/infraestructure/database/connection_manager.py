from sqlalchemy.orm import Session, declarative_base, scoped_session, sessionmaker
from sqlalchemy import QueuePool, create_engine

from app.utils.routes import DATABASE_DIR

Base = declarative_base()

class ConnectionManager():
    DATABASE_URL = "sqlite:///"

    def __init__(self, match_id: int):
        """
        Constructor de la clase ConnectionManager.

        Args:
            match_id (int): ID del partido para el que se crea la base de datos.

        """
        self.create_database(match_id)

    def create_session(self):
        """
        Crea una sesión de base de datos que se puede utilizar para interactuar
        con la base de datos. La sesión se cierra automáticamente cuando se
        sale del ámbito de la sesión.

        Returns:
            Session: Sesión de base de datos.
        """
        return Session(self.engine, expire_on_commit=False, autoflush=False)
        
    def close_session(self):
        """
        Cierra la sesión actual de la base de datos.
        Se llama automáticamente cuando se sale del ámbito de la sesión.
        """
        self.session.remove()
    
    def create_database(self, match_id: int):
        """
        Crea una base de datos temporal para el partido especificado por match_id.
        
        Se crea un motor de base de datos con una sesión de base de datos
        que se puede utilizar para interactuar con la base de datos. La sesión
        se cierra automáticamente cuando se sale del ámbito de la sesión.
        
        La base de datos se crea en la carpeta especificada por DATABASE_DIR
        y se llama "temp_db_<match_id>.sqlite".
        
        Args:
            match_id (int): ID del partido para el que se crea la base de datos.
        """
        from app.entities.models import Player, PlayerState, BallEventModel, HeatmapPointModel
        db_path = DATABASE_DIR / f"temp_db_{match_id}.sqlite"
        self.engine = create_engine(
            f"{self.DATABASE_URL}{db_path.as_posix()}",
            connect_args={
                "check_same_thread": False,
                "timeout": 30
            },
            pool_pre_ping=True,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=15,
            pool_recycle=120000,
            echo=False
        )
        
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False
            )
        self.session = scoped_session(self.session_factory)
        Base.metadata.create_all(bind=self.engine)
