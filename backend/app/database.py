from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_database_url


class Base(DeclarativeBase):
    pass


def create_database_engine(database_url: str | None = None) -> Engine:
    return create_engine(database_url or get_database_url(), pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
