from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    create_engine,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./rag_visualizer.db"  # Use SQLite as fallback for local development
)


# Use SQLite for local development if no DATABASE_URL is set
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class IDLink(Base):
    __tablename__ = "id_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    a_type = Column(String(64), nullable=False)
    a_id = Column(String(255), nullable=False)
    b_type = Column(String(64), nullable=False)
    b_id = Column(String(255), nullable=False)
    relation = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_id_links_a", "a_type", "a_id"),
        Index("idx_id_links_b", "b_type", "b_id"),
        Index("idx_id_links_relation", "relation"),
        UniqueConstraint(
            "a_type", "a_id", "b_type", "b_id", "relation", name="uq_id_links_pair"
        ),
    )


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()