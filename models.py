from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Replace with your database URL
DATABASE_URL = "sqlite:///./test.db"  # Example for SQLite

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class ExampleModel(Base):  # Replace ExampleModel with a meaningful name
    __tablename__ = "example_models"  # Replace example_models with a meaningful name

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)
