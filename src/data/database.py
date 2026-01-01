"""
Database models and utilities for VantageFlow AI credit scoring system.

This module provides SQLAlchemy ORM models for managing borrower information,
transaction data, and credit labels in a SQLite database.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session

Base = declarative_base()


class Borrower(Base):
    """
    Borrower information table.

    Stores demographic and identifying information for each borrower.
    Demographic fields are included for fairness testing and bias detection only,
    ensuring the model does not discriminate based on protected characteristics.

    Attributes:
        borrower_id: Unique identifier (UUID) for the borrower
        created_at: Timestamp when the borrower record was created
        age: Borrower's age (for fairness analysis)
        income: Annual income
        employment_status: Current employment status
        education_level: Highest education level achieved
        gender: Gender (for fairness testing only)
        ethnicity: Ethnicity (for fairness testing only)
        zip_code: Residential zip code
    """
    __tablename__ = "borrowers"

    borrower_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Demographic fields (for fairness testing)
    age = Column(Integer, nullable=True)
    income = Column(Float, nullable=True)
    employment_status = Column(String(50), nullable=True)
    education_level = Column(String(50), nullable=True)
    gender = Column(String(20), nullable=True)
    ethnicity = Column(String(50), nullable=True)
    zip_code = Column(String(10), nullable=True)

    # Relationships
    transactions = relationship("Transaction", back_populates="borrower", cascade="all, delete-orphan")
    label = relationship("Label", back_populates="borrower", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Borrower(borrower_id={self.borrower_id}, created_at={self.created_at})>"


class Transaction(Base):
    """
    Transaction history table.

    Stores alternative credit data from various sources such as utility payments,
    rent payments, mobile phone bills, and other financial activities that
    traditional credit scores may not capture.

    Attributes:
        transaction_id: Unique identifier for the transaction
        borrower_id: Foreign key to the borrower
        transaction_date: Date when the transaction occurred
        amount: Transaction amount (positive for payments, negative for charges)
        category: Transaction category (e.g., 'rent', 'utilities', 'mobile')
        description: Additional details about the transaction
    """
    __tablename__ = "transactions"

    transaction_id = Column(Integer, primary_key=True, autoincrement=True)
    borrower_id = Column(String(36), ForeignKey("borrowers.borrower_id"), nullable=False, index=True)
    transaction_date = Column(DateTime, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    category = Column(String(50), nullable=False, index=True)
    description = Column(String(255), nullable=True)

    # Relationship
    borrower = relationship("Borrower", back_populates="transactions")

    def __repr__(self):
        return f"<Transaction(id={self.transaction_id}, borrower={self.borrower_id}, amount={self.amount})>"


# Create composite index for common queries
Index('idx_borrower_date', Transaction.borrower_id, Transaction.transaction_date)


class Label(Base):
    """
    Credit labels table.

    Stores the target variable for credit scoring - whether a borrower
    defaulted on a loan and the predicted probability of default.

    Attributes:
        borrower_id: Foreign key to the borrower (primary key)
        default_label: Boolean indicating if the borrower defaulted (True) or not (False)
        default_probability: Predicted probability of default (0.0 to 1.0)
        label_date: When the label was assigned
    """
    __tablename__ = "labels"

    borrower_id = Column(String(36), ForeignKey("borrowers.borrower_id"), primary_key=True)
    default_label = Column(Boolean, nullable=False)
    default_probability = Column(Float, nullable=True)
    label_date = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship
    borrower = relationship("Borrower", back_populates="label")

    def __repr__(self):
        return f"<Label(borrower={self.borrower_id}, default={self.default_label}, prob={self.default_probability})>"


class DatabaseManager:
    """
    Database connection and session management.

    Provides methods to initialize the database, create tables,
    and manage database connections.
    """

    def __init__(self, db_path: str = "data/credit_scoring.db"):
        """
        Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.engine = None
        self.SessionLocal = None

    def initialize(self, echo: bool = False) -> None:
        """
        Initialize the database engine and create all tables.

        Args:
            echo: If True, SQLAlchemy will log all SQL statements
        """
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=echo,
            connect_args={"check_same_thread": False}  # Needed for SQLite
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            SQLAlchemy Session object
        """
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.SessionLocal()

    def drop_all_tables(self) -> None:
        """
        Drop all tables in the database.

        Warning: This will delete all data!
        """
        if self.engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        Base.metadata.drop_all(self.engine)

    def recreate_tables(self) -> None:
        """
        Drop and recreate all tables.

        Warning: This will delete all data!
        """
        self.drop_all_tables()
        Base.metadata.create_all(self.engine)


def create_database(db_path: str = "data/credit_scoring.db", echo: bool = False) -> DatabaseManager:
    """
    Create and initialize a new database.

    Args:
        db_path: Path to the SQLite database file
        echo: If True, SQLAlchemy will log all SQL statements

    Returns:
        Initialized DatabaseManager instance
    """
    db_manager = DatabaseManager(db_path)
    db_manager.initialize(echo=echo)
    return db_manager


def get_session(db_path: str = "data/credit_scoring.db") -> Session:
    """
    Quick helper to get a database session.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        SQLAlchemy Session object
    """
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


if __name__ == "__main__":
    # Example usage
    db = create_database("data/credit_scoring.db", echo=True)
    print("Database created successfully!")

    # Create a sample borrower
    session = db.get_session()
    try:
        borrower = Borrower(
            age=35,
            income=50000.0,
            employment_status="Full-time",
            education_level="Bachelor's",
            gender="Other",
            ethnicity="Prefer not to say",
            zip_code="10001"
        )
        session.add(borrower)
        session.commit()
        print(f"Created borrower: {borrower}")
    finally:
        session.close()
