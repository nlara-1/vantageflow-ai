-- VantageFlow AI Credit Scoring Database Schema
-- SQLite database schema for alternative credit scoring system
--
-- This schema includes three main tables:
-- 1. borrowers: Demographic and identifying information
-- 2. transactions: Alternative credit transaction history
-- 3. labels: Credit default labels and predictions

-- ============================================================================
-- BORROWERS TABLE
-- ============================================================================
-- Stores borrower demographic information for fairness testing and analysis.
-- Demographic fields are used only for bias detection and should NOT be
-- used as features in the credit scoring model.

CREATE TABLE IF NOT EXISTS borrowers (
    borrower_id VARCHAR(36) PRIMARY KEY,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Demographic fields (for fairness testing only)
    age INTEGER,
    income REAL,
    employment_status VARCHAR(50),
    education_level VARCHAR(50),
    gender VARCHAR(20),
    ethnicity VARCHAR(50),
    zip_code VARCHAR(10)
);

-- Index on created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_borrowers_created_at ON borrowers(created_at);

-- ============================================================================
-- TRANSACTIONS TABLE
-- ============================================================================
-- Stores alternative credit data including utility payments, rent, mobile bills,
-- and other financial activities that traditional credit bureaus may not track.
-- This data forms the basis for alternative credit scoring.

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    borrower_id VARCHAR(36) NOT NULL,
    transaction_date DATETIME NOT NULL,
    amount REAL NOT NULL,
    category VARCHAR(50) NOT NULL,
    description VARCHAR(255),

    -- Foreign key constraint
    FOREIGN KEY (borrower_id) REFERENCES borrowers(borrower_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_transactions_borrower_id ON transactions(borrower_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);

-- Composite index for common query patterns (borrower + date range)
CREATE INDEX IF NOT EXISTS idx_transactions_borrower_date
    ON transactions(borrower_id, transaction_date);

-- ============================================================================
-- LABELS TABLE
-- ============================================================================
-- Stores the target variable for supervised learning: whether a borrower
-- defaulted on a loan (ground truth) and the model's predicted probability.

CREATE TABLE IF NOT EXISTS labels (
    borrower_id VARCHAR(36) PRIMARY KEY,
    default_label BOOLEAN NOT NULL,
    default_probability REAL,
    label_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Foreign key constraint
    FOREIGN KEY (borrower_id) REFERENCES borrowers(borrower_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,

    -- Ensure probability is between 0 and 1
    CHECK (default_probability IS NULL OR (default_probability >= 0 AND default_probability <= 1))
);

-- ============================================================================
-- VIEWS
-- ============================================================================
-- Useful views for common queries and analytics

-- View: Borrowers with their label and transaction count
CREATE VIEW IF NOT EXISTS borrower_summary AS
SELECT
    b.borrower_id,
    b.created_at,
    b.age,
    b.income,
    b.employment_status,
    COUNT(t.transaction_id) AS transaction_count,
    COALESCE(SUM(t.amount), 0) AS total_transaction_amount,
    l.default_label,
    l.default_probability
FROM borrowers b
LEFT JOIN transactions t ON b.borrower_id = t.borrower_id
LEFT JOIN labels l ON b.borrower_id = l.borrower_id
GROUP BY b.borrower_id;

-- View: Monthly transaction aggregates by category
CREATE VIEW IF NOT EXISTS monthly_transaction_summary AS
SELECT
    borrower_id,
    strftime('%Y-%m', transaction_date) AS month,
    category,
    COUNT(*) AS transaction_count,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount,
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount
FROM transactions
GROUP BY borrower_id, month, category;

-- ============================================================================
-- SAMPLE QUERIES
-- ============================================================================

-- Example 1: Get all transactions for a specific borrower
-- SELECT * FROM transactions
-- WHERE borrower_id = '<uuid>'
-- ORDER BY transaction_date DESC;

-- Example 2: Get borrowers with high default probability
-- SELECT b.*, l.default_probability
-- FROM borrowers b
-- JOIN labels l ON b.borrower_id = l.borrower_id
-- WHERE l.default_probability > 0.7
-- ORDER BY l.default_probability DESC;

-- Example 3: Get transaction statistics by category
-- SELECT
--     category,
--     COUNT(*) AS count,
--     AVG(amount) AS avg_amount,
--     SUM(amount) AS total_amount
-- FROM transactions
-- GROUP BY category
-- ORDER BY count DESC;

-- Example 4: Get borrowers with no transactions (potential data quality issue)
-- SELECT b.*
-- FROM borrowers b
-- LEFT JOIN transactions t ON b.borrower_id = t.borrower_id
-- WHERE t.transaction_id IS NULL;
