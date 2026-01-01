"""
Complete data generation pipeline for VantageFlow AI.
Generates borrower profiles, transactions, and labels.
"""
import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation import profiles
from src.data_generation.synthesizer import TransactionSynthesizer
from src.data_generation.labels import CreditRiskLabeler
from sqlalchemy import create_engine
import pandas as pd

def main():
    """Run complete data generation pipeline."""
    
    # Configuration
    NUM_BORROWERS = 1000
    DB_PATH = "data/vantageflow.db"
    
    print("=" * 60)
    print("VantageFlow AI - Data Generation Pipeline")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create database engine
    engine = create_engine(f'sqlite:///{DB_PATH}')
    
    # Step 1: Generate borrower profiles
    print(f"\n[1/3] Generating {NUM_BORROWERS} borrower profiles...")
    borrowers_df = profiles.generate_borrower_profiles(n_profiles=NUM_BORROWERS)
    
    # Save to database
    borrowers_df.to_sql('borrowers', engine, if_exists='replace', index=False)
    print(f"✓ Generated {len(borrowers_df)} borrower profiles")
    print(f"  - Saved to {DB_PATH} (borrowers table)")
    
    # Step 2: Generate transactions
    print(f"\n[2/3] Generating 12 months of transactions for each borrower...")
    
    all_transactions = []
    for idx, borrower in borrowers_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  - Processing borrower {idx + 1}/{NUM_BORROWERS}...")
        
        # Create synthesizer with borrower profile
        borrower_dict = borrower.to_dict()
        transaction_synthesizer = TransactionSynthesizer(borrower_profile=borrower_dict)
        
        # Generate transactions
        transactions = transaction_synthesizer.generate_transactions()
        all_transactions.extend(transactions)
    
    transactions_df = pd.DataFrame(all_transactions)
    transactions_df.to_sql('transactions', engine, if_exists='replace', index=False)
    
    print(f"✓ Generated {len(transactions_df)} transactions")
    print(f"  - Avg transactions per borrower: {len(transactions_df) / NUM_BORROWERS:.1f}")
    print(f"  - Saved to {DB_PATH} (transactions table)")
    
    # Step 3: Generate labels
    print(f"\n[3/3] Generating default risk labels...")
    labeler = CreditRiskLabeler(default_rate=0.20)
    labels_df = labeler.assign_labels_batch(transactions_df=transactions_df)
    
    # Save to database
    labels_df.to_sql('labels', engine, if_exists='replace', index=False)
    
    default_rate = labels_df['default_label'].mean()
    print(f"✓ Generated labels for {len(labels_df)} borrowers")
    print(f"  - Default rate: {default_rate:.1%}")
    print(f"  - Saved to {DB_PATH} (labels table)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Data Generation Complete!")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"Borrowers: {len(borrowers_df)}")
    print(f"Transactions: {len(transactions_df)}")
    print(f"Default rate: {default_rate:.1%}")
    print(f"\nNext step: Run feature extraction")
    print(f"  python scripts/extract_features.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
