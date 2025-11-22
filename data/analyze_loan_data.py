"""
Script to analyze loan data and recommend columns to remove for modeling
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))

import pandas as pd
import numpy as np

# Load data
print("Loading loan data...")
df = pd.read_csv('loan.csv', low_memory=False)

print(f"\nDataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nColumn names:\n{df.columns.tolist()}")

# Analyze missing values
print("\n" + "="*80)
print("MISSING VALUE ANALYSIS")
print("="*80)
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
high_missing = missing_pct[missing_pct > 50]
print(f"\nColumns with >50% missing values ({len(high_missing)} columns):")
for col, pct in high_missing.items():
    print(f"  {col:30s} {pct:6.2f}%")

# Analyze unique values (potential identifiers)
print("\n" + "="*80)
print("IDENTIFIER COLUMNS (Unique or Near-Unique)")
print("="*80)
unique_ratios = df.nunique() / len(df)
high_unique = unique_ratios[unique_ratios > 0.95].sort_values(ascending=False)
print(f"\nColumns with >95% unique values ({len(high_unique)} columns):")
for col, ratio in high_unique.items():
    print(f"  {col:30s} {ratio*100:6.2f}% unique ({df[col].nunique():,} values)")

# Analyze constant/near-constant columns
print("\n" + "="*80)
print("CONSTANT OR NEAR-CONSTANT COLUMNS")
print("="*80)
low_variance = unique_ratios[unique_ratios < 0.01].sort_values()
print(f"\nColumns with <1% unique values ({len(low_variance)} columns):")
for col, ratio in low_variance.items():
    print(f"  {col:30s} {df[col].nunique()} unique values")
    if df[col].nunique() <= 5:
        print(f"    Values: {df[col].value_counts().to_dict()}")

# Analyze target variable
print("\n" + "="*80)
print("TARGET VARIABLE: loan_status")
print("="*80)
print(df['loan_status'].value_counts())

# Check data leakage columns (post-loan information)
print("\n" + "="*80)
print("POTENTIAL DATA LEAKAGE COLUMNS")
print("="*80)
leakage_candidates = [
    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
    'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d'
]
print("\nColumns that contain post-loan information (would not be available at loan origination):")
for col in leakage_candidates:
    if col in df.columns:
        print(f"  - {col}")

# Text/description columns
print("\n" + "="*80)
print("TEXT/DESCRIPTION COLUMNS")
print("="*80)
text_columns = ['url', 'desc', 'title', 'emp_title']
print("\nColumns with free-text descriptions:")
for col in text_columns:
    if col in df.columns:
        null_pct = df[col].isnull().sum() / len(df) * 100
        print(f"  - {col:20s} ({null_pct:.1f}% missing)")

# Date columns
print("\n" + "="*80)
print("DATE COLUMNS")
print("="*80)
date_columns = [col for col in df.columns if 'd' in col.lower() or 'date' in col.lower()]
print(f"\nPotential date columns ({len(date_columns)} columns):")
for col in date_columns:
    print(f"  - {col}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
