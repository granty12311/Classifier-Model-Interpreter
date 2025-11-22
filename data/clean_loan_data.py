"""
Clean and reorganize loan data for modeling
- Remove identifiers (except id), text fields, and constant columns
- Reorder columns: place loan performance/payment metrics (data leakage)
  ahead of credit history for easy identification
"""
import pandas as pd
import numpy as np

print("Loading loan data...")
df = pd.read_csv('loan.csv', low_memory=False)
print(f"Original shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Define columns to remove
columns_to_remove = [
    'member_id',      # Identifier (keep id as primary key)
    'url',            # URL (unique identifier)
    'emp_title',      # Free text job title
    'title',          # Free text loan title
    'desc',           # Free text description (85% missing)
    'policy_code',    # Constant (all values = 1.0)
]

print(f"\nRemoving {len(columns_to_remove)} columns:")
for col in columns_to_remove:
    print(f"  - {col}")

# Remove columns
df_clean = df.drop(columns=columns_to_remove)

# Define column order based on loan_data_header.txt structure
# Reorganized to put loan performance/payment tracking (data leakage) early
column_order = [
    # PRIMARY KEY
    'id',

    # LOAN IDENTIFICATION & AMOUNTS
    'loan_amnt',
    'funded_amnt',
    'funded_amnt_inv',
    'term',

    # INTEREST & PAYMENT
    'int_rate',
    'installment',
    'grade',
    'sub_grade',

    # BORROWER INFORMATION
    'emp_length',
    'home_ownership',
    'annual_inc',
    'verification_status',

    # LOAN DETAILS
    'issue_d',
    'loan_status',
    'pymnt_plan',
    'purpose',
    'zip_code',
    'addr_state',

    # ============================================================
    # LOAN PERFORMANCE (POTENTIAL DATA LEAKAGE - MOVED UP)
    # ============================================================
    'initial_list_status',
    'out_prncp',
    'out_prncp_inv',
    'total_pymnt',
    'total_pymnt_inv',
    'total_rec_prncp',
    'total_rec_int',
    'total_rec_late_fee',
    'recoveries',
    'collection_recovery_fee',

    # PAYMENT TRACKING (POTENTIAL DATA LEAKAGE - MOVED UP)
    'last_pymnt_d',
    'last_pymnt_amnt',
    'next_pymnt_d',
    'last_credit_pull_d',

    # ============================================================
    # CREDIT HISTORY (FEATURES AVAILABLE AT ORIGINATION)
    # ============================================================
    'dti',
    'delinq_2yrs',
    'earliest_cr_line',
    'inq_last_6mths',
    'mths_since_last_delinq',
    'mths_since_last_record',

    # ACCOUNT INFORMATION
    'open_acc',
    'pub_rec',
    'revol_bal',
    'revol_util',
    'total_acc',

    # ADDITIONAL METRICS
    'collections_12_mths_ex_med',
    'mths_since_last_major_derog',
    'application_type',
    'annual_inc_joint',
    'dti_joint',
    'verification_status_joint',
    'acc_now_delinq',
    'tot_coll_amt',
    'tot_cur_bal',

    # RECENT ACCOUNT ACTIVITY
    'open_acc_6m',
    'open_il_6m',
    'open_il_12m',
    'open_il_24m',
    'mths_since_rcnt_il',
    'total_bal_il',
    'il_util',
    'open_rv_12m',
    'open_rv_24m',
    'max_bal_bc',
    'all_util',
    'total_rev_hi_lim',

    # INQUIRIES
    'inq_fi',
    'total_cu_tl',
    'inq_last_12m',
]

# Verify all columns are accounted for
missing_cols = set(df_clean.columns) - set(column_order)
if missing_cols:
    print(f"\n⚠️  Warning: Columns not in order list (will be added at end):")
    for col in missing_cols:
        print(f"  - {col}")
        column_order.append(col)

extra_cols = set(column_order) - set(df_clean.columns)
if extra_cols:
    print(f"\n⚠️  Warning: Columns in order list but not in data (will be skipped):")
    for col in extra_cols:
        print(f"  - {col}")
    column_order = [col for col in column_order if col in df_clean.columns]

# Reorder columns
df_clean = df_clean[column_order]

print(f"\nCleaned shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
print(f"Columns removed: {df.shape[1] - df_clean.shape[1]}")

# Save cleaned data
output_file = 'loan_clean.csv'
print(f"\nSaving cleaned data to {output_file}...")
df_clean.to_csv(output_file, index=False)

# Create column mapping document
print("\nCreating column mapping document...")
with open('loan_clean_column_info.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CLEANED LOAN DATASET COLUMN INFORMATION\n")
    f.write("="*80 + "\n\n")
    f.write(f"Original columns: {df.shape[1]}\n")
    f.write(f"Cleaned columns:  {df_clean.shape[1]}\n")
    f.write(f"Removed columns:  {df.shape[1] - df_clean.shape[1]}\n\n")

    f.write("="*80 + "\n")
    f.write("REMOVED COLUMNS\n")
    f.write("="*80 + "\n")
    for i, col in enumerate(columns_to_remove, 1):
        f.write(f"{i}. {col}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("COLUMN ORDER (BY SECTION)\n")
    f.write("="*80 + "\n\n")

    sections = [
        ("PRIMARY KEY", 0, 1),
        ("LOAN IDENTIFICATION & AMOUNTS", 1, 5),
        ("INTEREST & PAYMENT", 5, 9),
        ("BORROWER INFORMATION", 9, 13),
        ("LOAN DETAILS", 13, 19),
        ("LOAN PERFORMANCE (⚠️  DATA LEAKAGE)", 19, 29),
        ("PAYMENT TRACKING (⚠️  DATA LEAKAGE)", 29, 33),
        ("CREDIT HISTORY", 33, 39),
        ("ACCOUNT INFORMATION", 39, 44),
        ("ADDITIONAL METRICS", 44, 53),
        ("RECENT ACCOUNT ACTIVITY", 53, 66),
        ("INQUIRIES", 66, 69),
    ]

    for section_name, start, end in sections:
        f.write(f"\n{section_name}\n")
        f.write("-" * len(section_name) + "\n")
        section_cols = column_order[start:end]
        for i, col in enumerate(section_cols, start+1):
            # Get null percentage
            null_pct = df_clean[col].isnull().sum() / len(df_clean) * 100
            f.write(f"{i:2d}. {col:30s} ({null_pct:5.1f}% missing)\n")

print("\n✅ Data cleaning complete!")
print(f"\nOutput files:")
print(f"  - {output_file}")
print(f"  - loan_clean_column_info.txt")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nData leakage columns (positions 20-33):")
leakage_cols = column_order[19:33]
for col in leakage_cols:
    print(f"  - {col}")

print(f"\nFeatures available at origination (positions 34+):")
print(f"  Total: {len(column_order) - 33} columns")
print(f"\nSample columns:")
print(df_clean.head())
