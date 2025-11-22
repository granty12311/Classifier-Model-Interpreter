"""
Create simplified version of loan_clean.csv with:
- Numeric transformations (term)
- Remove low-value columns (zip_code, addr_state)
- Simplify categorical columns
- Create interpretable target variables
"""
import pandas as pd
import numpy as np

print("Loading cleaned loan data...")
df = pd.read_csv('loan_clean.csv', low_memory=False)
print(f"Original shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# ANALYZE LOW VARIANCE COLUMNS
# ============================================================================
print("\n" + "="*80)
print("LOW VARIANCE COLUMN ANALYSIS")
print("="*80)

# Calculate unique value ratios
unique_ratios = df.nunique() / len(df)
low_variance = unique_ratios[unique_ratios < 0.01].sort_values()

print(f"\nColumns with <1% unique values ({len(low_variance)} columns):")
print("\nRECOMMENDATIONS FOR SIMPLIFICATION OR REMOVAL:")
print("-" * 80)

for col, ratio in low_variance.items():
    n_unique = df[col].nunique()
    print(f"\n{col:30s} {n_unique:3d} unique values ({ratio*100:.3f}%)")

    # Show value distribution for very low variance columns
    if n_unique <= 10:
        print(f"  Distribution:")
        value_counts = df[col].value_counts()
        for val, count in value_counts.head(10).items():
            pct = count / len(df) * 100
            print(f"    {str(val):40s} {count:8,} ({pct:5.2f}%)")

        # Recommendations
        if n_unique == 1:
            print(f"  ⚠️  RECOMMEND: REMOVE (constant)")
        elif n_unique == 2:
            print(f"  💡 RECOMMEND: Keep as binary (already simple)")
        elif n_unique <= 5:
            print(f"  💡 RECOMMEND: Keep as-is or simplify further")
        else:
            print(f"  💡 RECOMMEND: Consider grouping rare categories")

# ============================================================================
# CREATE SIMPLIFIED VERSION
# ============================================================================
print("\n" + "="*80)
print("CREATING SIMPLIFIED VERSION")
print("="*80)

df_simple = df.copy()

# Remove specified columns
columns_to_remove = ['zip_code', 'addr_state']
print(f"\n1. Removing columns: {columns_to_remove}")
df_simple = df_simple.drop(columns=columns_to_remove)

# Transform term to numeric
print("\n2. Transforming 'term' to numeric (remove 'months')")
print(f"   Before: {df_simple['term'].unique()}")
df_simple['term'] = df_simple['term'].str.extract('(\d+)').astype(int)
print(f"   After:  {df_simple['term'].unique()}")

# Simplify verification_status
print("\n3. Simplifying 'verification_status'")
print(f"   Before: {df_simple['verification_status'].unique()}")
verification_map = {
    'Source Verified': 'sv',
    'Verified': 'v',
    'Not Verified': 'nv'
}
df_simple['verification_status'] = df_simple['verification_status'].map(verification_map)
print(f"   After:  {df_simple['verification_status'].unique()}")

# Simplify verification_status_joint (if exists and has data)
if 'verification_status_joint' in df_simple.columns:
    print("\n4. Simplifying 'verification_status_joint'")
    non_null_count = df_simple['verification_status_joint'].notna().sum()
    print(f"   Non-null values: {non_null_count:,}")
    if non_null_count > 0:
        df_simple['verification_status_joint'] = df_simple['verification_status_joint'].map(verification_map)

# Simplify home_ownership
print("\n5. Simplifying 'home_ownership'")
print(f"   Before: {df_simple['home_ownership'].value_counts().to_dict()}")
# Group rare categories
home_map = {
    'RENT': 'rent',
    'MORTGAGE': 'mortgage',
    'OWN': 'own',
    'OTHER': 'other',
    'NONE': 'other',
    'ANY': 'other'
}
df_simple['home_ownership'] = df_simple['home_ownership'].map(
    lambda x: home_map.get(x, 'other') if pd.notna(x) else x
)
print(f"   After:  {df_simple['home_ownership'].value_counts().to_dict()}")

# Simplify application_type
print("\n6. Simplifying 'application_type'")
print(f"   Before: {df_simple['application_type'].value_counts().to_dict()}")
app_type_map = {
    'INDIVIDUAL': 'individual',
    'JOINT': 'joint'
}
df_simple['application_type'] = df_simple['application_type'].map(app_type_map)
print(f"   After:  {df_simple['application_type'].value_counts().to_dict()}")

# Simplify initial_list_status
print("\n7. Simplifying 'initial_list_status'")
print(f"   Before: {df_simple['initial_list_status'].value_counts().to_dict()}")
list_status_map = {
    'f': 'fractional',
    'w': 'whole'
}
df_simple['initial_list_status'] = df_simple['initial_list_status'].map(list_status_map)
print(f"   After:  {df_simple['initial_list_status'].value_counts().to_dict()}")

# Transform pymnt_plan to binary
print("\n8. Transforming 'pymnt_plan' to binary (0/1)")
print(f"   Before: {df_simple['pymnt_plan'].value_counts().to_dict()}")
df_simple['pymnt_plan'] = (df_simple['pymnt_plan'] == 'y').astype(int)
print(f"   After:  {df_simple['pymnt_plan'].value_counts().to_dict()}")

# ============================================================================
# CREATE TARGET VARIABLES
# ============================================================================
print("\n" + "="*80)
print("CREATING TARGET VARIABLES")
print("="*80)

print("\nOriginal loan_status distribution:")
print(df_simple['loan_status'].value_counts())

# Create charge-off flag
print("\n9. Creating 'chargeoff_flag' target variable:")
print("   0 = Good (Current, Fully Paid)")
print("   1 = Bad (Charged Off, Default)")
print("   2 = Exclude (Late, Issued, In Grace Period)")

chargeoff_map = {
    # Good loans (0)
    'Current': 0,
    'Fully Paid': 0,
    'Does not meet the credit policy. Status:Fully Paid': 0,

    # Bad loans (1)
    'Charged Off': 1,
    'Default': 1,
    'Does not meet the credit policy. Status:Charged Off': 1,

    # Exclude - uncertain status (2)
    'Late (31-120 days)': 2,
    'Late (16-30 days)': 2,
    'In Grace Period': 2,
    'Issued': 2,
}

df_simple['chargeoff_flag'] = df_simple['loan_status'].map(chargeoff_map)

print("\nchargeoff_flag distribution:")
flag_dist = df_simple['chargeoff_flag'].value_counts().sort_index()
for flag, count in flag_dist.items():
    pct = count / len(df_simple) * 100
    if flag == 0:
        label = "Good (Current/Fully Paid)"
    elif flag == 1:
        label = "Bad (Charged Off/Default)"
    else:
        label = "Exclude (Uncertain)"
    print(f"   {flag}: {label:30s} {count:8,} ({pct:5.2f}%)")

# Create binary good/bad flag (excluding uncertain)
print("\n10. Creating 'is_bad' binary target (excluding uncertain statuses):")
df_simple['is_bad'] = df_simple['chargeoff_flag'].apply(
    lambda x: np.nan if x == 2 else x
)

print("\nis_bad distribution (NaN = excluded):")
is_bad_dist = df_simple['is_bad'].value_counts().sort_index()
for flag, count in is_bad_dist.items():
    pct = count / len(df_simple) * 100
    label = "Good" if flag == 0 else "Bad"
    print(f"   {int(flag)}: {label:30s} {count:8,} ({pct:5.2f}%)")
excluded = df_simple['is_bad'].isna().sum()
print(f"   Excluded:                      {excluded:8,} ({excluded/len(df_simple)*100:5.2f}%)")

# Calculate bad rate (excluding uncertain)
modeling_data = df_simple[df_simple['chargeoff_flag'] != 2]
bad_rate = modeling_data['is_bad'].mean() * 100
print(f"\nBad rate (for modeling subset): {bad_rate:.2f}%")

# ============================================================================
# SAVE SIMPLIFIED VERSION
# ============================================================================
print("\n" + "="*80)
print("SAVING SIMPLIFIED VERSION")
print("="*80)

print(f"\nFinal shape: {df_simple.shape[0]:,} rows × {df_simple.shape[1]} columns")

# Move target variables to the front (after id)
cols = df_simple.columns.tolist()
cols.remove('chargeoff_flag')
cols.remove('is_bad')
cols.insert(1, 'chargeoff_flag')
cols.insert(2, 'is_bad')
df_simple = df_simple[cols]

# Save
output_file = 'loan_simple.csv'
print(f"\nSaving to {output_file}...")
df_simple.to_csv(output_file, index=False)

# Create documentation
print("\nCreating documentation...")
with open('loan_simple_transformations.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SIMPLIFIED LOAN DATASET TRANSFORMATIONS\n")
    f.write("="*80 + "\n\n")

    f.write(f"Original (loan_clean.csv):  {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    f.write(f"Simplified (loan_simple.csv): {df_simple.shape[0]:,} rows × {df_simple.shape[1]} columns\n\n")

    f.write("="*80 + "\n")
    f.write("TRANSFORMATIONS APPLIED\n")
    f.write("="*80 + "\n\n")

    f.write("1. COLUMNS REMOVED (2)\n")
    f.write("   - zip_code\n")
    f.write("   - addr_state\n\n")

    f.write("2. NUMERIC TRANSFORMATIONS\n")
    f.write("   - term: ' 36 months' -> 36, ' 60 months' -> 60\n\n")

    f.write("3. CATEGORICAL SIMPLIFICATIONS\n")
    f.write("   - verification_status: 'Source Verified' -> 'sv', 'Verified' -> 'v', 'Not Verified' -> 'nv'\n")
    f.write("   - verification_status_joint: Same as above\n")
    f.write("   - home_ownership: Lowercase, group rare (OTHER, NONE, ANY -> 'other')\n")
    f.write("   - application_type: 'INDIVIDUAL' -> 'individual', 'JOINT' -> 'joint'\n")
    f.write("   - initial_list_status: 'f' -> 'fractional', 'w' -> 'whole'\n")
    f.write("   - pymnt_plan: 'y' -> 1, 'n' -> 0\n\n")

    f.write("4. TARGET VARIABLES CREATED\n")
    f.write("   - chargeoff_flag:\n")
    f.write("       0 = Good (Current, Fully Paid)\n")
    f.write("       1 = Bad (Charged Off, Default)\n")
    f.write("       2 = Exclude (Late, Issued, In Grace Period)\n\n")
    f.write("   - is_bad (binary for modeling):\n")
    f.write("       0 = Good\n")
    f.write("       1 = Bad\n")
    f.write("       NaN = Excluded (uncertain status)\n\n")

    f.write(f"   Bad rate (modeling subset): {bad_rate:.2f}%\n\n")

    f.write("="*80 + "\n")
    f.write("COLUMN ORDER\n")
    f.write("="*80 + "\n\n")

    for i, col in enumerate(df_simple.columns, 1):
        dtype = df_simple[col].dtype
        null_pct = df_simple[col].isnull().sum() / len(df_simple) * 100
        f.write(f"{i:2d}. {col:35s} {str(dtype):10s} ({null_pct:5.1f}% missing)\n")

print("\n✅ Simplification complete!")
print(f"\nOutput files:")
print(f"  - {output_file}")
print(f"  - loan_simple_transformations.txt")
print(f"\nColumn count: {df.shape[1]} -> {df_simple.shape[1]} ({df.shape[1] - df_simple.shape[1]} removed)")
print(f"\nFirst few columns:")
print(df_simple.iloc[:, :10].head())
