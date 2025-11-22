# Data Directory

## Files

### Source Data (from ml-modeling-pipeline)
- **loan.csv** (422 MB, 887,379 rows × 74 columns)
  - Original LendingClub loan dataset
  - Source: `/home/granty1231/ml-modeling-pipeline/data/raw/`

- **loan_data_header.txt**
  - Detailed column definitions and dataset documentation
  - Describes all 74 original columns with examples

### Cleaned Data
- **loan_clean.csv** (293 MB, 887,379 rows × 68 columns)
  - Cleaned version ready for modeling
  - 6 columns removed (identifiers, text fields, constants)
  - Columns reorganized with data leakage indicators clearly marked

- **loan_clean_column_info.txt**
  - Documentation of cleaned dataset structure
  - Lists all 68 columns organized by category
  - Shows missing value percentages
  - Identifies data leakage columns (positions 20-33)

### Analysis Scripts
- **analyze_loan_data.py**
  - Analyzes original dataset for missing values, unique values, etc.
  - Generates recommendations for data cleaning

- **clean_loan_data.py**
  - Performs data cleaning and column reorganization
  - Creates loan_clean.csv and loan_clean_column_info.txt

### Analysis Results
- **column_removal_recommendations.md**
  - Detailed analysis and recommendations for column removal
  - Categorizes columns by removal reason

## Data Cleaning Summary

### Removed Columns (6 total)
1. `member_id` - Redundant identifier
2. `url` - Unique URL (no predictive value)
3. `emp_title` - Free-text job titles
4. `title` - Free-text loan titles
5. `desc` - Loan descriptions (85.8% missing)
6. `policy_code` - Constant value (all = 1.0)

### Kept Important Columns
- **Primary key**: `id`
- **Loan purpose**: `purpose` (14 categories)
- **Columns with high missing values**: Retained as missing values may be informative

### Column Organization

The cleaned dataset is organized into clear sections:

1. **Primary Key** (1 column)
2. **Loan Identification & Amounts** (4 columns)
3. **Interest & Payment** (4 columns)
4. **Borrower Information** (4 columns)
5. **Loan Details** (6 columns)
6. **⚠️  Loan Performance (DATA LEAKAGE)** (10 columns, positions 20-29)
7. **⚠️  Payment Tracking (DATA LEAKAGE)** (4 columns, positions 30-33)
8. **Credit History** (6 columns, positions 34-39)
9. **Account Information** (5 columns)
10. **Additional Metrics** (9 columns)
11. **Recent Account Activity** (13 columns)
12. **Inquiries** (2 columns)

### Data Leakage Warning

**Columns 20-33** contain loan performance and payment information that would NOT be available at loan origination time. These should be EXCLUDED when training predictive models for loan approval/risk assessment.

Use these columns only for:
- Post-loan performance analysis
- Collection strategy models
- Loss given default (LGD) modeling

### Features Available at Origination

**Columns 1-19 and 34-68** (54 total) contain information available at the time of loan application and can be used for predictive modeling.

## Usage

### Load Cleaned Data
```python
import pandas as pd

# Load cleaned dataset
df = pd.read_csv('data/loan_clean.csv')

# For modeling, exclude data leakage columns
leakage_cols = df.columns[19:33]  # positions 20-33 (0-indexed)
df_modeling = df.drop(columns=leakage_cols)

# Or select specific column ranges
primary_features = df.iloc[:, [0] + list(range(1, 19)) + list(range(33, 68))]
```

### Column Reference
See `loan_clean_column_info.txt` for complete column listing with missing value percentages.

## File Sizes

| File | Size | Rows | Columns |
|------|------|------|---------|
| loan.csv | 422 MB | 887,379 | 74 |
| loan_clean.csv | 293 MB | 887,379 | 68 |

## Notes

- CSV files are excluded from git (see .gitignore)
- Scripts and documentation are version controlled
- Raw data should be preserved - always work with loan_clean.csv for modeling
