# Additional Simplification Recommendations

Based on the low variance analysis, here are additional recommendations to further simplify the dataset.

## Current Status
- **loan_simple.csv**: 887,379 rows × 68 columns
- **Bad rate**: 5.50% (modeling subset)
- **Target variables created**: `chargeoff_flag` and `is_bad`

## Recommended Additional Simplifications

### 1. Remove High Missing Value Columns (>97% missing)

**Joint Application Fields (99.9% missing)**
- `annual_inc_joint`
- `dti_joint`
- `verification_status_joint`
- **Reason**: Only 511 rows (0.06%) with data, not useful for general modeling

**Recent Account Activity Fields (97.6-97.9% missing)**
- `open_acc_6m`, `open_il_6m`, `open_il_12m`, `open_il_24m`
- `mths_since_rcnt_il`, `total_bal_il`, `il_util`
- `open_rv_12m`, `open_rv_24m`, `max_bal_bc`, `all_util`
- `inq_fi`, `total_cu_tl`, `inq_last_12m`
- **Reason**: Too sparse to provide reliable signal
- **Impact**: Remove 16 columns

### 2. Simplify Grade to Numeric

**Current**: Letters A-G (7 categories)
**Recommendation**: Convert to numeric 1-7 or keep sub_grade instead
```python
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
# Or drop grade and keep only sub_grade (more granular)
```

### 3. Simplify Employment Length

**Current**: 11 categories (< 1 year, 1 year, 2 years, ..., 10+ years, n/a)
**Recommendation**: Group into bins
```python
emp_length_bins:
  '0-2 years'    (< 1 year, 1 year, 2 years)
  '3-5 years'    (3, 4, 5 years)
  '6-9 years'    (6, 7, 8, 9 years)
  '10+ years'    (10+ years)
  'unknown'      (n/a)
```

### 4. Cap/Bin Extreme Value Columns

**Delinquency Counts**
- `acc_now_delinq`: 99.5% are 0, max is 14
  - **Recommendation**: Cap at 2+ (0, 1, 2+)

- `delinq_2yrs`: Similar distribution
  - **Recommendation**: Cap at 3+ (0, 1, 2, 3+)

- `pub_rec`: 32 unique values, most are 0
  - **Recommendation**: Cap at 2+ (0, 1, 2+)

### 5. Simplify Purpose Categories

**Current**: 14 categories
**Recommendation**: Group into 5-6 main categories
```python
purpose_groups:
  'debt_consolidation'  (keep as-is, largest category)
  'credit_card'         (keep as-is, second largest)
  'home_improvement'    (home_improvement, major_purchase)
  'personal'            (wedding, vacation, moving, other)
  'business'            (small_business)
  'auto'                (car, motorcycle)
```

### 6. Binary Flags from Sparse Columns

**Collections/Major Derogatory**
- `collections_12_mths_ex_med`: Convert to binary (0 vs >0)
- `mths_since_last_major_derog`: Convert to binary (has_major_derog: yes/no)
- `mths_since_last_record`: Convert to binary (has_public_record: yes/no)
- `mths_since_last_delinq`: Convert to binary (has_recent_delinq: yes/no)

**Reasoning**: The existence of the event is more important than timing for many models

### 7. Remove Near-Constant Columns

**pymnt_plan**: Only 10 out of 887,379 (0.001%) have payment plans
- **Recommendation**: Remove (insufficient variation)

**application_type**: 99.94% are individual, 0.06% joint
- **Recommendation**: Remove or keep as binary flag if needed

## Summary of Additional Cuts

### Recommended Removals (18 columns total)
**High Missing (16 columns)**:
- Joint fields: `annual_inc_joint`, `dti_joint`, `verification_status_joint`
- Recent activity: `open_acc_6m`, `open_il_6m`, `open_il_12m`, `open_il_24m`, `mths_since_rcnt_il`, `total_bal_il`, `il_util`, `open_rv_12m`, `open_rv_24m`, `max_bal_bc`, `all_util`, `inq_fi`, `total_cu_tl`, `inq_last_12m`

**Near-Constant (2 columns)**:
- `pymnt_plan` (0.001% variation)
- `application_type` (0.06% variation)

### After Additional Cuts
- **Current**: 68 columns
- **After removals**: 50 columns
- **After transformations**: ~45-48 columns (some grouped/simplified)

## Modeling-Ready Dataset Structure

### Recommended Final Structure (48 columns)

**Target Variables (2)**
- `is_bad` (primary target)
- `chargeoff_flag` (for segmentation)

**Primary Key (1)**
- `id`

**Loan Characteristics (8)**
- `loan_amnt`, `funded_amnt`, `funded_amnt_inv`
- `term` (numeric: 36/60)
- `int_rate`, `installment`
- `grade_numeric` (1-7)
- `sub_grade`

**Borrower Demographics (4)**
- `emp_length_group` (binned)
- `home_ownership` (rent/mortgage/own/other)
- `annual_inc`
- `verification_status` (sv/v/nv)

**Loan Details (3)**
- `issue_d` (date)
- `purpose_group` (grouped categories)
- `initial_list_status` (fractional/whole)

**Credit History - Continuous (6)**
- `dti`
- `earliest_cr_line`
- `open_acc`
- `revol_bal`, `revol_util`
- `total_acc`

**Credit History - Counts (3)**
- `delinq_2yrs_capped` (0/1/2/3+)
- `pub_rec_capped` (0/1/2+)
- `inq_last_6mths`

**Credit History - Binary Flags (4)**
- `has_delinq_flag` (from mths_since_last_delinq)
- `has_public_record_flag` (from mths_since_last_record)
- `has_major_derog_flag` (from mths_since_last_major_derog)
- `has_collections_flag` (from collections_12_mths_ex_med)

**Current Balances (3)**
- `tot_coll_amt`
- `tot_cur_bal`
- `total_rev_hi_lim`

**Other (1)**
- `acc_now_delinq_capped` (0/1/2+)

**Data Leakage Columns (13)** - Keep but flag for exclusion
- All payment/performance columns (positions 20-33 in current dataset)

## Implementation Script

Would you like me to create a script to apply these additional simplifications?
The script would:
1. Remove high missing value columns (>97%)
2. Remove near-constant columns
3. Group/bin categorical variables
4. Create binary flags from sparse columns
5. Cap extreme values

This would result in a **loan_model_ready.csv** file optimized for modeling.
