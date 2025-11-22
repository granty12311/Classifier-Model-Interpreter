# Loan Data Column Removal Recommendations

## Dataset Overview
- **Total rows**: 887,379
- **Total columns**: 74
- **Target variable**: loan_status (10 categories)

## Recommended Columns to Remove

### Category 1: Identifiers (3 columns)
**Reason**: No predictive value, 100% unique
- `id` - Unique loan identifier
- `member_id` - Unique member identifier
- `url` - LC page URL (100% unique)

### Category 2: Data Leakage - Post-Loan Information (13 columns)
**Reason**: Not available at loan origination time, would cause data leakage
- `out_prncp` - Outstanding principal (known only after loan is issued)
- `out_prncp_inv` - Outstanding principal for investors
- `total_pymnt` - Total payments received
- `total_pymnt_inv` - Total payments to investors
- `total_rec_prncp` - Principal received to date
- `total_rec_int` - Interest received to date
- `total_rec_late_fee` - Late fees received
- `recoveries` - Post charge-off recovery
- `collection_recovery_fee` - Collection fees
- `last_pymnt_d` - Last payment date
- `last_pymnt_amnt` - Last payment amount
- `next_pymnt_d` - Next payment date
- `last_credit_pull_d` - Last credit pull date

### Category 3: High Missing Values >97% (17 columns)
**Reason**: Too sparse to be useful, mostly from joint applications and recent account activity
- `annual_inc_joint` (99.94% missing)
- `dti_joint` (99.94% missing)
- `verification_status_joint` (99.94% missing)
- `il_util` (97.90% missing)
- `mths_since_rcnt_il` (97.65% missing)
- `inq_fi` (97.59% missing)
- `open_rv_24m` (97.59% missing)
- `open_acc_6m` (97.59% missing)
- `all_util` (97.59% missing)
- `inq_last_12m` (97.59% missing)
- `total_cu_tl` (97.59% missing)
- `open_il_12m` (97.59% missing)
- `max_bal_bc` (97.59% missing)
- `open_il_6m` (97.59% missing)
- `open_il_24m` (97.59% missing)
- `open_rv_12m` (97.59% missing)
- `total_bal_il` (97.59% missing)

### Category 4: Text/Description Fields (2 columns)
**Reason**: Free-text fields requiring NLP processing, high missing values for desc
- `desc` (85.80% missing) - Loan description
- `title` - Loan title (would need text processing)

### Category 5: Constant/No Variance (1 column)
**Reason**: No predictive value - all rows have same value
- `policy_code` - Only 1 unique value (1.0 for all rows)

### Category 6: Quasi-Identifier (1 column)
**Reason**: Partial zip code, limited predictive value, privacy concerns
- `zip_code` - First 3 digits only (935 unique values)

## Columns to KEEP but Consider

### Keep with Caution:
- **`purpose`** (14 categories) - User mentioned removing, but could be predictive
- **`emp_title`** (5.8% missing) - Could be useful if categorized/engineered
- **`mths_since_last_delinq`** (51.20% missing) - Moderate missing, but could be informative
- **`mths_since_last_record`** (84.56% missing) - High missing but might indicate clean record
- **`mths_since_last_major_derog`** (75.02% missing) - High missing but could be useful

### Potentially Useful Features to Retain:
- Credit history: `dti`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`
- Revolving credit: `revol_bal`, `revol_util`, `total_acc`
- Loan details: `loan_amnt`, `funded_amnt`, `term`, `int_rate`, `installment`
- Grades: `grade`, `sub_grade`
- Demographics: `emp_length`, `home_ownership`, `annual_inc`, `addr_state`
- Status: `verification_status`, `issue_d`, `initial_list_status`, `application_type`

## Summary

### Recommended for Removal: 37 columns total
- Identifiers: 3
- Data leakage: 13
- High missing (>97%): 17
- Text fields: 2
- Constant: 1
- Quasi-identifier: 1

### After Removal: 37 columns remaining (from 74)
- Reduction: 50% of columns removed
- Remaining columns focus on credit risk factors available at origination

## Questions for Review

1. **Purpose column**: You mentioned removing it, but it has 14 categories and could be predictive. Should we keep it?
2. **emp_title**: Could be categorized into job sectors. Keep or remove?
3. **Moderate missing value columns** (51-85% missing):
   - `mths_since_last_delinq` (51%)
   - `mths_since_last_major_derog` (75%)
   - `mths_since_last_record` (85%)
   - These could indicate "no delinquency/record" when missing. Keep or remove?
