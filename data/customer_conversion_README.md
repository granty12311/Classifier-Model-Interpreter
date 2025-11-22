# Customer Conversion Dataset

## Overview

Synthetic dataset simulating **online course trial-to-paid conversion** for testing model interpretation tools.

- **Rows**: 5,000
- **Columns**: 18
- **Conversion Rate**: 35.98%
- **Use Case**: Testing SHAP values and model interpretation

## Dataset Description

This dataset represents customers who signed up for a free trial of an online course platform. The goal is to predict whether they convert to a paid subscription (`converted=1`) or not (`converted=0`).

## Features

### Target Variable
- **converted** (0/1): Whether customer converted to paid (1=yes, 0=no)

### Engagement Metrics (Strongest Predictors)
- **pages_viewed**: Number of course pages viewed during trial
- **time_on_site_mins**: Total time spent on platform (minutes)
- **videos_watched**: Number of course videos watched
- **email_opens**: Number of marketing emails opened
- **session_count**: Number of login sessions during trial

### Marketing Features
- **discount_offered**: Discount percentage offered (0%, 10%, 20%, 30%)
- **referral_source**: How customer found the platform
  - organic_search, paid_ads, social_media, email_campaign, referral, direct

### Demographics
- **age**: Customer age (18-70)
- **country**: Customer country (USA, UK, Canada, Australia, Germany, France, Other)
- **occupation**: Employment status (student, professional, self_employed, retired, unemployed)
- **device_type**: Primary device used (mobile, desktop, tablet)

### Historical Behavior
- **previous_courses**: Number of courses completed on platform before (0-3)
- **account_age_days**: Days since account creation
- **days_since_signup**: Days since trial signup

### Temporal
- **signup_month**: Month of trial signup (1-12)
- **signup_day_of_week**: Day of week (1=Monday, 7=Sunday)

## Known Patterns (for Testing Interpretation)

The dataset was generated with the following interpretable patterns:

1. **Engagement is the strongest driver**: More pages viewed, videos watched → higher conversion
2. **Discount effect**: Larger discounts → higher conversion (non-linear)
3. **Loyalty effect**: Previous course completions → higher conversion
4. **Referral quality**: Referrals and email campaigns → highest conversion
5. **Age effect**: Inverted U-shape (25-35 age group converts best)
6. **Occupation effect**: Professionals convert best, unemployed least
7. **Device effect**: Desktop users convert slightly better
8. **Interaction effect**: Discounts are MORE effective for students
9. **Recency effect**: Early engagement (low days_since_signup) → better conversion

## Usage

```python
import pandas as pd

# Load data
df = pd.read_csv('customer_conversion.csv')

# Separate features and target
X = df.drop(['customer_id', 'converted'], axis=1)
y = df['converted']

# Check conversion rate
print(f'Conversion rate: {y.mean()*100:.2f}%')
```

## Column Reference

| Column | Type | Description | Missing |
|--------|------|-------------|----------|
| customer_id | int64 | Unique customer identifier | 0% |
| converted | int64 | Target: 1=converted, 0=not converted | 0% |
| pages_viewed | int64 | Number of pages viewed | 0% |
| time_on_site_mins | float64 | Time on site (minutes) | 0% |
| videos_watched | int64 | Number of videos watched | 0% |
| email_opens | float64 | Number of emails opened | 5.2% |
| session_count | int64 | Number of sessions | 0% |
| discount_offered | int64 | Discount % (0, 10, 20, 30) | 0% |
| referral_source | object | Acquisition channel | 0% |
| age | int64 | Customer age | 0% |
| country | object | Customer country | 0% |
| occupation | object | Employment status | 0% |
| device_type | object | Primary device | 0% |
| previous_courses | float64 | Prior course completions | 3.2% |
| account_age_days | int64 | Account age in days | 0% |
| days_since_signup | int64 | Days since trial start | 0% |
| signup_month | int64 | Signup month (1-12) | 0% |
| signup_day_of_week | int64 | Day of week (1=Mon) | 0% |

## Notes

- This is a synthetic dataset created specifically for testing model interpretation tools
- Patterns are intentionally clear and interpretable
- Small amount of missing data (~3-5%) in email_opens and previous_courses
- Class balance is realistic (~20% conversion rate)
- Features include numeric, categorical, and ordinal types
