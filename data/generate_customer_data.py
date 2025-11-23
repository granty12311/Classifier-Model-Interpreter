"""
Generate realistic customer conversion dataset with strong predictive patterns.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 5000

# Base features
data = {
    'customer_id': range(1, n_samples + 1),

    # Engagement metrics (strong predictors)
    'engagement_score': np.random.beta(2, 5, n_samples) * 100,  # 0-100, skewed low
    'pages_viewed': np.random.poisson(8, n_samples),
    'time_on_site_mins': np.random.gamma(3, 5, n_samples),
    'videos_watched': np.random.poisson(2, n_samples),
    'email_opens': np.random.poisson(2, n_samples),
    'session_count': np.random.poisson(5, n_samples),

    # Promotional
    'discount_offered': np.random.choice([0, 10, 20, 30], n_samples, p=[0.3, 0.2, 0.3, 0.2]),

    # Demographics
    'age': np.random.normal(40, 12, n_samples).clip(18, 70).astype(int),
    'occupation': np.random.choice(
        ['professional', 'student', 'retired', 'unemployed'],
        n_samples,
        p=[0.5, 0.25, 0.15, 0.1]
    ),
    'country': np.random.choice(
        ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Other'],
        n_samples,
        p=[0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]
    ),

    # Technical
    'device_type': np.random.choice(['desktop', 'mobile', 'tablet'], n_samples, p=[0.5, 0.4, 0.1]),
    'referral_source': np.random.choice(
        ['organic', 'paid_ads', 'social_media', 'email_campaign'],
        n_samples,
        p=[0.3, 0.3, 0.25, 0.15]
    ),

    # History
    'previous_courses': np.random.poisson(0.5, n_samples),
    'account_age_days': np.random.exponential(180, n_samples).clip(1, 730).astype(int),

    # Temporal
    'signup_month': np.random.randint(1, 13, n_samples),
    'signup_day_of_week': np.random.randint(0, 7, n_samples),
    'days_since_signup': np.random.exponential(30, n_samples).clip(0, 365).astype(int),
}

df = pd.DataFrame(data)

# Handle missing values for engagement (some users don't open emails)
email_missing_mask = np.random.random(n_samples) < 0.2
df.loc[email_missing_mask, 'email_opens'] = np.nan

# ============================================================================
# CONVERSION PROBABILITY - STRONG REALISTIC PATTERNS
# ============================================================================

conversion_prob = np.zeros(n_samples)

# 1. ENGAGEMENT SCORE - Strongest predictor (sigmoid relationship)
# High engagement → much higher conversion
engagement_effect = 1 / (1 + np.exp(-(df['engagement_score'] - 50) / 15))
conversion_prob += engagement_effect * 0.18  # Up to 18% boost

# 2. DISCOUNT - Strong linear relationship
discount_effect = df['discount_offered'] / 100 * 0.4
conversion_prob += discount_effect

# 3. PREVIOUS COURSES - Very strong predictor
# Existing customers convert much better
previous_effect = np.minimum(df['previous_courses'] * 0.10, 0.20)
conversion_prob += previous_effect

# 4. DEVICE TYPE - Desktop performs better
device_effect = df['device_type'].map({
    'desktop': 0.08,
    'mobile': 0.0,
    'tablet': 0.04
})
conversion_prob += device_effect

# 5. OCCUPATION - Professional/Student higher conversion
occupation_effect = df['occupation'].map({
    'professional': 0.06,
    'student': 0.08,
    'retired': 0.01,
    'unemployed': -0.04
})
conversion_prob += occupation_effect

# 6. AGE - Sweet spot 28-45 (inverted U-shape)
age_centered = (df['age'] - 36.5) / 15
age_effect = 0.06 * np.exp(-age_centered**2)
conversion_prob += age_effect

# 7. COUNTRY - USA/Canada higher conversion
country_effect = df['country'].map({
    'USA': 0.05,
    'Canada': 0.05,
    'UK': 0.02,
    'Australia': 0.02,
    'Germany': 0.0,
    'France': 0.0,
    'Other': -0.01
})
conversion_prob += country_effect

# 8. REFERRAL SOURCE - Organic and email better
referral_effect = df['referral_source'].map({
    'organic': 0.05,
    'email_campaign': 0.06,
    'paid_ads': 0.01,
    'social_media': 0.0
})
conversion_prob += referral_effect

# 9. TIME ON SITE - More time → higher conversion
time_effect = np.minimum(df['time_on_site_mins'] / 100, 0.10)
conversion_prob += time_effect

# 10. PAGES VIEWED - More pages → higher conversion
pages_effect = np.minimum(df['pages_viewed'] / 80, 0.06)
conversion_prob += pages_effect

# 11. INTERACTION EFFECT: Discount × Occupation
# Discounts work especially well for students and unemployed
high_discount_mask = df['discount_offered'] >= 20
student_mask = df['occupation'].isin(['student', 'unemployed'])
conversion_prob += (high_discount_mask & student_mask) * 0.08

# 12. INTERACTION EFFECT: Engagement × Previous Courses
# High engagement existing customers are golden
high_engagement_mask = df['engagement_score'] > 60
has_previous_mask = df['previous_courses'] > 0
conversion_prob += (high_engagement_mask & has_previous_mask) * 0.10

# Add base rate and clip to valid probability range
conversion_prob += 0.02  # 2% base conversion
conversion_prob = np.clip(conversion_prob, 0.01, 0.95)

# Generate actual conversions
df['converted'] = (np.random.random(n_samples) < conversion_prob).astype(int)

# Reorder columns
cols = ['customer_id', 'converted'] + [col for col in df.columns if col not in ['customer_id', 'converted']]
df = df[cols]

# Save
output_path = '/home/granty1231/Classifier-Model-Interpreter/data/customer_conversion.csv'
df.to_csv(output_path, index=False)

print(f"Generated {n_samples} customers")
print(f"Conversion rate: {df['converted'].mean():.2%}")
print(f"\nSaved to: {output_path}")

# Show some statistics
print("\n" + "="*60)
print("KEY PATTERN VERIFICATION")
print("="*60)

print("\nConversion by Engagement Score:")
df['engagement_bin'] = pd.cut(df['engagement_score'], bins=[0, 30, 50, 70, 100], labels=['Low', 'Medium', 'High', 'Very High'])
print(df.groupby('engagement_bin')['converted'].agg(['mean', 'count']))

print("\nConversion by Discount:")
print(df.groupby('discount_offered')['converted'].agg(['mean', 'count']))

print("\nConversion by Device:")
print(df.groupby('device_type')['converted'].agg(['mean', 'count']))

print("\nConversion by Occupation:")
print(df.groupby('occupation')['converted'].agg(['mean', 'count']))

print("\nConversion by Previous Courses:")
df['has_previous'] = df['previous_courses'] > 0
print(df.groupby('has_previous')['converted'].agg(['mean', 'count']))
