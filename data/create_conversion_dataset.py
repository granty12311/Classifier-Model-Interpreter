"""
Create a synthetic customer conversion dataset for testing model interpretation

Dataset: Online Course Trial-to-Paid Conversion
- Clean, interpretable features
- Clear patterns for SHAP analysis
- >5000 rows for compelling visualizations
- Known interactions for testing interpretation tools
"""
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 5000

print("Generating synthetic customer conversion dataset...")
print(f"Target samples: {n_samples:,}")

# ============================================================================
# FEATURE GENERATION
# ============================================================================

# Demographics
age = np.random.normal(35, 12, n_samples).clip(18, 70).astype(int)
country = np.random.choice(
    ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Other'],
    n_samples,
    p=[0.35, 0.15, 0.10, 0.08, 0.10, 0.07, 0.15]
)
occupation = np.random.choice(
    ['student', 'professional', 'self_employed', 'retired', 'unemployed'],
    n_samples,
    p=[0.25, 0.45, 0.15, 0.10, 0.05]
)

# Engagement metrics (key drivers)
pages_viewed = np.random.poisson(12, n_samples).clip(1, 50)
time_on_site_mins = np.random.gamma(3, 8, n_samples).clip(5, 200).round(1)
videos_watched = np.random.poisson(3, n_samples).clip(0, 15)
email_opens = np.random.poisson(2, n_samples).clip(0, 10)
days_since_signup = np.random.uniform(1, 30, n_samples).round(0).astype(int)

# Marketing features
referral_source = np.random.choice(
    ['organic_search', 'paid_ads', 'social_media', 'email_campaign', 'referral', 'direct'],
    n_samples,
    p=[0.25, 0.20, 0.20, 0.15, 0.12, 0.08]
)
discount_offered = np.random.choice([0, 10, 20, 30], n_samples, p=[0.40, 0.25, 0.25, 0.10])

# Historical behavior
previous_courses = np.random.choice([0, 1, 2, 3], n_samples, p=[0.60, 0.25, 0.10, 0.05])
account_age_days = np.random.exponential(180, n_samples).clip(0, 1000).round(0).astype(int)

# Temporal features
signup_month = np.random.choice(range(1, 13), n_samples)
signup_day_of_week = np.random.choice(range(1, 8), n_samples)  # 1=Mon, 7=Sun

# Device and session
device_type = np.random.choice(['mobile', 'desktop', 'tablet'], n_samples, p=[0.50, 0.40, 0.10])
session_count = np.random.poisson(4, n_samples).clip(1, 20)

# ============================================================================
# TARGET VARIABLE GENERATION (WITH CLEAR PATTERNS)
# ============================================================================

print("\nGenerating target variable with interpretable patterns...")

# Initialize probability
conversion_prob = np.zeros(n_samples)

# Base conversion rate (adjusted for realistic 20-25% overall conversion)
base_rate = -2.5

# Pattern 1: Engagement is the strongest driver (scaled up for logistic)
engagement_score = (
    (pages_viewed / pages_viewed.max()) * 1.8 +
    (time_on_site_mins / time_on_site_mins.max()) * 1.5 +
    (videos_watched / videos_watched.max()) * 1.2 +
    (email_opens / email_opens.max()) * 0.9
)
conversion_prob += engagement_score

# Pattern 2: Discount effect (non-linear)
discount_effect = {0: 0.0, 10: 0.4, 20: 0.8, 30: 1.2}
conversion_prob += np.array([discount_effect[d] for d in discount_offered])

# Pattern 3: Previous courses (loyalty effect)
conversion_prob += previous_courses * 0.5

# Pattern 4: Referral source quality
referral_quality = {
    'referral': 0.7,
    'email_campaign': 0.5,
    'organic_search': 0.3,
    'social_media': 0.2,
    'paid_ads': 0.1,
    'direct': 0.05
}
conversion_prob += np.array([referral_quality[r] for r in referral_source])

# Pattern 5: Age effect (inverted U-shape)
age_normalized = (age - 30) / 40  # Center around 25-35
age_effect = 0.5 - (age_normalized ** 2) * 0.4
conversion_prob += age_effect

# Pattern 6: Occupation effect
occupation_effect = {
    'professional': 0.6,
    'self_employed': 0.4,
    'student': 0.2,
    'retired': 0.1,
    'unemployed': -0.3
}
conversion_prob += np.array([occupation_effect[o] for o in occupation])

# Pattern 7: Device type
device_effect = {'desktop': 0.3, 'mobile': 0.1, 'tablet': -0.1}
conversion_prob += np.array([device_effect[d] for d in device_type])

# Pattern 8: Interaction - discount is more effective for students
student_mask = occupation == 'student'
conversion_prob[student_mask] += (discount_offered[student_mask] / 100) * 0.8

# Pattern 9: Recency - early engagement matters
recency_effect = np.exp(-days_since_signup / 15) * 0.4
conversion_prob += recency_effect

# Apply logistic transformation for realistic probabilities
conversion_prob = 1 / (1 + np.exp(-conversion_prob))

# Adjust to get realistic 20-30% conversion rate
# Normalize to 0-1 range, then scale to target distribution
conversion_prob = (conversion_prob - conversion_prob.min()) / (conversion_prob.max() - conversion_prob.min())
conversion_prob = conversion_prob * 0.42  # Scale to 0-0.42 range for ~25-30% conversion
conversion_prob = conversion_prob.clip(0.01, 0.90)

# Generate binary target
converted = (np.random.random(n_samples) < conversion_prob).astype(int)

# ============================================================================
# CREATE DATAFRAME
# ============================================================================

df = pd.DataFrame({
    # Target variable (position 0)
    'converted': converted,

    # Engagement metrics (strongest predictors)
    'pages_viewed': pages_viewed,
    'time_on_site_mins': time_on_site_mins,
    'videos_watched': videos_watched,
    'email_opens': email_opens,
    'session_count': session_count,

    # Marketing features
    'discount_offered': discount_offered,
    'referral_source': referral_source,

    # Demographics
    'age': age,
    'country': country,
    'occupation': occupation,
    'device_type': device_type,

    # Historical behavior
    'previous_courses': previous_courses,
    'account_age_days': account_age_days,
    'days_since_signup': days_since_signup,

    # Temporal
    'signup_month': signup_month,
    'signup_day_of_week': signup_day_of_week,
})

# ============================================================================
# ADD SOME REALISTIC NOISE AND MISSING VALUES
# ============================================================================

print("\nAdding realistic noise and missing values...")

# Add missing values to some columns (realistic scenarios)
missing_mask_email = np.random.random(n_samples) < 0.05
df.loc[missing_mask_email, 'email_opens'] = np.nan

missing_mask_previous = np.random.random(n_samples) < 0.03
df.loc[missing_mask_previous, 'previous_courses'] = np.nan

# Add a customer ID
df.insert(0, 'customer_id', range(1, n_samples + 1))

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)

print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nConversion rate: {df['converted'].mean()*100:.2f}%")
print(f"  Converted:     {df['converted'].sum():,} ({df['converted'].mean()*100:.1f}%)")
print(f"  Not converted: {(1-df['converted']).sum():,} ({(1-df['converted']).mean()*100:.1f}%)")

print("\nFeature Summary:")
print(f"  Numeric features:     {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"  Categorical features: {df.select_dtypes(include=['object']).shape[1]}")

print("\nMissing values:")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    for col, count in missing.items():
        print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
else:
    print("  None")

print("\nKey Feature Distributions:")
print("\nEngagement Metrics:")
print(f"  pages_viewed:      mean={df['pages_viewed'].mean():.1f}, median={df['pages_viewed'].median():.0f}")
print(f"  time_on_site_mins: mean={df['time_on_site_mins'].mean():.1f}, median={df['time_on_site_mins'].median():.1f}")
print(f"  videos_watched:    mean={df['videos_watched'].mean():.1f}, median={df['videos_watched'].median():.0f}")
print(f"  email_opens:       mean={df['email_opens'].mean():.1f}, median={df['email_opens'].median():.0f}")

print("\nMarketing:")
print(df['referral_source'].value_counts())
print(f"\nDiscount distribution:")
print(df['discount_offered'].value_counts().sort_index())

print("\nDemographics:")
print(f"  Age: mean={df['age'].mean():.1f}, median={df['age'].median():.0f}")
print(f"\nOccupation:")
print(df['occupation'].value_counts())

print("\nConversion rates by key segments:")
print(f"\nBy discount:")
for discount in sorted(df['discount_offered'].unique()):
    conv_rate = df[df['discount_offered']==discount]['converted'].mean()*100
    count = (df['discount_offered']==discount).sum()
    print(f"  {discount}%: {conv_rate:.1f}% ({count:,} customers)")

print(f"\nBy occupation:")
for occ in df['occupation'].value_counts().index:
    conv_rate = df[df['occupation']==occ]['converted'].mean()*100
    count = (df['occupation']==occ).sum()
    print(f"  {occ:15s}: {conv_rate:.1f}% ({count:,} customers)")

print(f"\nBy referral source:")
for source in df['referral_source'].value_counts().index:
    conv_rate = df[df['referral_source']==source]['converted'].mean()*100
    count = (df['referral_source']==source).sum()
    print(f"  {source:15s}: {conv_rate:.1f}% ({count:,} customers)")

# ============================================================================
# SAVE DATASET
# ============================================================================

output_file = 'customer_conversion.csv'
print(f"\n{'='*80}")
print(f"Saving to {output_file}...")
df.to_csv(output_file, index=False)

# Create documentation
doc_file = 'customer_conversion_README.md'
print(f"Creating documentation: {doc_file}...")

with open(doc_file, 'w') as f:
    f.write("# Customer Conversion Dataset\n\n")
    f.write("## Overview\n\n")
    f.write("Synthetic dataset simulating **online course trial-to-paid conversion** for testing model interpretation tools.\n\n")
    f.write(f"- **Rows**: {df.shape[0]:,}\n")
    f.write(f"- **Columns**: {df.shape[1]}\n")
    f.write(f"- **Conversion Rate**: {df['converted'].mean()*100:.2f}%\n")
    f.write(f"- **Use Case**: Testing SHAP values and model interpretation\n\n")

    f.write("## Dataset Description\n\n")
    f.write("This dataset represents customers who signed up for a free trial of an online course platform. ")
    f.write("The goal is to predict whether they convert to a paid subscription (`converted=1`) or not (`converted=0`).\n\n")

    f.write("## Features\n\n")
    f.write("### Target Variable\n")
    f.write("- **converted** (0/1): Whether customer converted to paid (1=yes, 0=no)\n\n")

    f.write("### Engagement Metrics (Strongest Predictors)\n")
    f.write("- **pages_viewed**: Number of course pages viewed during trial\n")
    f.write("- **time_on_site_mins**: Total time spent on platform (minutes)\n")
    f.write("- **videos_watched**: Number of course videos watched\n")
    f.write("- **email_opens**: Number of marketing emails opened\n")
    f.write("- **session_count**: Number of login sessions during trial\n\n")

    f.write("### Marketing Features\n")
    f.write("- **discount_offered**: Discount percentage offered (0%, 10%, 20%, 30%)\n")
    f.write("- **referral_source**: How customer found the platform\n")
    f.write("  - organic_search, paid_ads, social_media, email_campaign, referral, direct\n\n")

    f.write("### Demographics\n")
    f.write("- **age**: Customer age (18-70)\n")
    f.write("- **country**: Customer country (USA, UK, Canada, Australia, Germany, France, Other)\n")
    f.write("- **occupation**: Employment status (student, professional, self_employed, retired, unemployed)\n")
    f.write("- **device_type**: Primary device used (mobile, desktop, tablet)\n\n")

    f.write("### Historical Behavior\n")
    f.write("- **previous_courses**: Number of courses completed on platform before (0-3)\n")
    f.write("- **account_age_days**: Days since account creation\n")
    f.write("- **days_since_signup**: Days since trial signup\n\n")

    f.write("### Temporal\n")
    f.write("- **signup_month**: Month of trial signup (1-12)\n")
    f.write("- **signup_day_of_week**: Day of week (1=Monday, 7=Sunday)\n\n")

    f.write("## Known Patterns (for Testing Interpretation)\n\n")
    f.write("The dataset was generated with the following interpretable patterns:\n\n")
    f.write("1. **Engagement is the strongest driver**: More pages viewed, videos watched → higher conversion\n")
    f.write("2. **Discount effect**: Larger discounts → higher conversion (non-linear)\n")
    f.write("3. **Loyalty effect**: Previous course completions → higher conversion\n")
    f.write("4. **Referral quality**: Referrals and email campaigns → highest conversion\n")
    f.write("5. **Age effect**: Inverted U-shape (25-35 age group converts best)\n")
    f.write("6. **Occupation effect**: Professionals convert best, unemployed least\n")
    f.write("7. **Device effect**: Desktop users convert slightly better\n")
    f.write("8. **Interaction effect**: Discounts are MORE effective for students\n")
    f.write("9. **Recency effect**: Early engagement (low days_since_signup) → better conversion\n\n")

    f.write("## Usage\n\n")
    f.write("```python\n")
    f.write("import pandas as pd\n\n")
    f.write("# Load data\n")
    f.write("df = pd.read_csv('customer_conversion.csv')\n\n")
    f.write("# Separate features and target\n")
    f.write("X = df.drop(['customer_id', 'converted'], axis=1)\n")
    f.write("y = df['converted']\n\n")
    f.write("# Check conversion rate\n")
    f.write("print(f'Conversion rate: {y.mean()*100:.2f}%')\n")
    f.write("```\n\n")

    f.write("## Column Reference\n\n")
    f.write("| Column | Type | Description | Missing |\n")
    f.write("|--------|------|-------------|----------|\n")

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = df[col].isnull().sum() / len(df) * 100

        # Get description
        descriptions = {
            'customer_id': 'Unique customer identifier',
            'converted': 'Target: 1=converted, 0=not converted',
            'pages_viewed': 'Number of pages viewed',
            'time_on_site_mins': 'Time on site (minutes)',
            'videos_watched': 'Number of videos watched',
            'email_opens': 'Number of emails opened',
            'session_count': 'Number of sessions',
            'discount_offered': 'Discount % (0, 10, 20, 30)',
            'referral_source': 'Acquisition channel',
            'age': 'Customer age',
            'country': 'Customer country',
            'occupation': 'Employment status',
            'device_type': 'Primary device',
            'previous_courses': 'Prior course completions',
            'account_age_days': 'Account age in days',
            'days_since_signup': 'Days since trial start',
            'signup_month': 'Signup month (1-12)',
            'signup_day_of_week': 'Day of week (1=Mon)',
        }

        desc = descriptions.get(col, '')
        missing_str = f"{missing_pct:.1f}%" if missing_pct > 0 else "0%"

        f.write(f"| {col} | {dtype} | {desc} | {missing_str} |\n")

    f.write("\n## Notes\n\n")
    f.write("- This is a synthetic dataset created specifically for testing model interpretation tools\n")
    f.write("- Patterns are intentionally clear and interpretable\n")
    f.write("- Small amount of missing data (~3-5%) in email_opens and previous_courses\n")
    f.write("- Class balance is realistic (~20% conversion rate)\n")
    f.write("- Features include numeric, categorical, and ordinal types\n")

print("\n✅ Dataset creation complete!")
print(f"\nOutput files:")
print(f"  - {output_file} ({df.shape[0]:,} rows × {df.shape[1]} columns)")
print(f"  - {doc_file}")

print("\nSample data:")
print(df.head(10))

print("\nDataset is ready for model interpretation testing!")
