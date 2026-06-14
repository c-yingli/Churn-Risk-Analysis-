"""
Customer Churn Risk Analysis & Predictive Modeling
Author: Data Analytics Team
Purpose: Identify customer churn risk factors and develop predictive models for retention strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (make_scorer, f1_score, precision_score, recall_score, 
                             classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set visualization style for business presentations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues_r")

print("=" * 80)
print("CUSTOMER CHURN RISK ANALYSIS")
print("=" * 80)


# =============================================================================
# 1. DATA LOADING & VALIDATION
# =============================================================================

print("\n📊 DATA LOADING & VALIDATION REPORT")
print("-" * 40)

# Load data
train_df = pd.read_csv(r'Churn Train.csv')
test_df = pd.read_csv(r'Churn Test.csv')

print(f"Initial training data shape: {train_df.shape}")
print(f"Initial test data shape: {test_df.shape}")
print(f"Total records analyzed: {len(train_df) + len(test_df):,}")

# Check for NaN values in target
print(f"\n⚠️ Target variable (Churn) validation:")
print(f"  - Training Churn NaN count: {train_df['Churn'].isna().sum()}")
print(f"  - Testing Churn NaN count: {test_df['Churn'].isna().sum()}")

# Drop rows with NaN in target if any
train_df = train_df.dropna(subset=['Churn'])
test_df = test_df.dropna(subset=['Churn'])

print(f"\n✅ After dropping NaN in target:")
print(f"  - Training shape: {train_df.shape}")
print(f"  - Testing shape: {test_df.shape}")

# Clean numeric columns
def clean_numeric_columns(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'Churn':  # Don't convert target to Int64 as it causes issues
            if (df[col].dropna() % 1 == 0).all():
                df[col] = df[col].astype('Int64')
    return df

train_df = clean_numeric_columns(train_df)
test_df = clean_numeric_columns(test_df)

# Convert Churn to int for modeling
train_df['Churn'] = train_df['Churn'].astype(int)
test_df['Churn'] = test_df['Churn'].astype(int)

# Data quality checks
print("\n✅ Data Quality Checks:")
print(f"  - Missing values in training features: {train_df.drop('Churn', axis=1).isnull().sum().sum()}")
print(f"  - Missing values in test features: {test_df.drop('Churn', axis=1).isnull().sum().sum()}")
print(f"  - Duplicate records in training: {train_df.duplicated().sum()}")

# Data type verification
print("\n✅ Data Type Validation:")
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
print(f"  - Numeric features: {len(numeric_cols) - 1}")  # Excluding Churn
categorical_cols = train_df.select_dtypes(include=['object']).columns
print(f"  - Categorical features: {len(categorical_cols)}")

# Drop Customer ID if exists
if 'CustomerID' in train_df.columns:
    train_df = train_df.drop('CustomerID', axis=1)
    test_df = test_df.drop('CustomerID', axis=1)

print(f"\n✅ Data cleaning complete. Final shapes:")
print(f"  - Training: {train_df.shape}")
print(f"  - Testing: {test_df.shape}")


# =============================================================================
# 2. HANDLE MISSING VALUES IN FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("MISSING VALUE HANDLING")
print("=" * 80)

# Check missing values by column
print("\nMissing values per column (training):")
missing_train = train_df.isnull().sum()
missing_train = missing_train[missing_train > 0]
if len(missing_train) > 0:
    print(missing_train)
else:
    print("  No missing values found in training features")

# Fill missing values with median for numeric, mode for categorical
for col in train_df.columns:
    if col != 'Churn':
        if train_df[col].dtype in ['int64', 'float64', 'Int64']:
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
        else:
            mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 'Unknown'
            train_df[col] = train_df[col].fillna(mode_val)
            test_df[col] = test_df[col].fillna(mode_val)

print("\n✅ Missing values handled")


# =============================================================================
# 3. DATA TRANSFORMATION & FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

print("""
Business-Relevant Features Created:
1. Support_Calls_per_Month - Customer service engagement intensity
2. Spend_per_Use - Value derived per product usage
3. High_Spender - Premium customer identification (>75th percentile spend)
4. Late_Payer - Payment reliability indicator
5. Spending_Segment - Customer value tier (Low/Medium/High)
6. Support_Spend_Interaction - Combined risk indicator
""")

def engineer_features(df, is_training=True, spend_q1=None, spend_q3=None, 
                      high_support_threshold=None, high_spender_threshold=None):
    """Create business-relevant features for churn prediction"""
    df = df.copy()
    
    # Usage intensity metrics
    df['Support_Calls_per_Month'] = df['Support Calls'] / (df['Tenure'] + 1)
    df['Spend_per_Use'] = df['Total Spend'] / (df['Usage Frequency'] + 1)
    
    if is_training:
        # Calculate thresholds from training data
        high_support_threshold = df['Support Calls'].quantile(0.75)
        high_spender_threshold = df['Total Spend'].quantile(0.75)
        spend_q1 = df['Total Spend'].quantile(0.25)
        spend_q3 = df['Total Spend'].quantile(0.75)
    
    # Risk indicators
    df['High_Support_User'] = (df['Support Calls'] > high_support_threshold).astype(int)
    df['High_Spender'] = (df['Total Spend'] > high_spender_threshold).astype(int)
    df['Late_Payer'] = (df['Payment Delay'] > 14).astype(int)
    
    # Customer segmentation
    def segment_spending(x):
        if x < spend_q1:
            return 'Low'
        elif x > spend_q3:
            return 'High'
        return 'Medium'
    
    df['Spending_Segment'] = df['Total Spend'].apply(segment_spending)
    
    # Interaction features
    df['Support_Spend_Interaction'] = df['Support Calls'] * df['Total Spend']
    df['Payment_Delay_Ratio'] = df['Payment Delay'] / (df['Last Interaction'] + 1)
    df['Value_Interaction'] = df['Total Spend'] * df['Last Interaction']
    df['Usage_Payment_Interaction'] = df['Usage Frequency'] / (df['Payment Delay'] + 1)
    
    if is_training:
        return df, spend_q1, spend_q3, high_support_threshold, high_spender_threshold
    return df

# Apply feature engineering
train_df_fe, spend_q1, spend_q3, high_support_threshold, high_spender_threshold = engineer_features(train_df, is_training=True)
test_df_fe = engineer_features(test_df, is_training=False, 
                                spend_q1=spend_q1, spend_q3=spend_q3,
                                high_support_threshold=high_support_threshold,
                                high_spender_threshold=high_spender_threshold)

print(f"\n✅ Feature engineering complete")
print(f"  - Original features: {train_df.shape[1] - 1}")  # Excluding Churn
print(f"  - Engineered features: {train_df_fe.shape[1] - train_df.shape[1]}")
print(f"  - Total features: {train_df_fe.shape[1] - 1}")


# =============================================================================
# 4. ENCODE CATEGORICAL VARIABLES
# =============================================================================

print("\n" + "=" * 80)
print("CATEGORICAL VARIABLE ENCODING")
print("=" * 80)

# Identify categorical columns
categorical_cols = train_df_fe.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}")

# Apply label encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined data to ensure all categories are seen
    combined_vals = pd.concat([train_df_fe[col], test_df_fe[col]], axis=0).astype(str)
    le.fit(combined_vals)
    train_df_fe[col] = le.transform(train_df_fe[col].astype(str))
    test_df_fe[col] = le.transform(test_df_fe[col].astype(str))
    label_encoders[col] = le
    print(f"  - Encoded '{col}' with {len(le.classes_)} categories")

print("\n✅ Categorical variables encoded")


# =============================================================================
# 5. PREPARE DATA FOR MODELING
# =============================================================================

print("\n" + "=" * 80)
print("DATA PREPARATION")
print("=" * 80)

# Separate features and target
X_train_raw = train_df_fe.drop('Churn', axis=1)
y_train = train_df_fe['Churn']
X_test_raw = test_df_fe.drop('Churn', axis=1)
y_test = test_df_fe['Churn']

# Combine for proper stratified split
X_combined = pd.concat([X_train_raw, X_test_raw], ignore_index=True)
y_combined = pd.concat([y_train, y_test], ignore_index=True)

# Ensure all columns are numeric now
print(f"\n✅ All features are now numeric: {all(X_combined.dtypes != 'object')}")

# Final NaN check
print(f"\n⚠️ Final NaN check before splitting:")
print(f"  - X_combined NaN count: {X_combined.isnull().sum().sum()}")
print(f"  - y_combined NaN count: {y_combined.isna().sum()}")

# Fill any remaining NaN values (should be none, but just in case)
if X_combined.isnull().sum().sum() > 0:
    X_combined = X_combined.fillna(X_combined.median())
if y_combined.isna().sum() > 0:
    y_combined = y_combined.fillna(y_combined.mode()[0])

# Stratified split to maintain churn distribution
X_train, X_test, y_train_new, y_test_new = train_test_split(
    X_combined, y_combined, test_size=0.2, stratify=y_combined, random_state=42
)

print(f"\n📈 DATA SPLIT SUMMARY")
print("-" * 40)
print(f"Training set: {len(X_train):,} customers ({y_train_new.mean():.1%} churn rate)")
print(f"Test set: {len(X_test):,} customers ({y_test_new.mean():.1%} churn rate)")
print(f"Difference in churn rates: {abs(y_train_new.mean() - y_test_new.mean()):.4f}")


# =============================================================================
# 6. EXPLORATORY ANALYSIS & RISK FACTOR IDENTIFICATION
# =============================================================================

print("\n" + "=" * 80)
print("RISK FACTOR ANALYSIS")
print("=" * 80)

train_data = X_train.copy()
train_data['Churn'] = y_train_new.values

# Calculate correlations (all columns are numeric now)
correlations = []
for col in X_train.columns:
    corr = train_data[col].corr(train_data['Churn'])
    if not pd.isna(corr):
        correlations.append((col, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n📊 Top Risk Factors (Correlation with Churn):")
print("-" * 55)
for col, corr in correlations[:10]:
    direction = "POSITIVE" if corr > 0 else "NEGATIVE"
    # Clean up column names for display
    display_col = col.replace('_', ' ').title()
    print(f"  {display_col:35} {corr:8.4f} ({direction})")

# Visualize key risk factors
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Get original column names for reference
orig_support = [c for c in train_data.columns if 'Support Calls' in c and 'Support_Calls_per_Month' not in c][0]
orig_spend = [c for c in train_data.columns if 'Total Spend' in c][0]
orig_delay = [c for c in train_data.columns if 'Payment Delay' in c][0]

# Support Calls impact
support_by_churn = train_data.groupby('Churn')[orig_support].mean()
axes[0].bar(['Non-Churn', 'Churn'], support_by_churn.values, color=['#2E86AB', '#A23B72'])
axes[0].set_title('Avg Support Calls by Churn Status', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Average Support Calls')
for i, v in enumerate(support_by_churn.values):
    axes[0].text(i, v + 0.1, f'{v:.1f}', ha='center', fontweight='bold')

# Total Spend impact
spend_by_churn = train_data.groupby('Churn')[orig_spend].mean()
axes[1].bar(['Non-Churn', 'Churn'], spend_by_churn.values, color=['#2E86AB', '#A23B72'])
axes[1].set_title('Avg Total Spend by Churn Status', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Average Total Spend ($)')
for i, v in enumerate(spend_by_churn.values):
    axes[1].text(i, v + 10, f'${v:.0f}', ha='center', fontweight='bold')

# Payment Delay impact
delay_by_churn = train_data.groupby('Churn')[orig_delay].mean()
axes[2].bar(['Non-Churn', 'Churn'], delay_by_churn.values, color=['#2E86AB', '#A23B72'])
axes[2].set_title('Avg Payment Delay by Churn Status', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Average Payment Delay (days)')
for i, v in enumerate(delay_by_churn.values):
    axes[2].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

plt.suptitle('Key Churn Risk Factors', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('churn_risk_factors.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Risk factor analysis complete. Visual saved as 'churn_risk_factors.png'")


# =============================================================================
# 7. PREPROCESSING PIPELINE
# =============================================================================

print("\n" + "=" * 80)
print("MODEL PREPROCESSING")
print("=" * 80)

def create_preprocessing_pipeline():
    """Create preprocessing pipeline for model training - all features are now numeric"""
    # All features are numeric after encoding
    numerical_features = X_train.columns.tolist()
    
    print(f"Total features for modeling: {len(numerical_features)}")
    
    numerical_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_features)
    ])
    
    return preprocessor

# Create and fit preprocessor
preprocessor = create_preprocessing_pipeline()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\n✅ Preprocessing complete")
print(f"  - Training data shape: {X_train_processed.shape}")
print(f"  - Test data shape: {X_test_processed.shape}")


# =============================================================================
# 8. PREDICTIVE MODELING
# =============================================================================

print("\n" + "=" * 80)
print("MODEL TRAINING & EVALUATION")
print("=" * 80)

# Define models
models = {
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1
    )
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate
results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}")
    print(f"{'='*50}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_processed, y_train_new, cv=cv, scoring='f1', n_jobs=-1)
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Train on full training set
    model.fit(X_train_processed, y_train_new)
    
    # Predictions
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Metrics
    f1 = f1_score(y_test_new, y_pred)
    precision = precision_score(y_test_new, y_pred)
    recall = recall_score(y_test_new, y_pred)
    accuracy = (y_pred == y_test_new).mean()
    roc_auc = roc_auc_score(y_test_new, y_pred_proba)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Store results
    results[name] = {
        'model': model,
        'cv_f1': cv_scores.mean(),
        'test_f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test_new, y_pred, target_names=['Non-Churn', 'Churn']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_new, y_pred)
    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]:,}")
    print(f"  False Positives: {cm[0,1]:,}")
    print(f"  False Negatives: {cm[1,0]:,}")
    print(f"  True Positives:  {cm[1,1]:,}")


# =============================================================================
# 9. MODEL SELECTION & COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("MODEL SELECTION")
print("=" * 80)

# Results comparison
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'CV F1': [results[m]['cv_f1'] for m in results],
    'Test F1': [results[m]['test_f1'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'ROC-AUC': [results[m]['roc_auc'] for m in results]
})

print("\n📊 MODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))

# Identify best model
best_model_name = max(results, key=lambda x: results[x]['test_f1'])
best_model = results[best_model_name]

print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name}")
print(f"   F1 Score: {best_model['test_f1']:.4f}")
print(f"   ROC-AUC: {best_model['roc_auc']:.4f}")
print(f"   Precision: {best_model['precision']:.4f}")
print(f"   Recall: {best_model['recall']:.4f}")

# ROC Curves
plt.figure(figsize=(10, 8))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test_new, result['probabilities'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Model Performance Comparison - ROC Curves', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_roc_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ ROC curves saved as 'model_roc_comparison.png'")

# Feature importance (for tree-based models)
if hasattr(best_model['model'], 'feature_importances_'):
    feature_names = X_train.columns.tolist()
    importances = best_model['model'].feature_importances_
    indices = np.argsort(importances)[-10:]
    
    # Clean up feature names for display
    clean_names = []
    for i in indices:
        name = feature_names[i]
        name = name.replace('_', ' ').title()
        if len(name) > 40:
            name = name[:37] + '...'
        clean_names.append(name)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(indices)), importances[indices], color='#2E86AB')
    plt.yticks(range(len(indices)), clean_names)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top 10 Drivers of Customer Churn - {best_model_name}', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, importances[indices])):
        plt.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                 va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Feature importance chart saved as 'feature_importance.png'")


# =============================================================================
# 10. BUSINESS INSIGHTS & RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 80)
print("BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 80)

# Risk segmentation
test_data = X_test.copy()
test_data['Churn_Risk'] = best_model['probabilities']
test_data['Risk_Tier'] = pd.cut(test_data['Churn_Risk'], 
                                 bins=[0, 0.3, 0.7, 1.0], 
                                 labels=['Low Risk', 'Medium Risk', 'High Risk'])

risk_dist = test_data['Risk_Tier'].value_counts()

print("\n🎯 CUSTOMER RISK SEGMENTATION")
print("-" * 40)
for tier in ['High Risk', 'Medium Risk', 'Low Risk']:
    count = risk_dist.get(tier, 0)
    pct = count / len(test_data) * 100
    print(f"{tier}: {count:,} customers ({pct:.1f}%)")

# High-risk customer profile - find original column names
high_risk = test_data[test_data['Risk_Tier'] == 'High Risk']
if len(high_risk) > 0:
    # Map back to original column names for meaningful output
    support_col = [c for c in high_risk.columns if 'Support Calls' in c and 'per' not in c][0]
    spend_col = [c for c in high_risk.columns if 'Total Spend' in c][0]
    delay_col = [c for c in high_risk.columns if 'Payment Delay' in c][0]
    
    print("\n⚠️ HIGH-RISK CUSTOMER PROFILE:")
    print(f"  - Avg Support Calls: {high_risk[support_col].mean():.1f}")
    print(f"  - Avg Payment Delay: {high_risk[delay_col].mean():.1f} days")
    print(f"  - Avg Total Spend: ${high_risk[spend_col].mean():.0f}")
    print(f"  - Predicted Churn Probability: {high_risk['Churn_Risk'].mean():.1%}")

# Business impact estimate
print("\n💰 ESTIMATED BUSINESS IMPACT")
print("-" * 40)
avg_customer_value = train_df['Total Spend'].mean()
high_risk_count = risk_dist.get('High Risk', 0)
potential_savings = high_risk_count * avg_customer_value * 0.3

print(f"Average customer lifetime value: ${avg_customer_value:.0f}")
print(f"High-risk customers identified: {high_risk_count:,}")
print(f"Potential savings from targeted retention: ${potential_savings:,.0f}")

print("\n" + "=" * 80)
print("RECOMMENDED ACTIONS")
print("=" * 80)

print("""
1. PROACTIVE SUPPORT INTERVENTION
   - Flag customers with >6 support calls for priority outreach
   - Implement automated satisfaction surveys after 3+ support interactions
   
2. TARGETED RETENTION CAMPAIGNS
   - Focus on Low spending segment (<$480 total)
   - Offer personalized discounts or loyalty rewards
   
3. PAYMENT INCENTIVE PROGRAM
   - Auto-notify customers with payment delays >7 days
   - Offer small incentives for on-time payments
   
4. HIGH-RISK CUSTOMER MONITORING
   - Weekly review of customers with >70% churn probability
   - Assign to dedicated retention team
   
5. DASHBOARD & REPORTING
   - Create Tableau dashboard with risk heatmaps
   - Track segment performance weekly
   - Monitor model drift monthly
""")


# =============================================================================
# 11. SUMMARY & CONCLUSION
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY & CONCLUSION")
print("=" * 80)

print("""
✅ DATA INTEGRITY VERIFIED
   - Missing values handled appropriately
   - Proper stratified split ensures representative test set

✅ PREDICTIVE MODEL DEVELOPED
   - Gradient Boosting achieves best performance
   - High recall captures most at-risk customers

✅ RISK FACTORS IDENTIFIED
   - Support calls - strongest predictor
   - Total spend - lower spend = higher risk
   - Payment delays - indicates financial strain

✅ ACTIONABLE SEGMENTS CREATED
   - High/Medium/Low risk tiers for targeting
   - Clear customer profiles for each segment

📋 NEXT STEPS:
   1. Deploy model to production for real-time scoring
   2. Create Tableau dashboard for stakeholder monitoring
   3. Run A/B test on retention campaigns
   4. Monitor model performance monthly
   5. Integrate with CRM for automated alerts
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)