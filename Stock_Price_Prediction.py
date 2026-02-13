import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STOCK PRICE PREDICTION - NEXT DAY PRICE FORECASTING
# ============================================================================

print("="*70)
print("APPLE (AAPL) STOCK PRICE PREDICTION MODEL")
print("Next-Day Price Forecasting using Gradient Boosting")
print("="*70)

# Step 1: Load Data
print("\n[1] Loading AAPL stock data...")
df = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Stock prediction\aapl_stock_data.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df = df.sort_values('Date').reset_index(drop=True)

# Use last 2 years of data
df = df.tail(500).reset_index(drop=True)
print(f"    Data loaded: {len(df)} records from {df['Date'].min().date()} to {df['Date'].max().date()}")

# ============================================================================
# Feature Engineering
# ============================================================================
print("\n[2] Engineering features...")

# Create features based on historical prices
df['Close_Lag1'] = df['Close'].shift(1)      # Previous day close
df['Close_Lag2'] = df['Close'].shift(2)      # 2 days ago
df['Close_Lag3'] = df['Close'].shift(3)      # 3 days ago
df['Close_Lag5'] = df['Close'].shift(5)      # 5 days ago

# Moving averages
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# Price momentum
df['Momentum_5'] = df['Close'] - df['Close'].shift(5)

# Volatility
df['Returns'] = df['Close'].pct_change()
df['Volatility_5'] = df['Returns'].rolling(window=5).std()

# Volume indicators
df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

# High-Low range
df['HL_Range'] = (df['High'] - df['Low']) / df['Close']

# Target: Next day's closing price
df['Target'] = df['Close'].shift(-1)

# Drop NaN values
df = df.dropna().reset_index(drop=True)

print(f"    Features created: Lag features, SMA, Momentum, Volatility, Volume, Range")
print(f"    Total features: 10")
print(f"    Data after feature engineering: {len(df)} records")

# ============================================================================
# Prepare Features and Target
# ============================================================================
print("\n[3] Preparing training and test data...")

feature_columns = ['Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Close_Lag5', 
                   'SMA_5', 'SMA_10', 'SMA_20', 'Momentum_5', 'Volatility_5', 'Volume_Ratio']

X = df[feature_columns].values
y = df['Target'].values

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (temporal split - 80/20)
split_point = int(len(X) * 0.80)
X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"    Training set: {len(X_train)} samples (80%)")
print(f"    Test set: {len(X_test)} samples (20%)")
print(f"    Train period: {df['Date'].iloc[0].date()} to {df['Date'].iloc[split_point-1].date()}")
print(f"    Test period:  {df['Date'].iloc[split_point].date()} to {df['Date'].iloc[-1].date()}")

# ============================================================================
# Train Gradient Boosting Model
# ============================================================================
print("\n[4] Training Gradient Boosting model...")

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    min_samples_split=4,
    min_samples_leaf=2,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

model.fit(X_train, y_train)
print("    ✓ Model trained successfully!")
print(f"    Hyperparameters: n_estimators=200, max_depth=7, learning_rate=0.05")

# ============================================================================
# Make Predictions
# ============================================================================
print("\n[5] Making next-day price predictions...")

y_pred = model.predict(X_test)
print(f"    Predictions generated for {len(y_pred)} test samples")

# ============================================================================
# Calculate Metrics
# ============================================================================
print("\n[6] Calculating performance metrics...")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)
print(f"  RMSE (Root Mean Squared Error):  ${rmse:.2f}")
print(f"  MAE (Mean Absolute Error):       ${mae:.2f}")
print(f"  R² Score:                        {r2:.4f}")
print("="*70)

# Detailed interpretation
avg_price = y_test.mean()
error_percentage = (rmse / avg_price) * 100

print(f"\n  DETAILED INTERPRETATION:")
print(f"  ├─ Average test price: ${avg_price:.2f}")
print(f"  ├─ RMSE: ${rmse:.2f} (~{error_percentage:.2f}% of average price)")
print(f"  ├─ MAE: ${mae:.2f}")
print(f"  ├─ Model explains: {r2*100:.2f}% of price movements")

if r2 > 0.70:
    status = "✓ EXCELLENT - Model captures majority of price patterns"
elif r2 > 0.50:
    status = "✓ GOOD - Model has strong predictive power"
elif r2 > 0.30:
    status = "⚠ MODERATE - Model captures some patterns"
elif r2 > 0.0:
    status = "⚠ WEAK - Model explains minimal variance"
else:
    status = "✗ POOR - Model performs below baseline"

print(f"  └─ Status: {status}")

# Directional accuracy
direction_actual = np.sign(np.diff(y_test))
direction_pred = np.sign(np.diff(y_pred))
directional_accuracy = np.mean(direction_actual == direction_pred) * 100

print(f"\n  DIRECTIONAL ACCURACY: {directional_accuracy:.1f}%")
print(f"    (How often model predicts correct price direction)")

print("="*70)

# ============================================================================
# Visualization 1: Main Predictions
# ============================================================================
print("\n[7] Creating visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Actual vs Predicted (Time Series)
ax1 = fig.add_subplot(gs[0, :])
test_indices = range(len(y_test))
ax1.plot(test_indices, y_test, label='Actual Next-Day Price', 
         color='blue', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
ax1.plot(test_indices, y_pred, label='Predicted Next-Day Price', 
         color='red', linewidth=2.5, marker='x', markersize=5, alpha=0.8)
ax1.fill_between(test_indices, y_test, y_pred, alpha=0.15, color='gray')
ax1.set_xlabel('Days in Test Set', fontsize=11, fontweight='bold')
ax1.set_ylabel('Stock Price ($)', fontsize=11, fontweight='bold')
ax1.set_title('AAPL: Actual vs Predicted Next-Day Prices', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter Plot
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(y_test, y_pred, alpha=0.6, s=60, color='purple', edgecolors='black', linewidth=0.5)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')
ax2.set_xlabel('Actual Price ($)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted Price ($)', fontsize=11, fontweight='bold')
ax2.set_title('Prediction Accuracy: Actual vs Predicted', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Error Distribution
ax3 = fig.add_subplot(gs[1, 1])
errors = y_test - y_pred
ax3.hist(errors, bins=30, color='green', alpha=0.7, edgecolor='black', linewidth=1)
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero Error')
ax3.axvline(x=errors.mean(), color='orange', linestyle='--', linewidth=2.5, 
           label=f'Mean: ${errors.mean():.2f}')
ax3.set_xlabel('Prediction Error ($)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Residuals over time
ax4 = fig.add_subplot(gs[2, 0])
ax4.scatter(test_indices, errors, alpha=0.6, s=60, color='darkred', edgecolors='black', linewidth=0.5)
ax4.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax4.fill_between(test_indices, errors, 0, alpha=0.2, color='red')
ax4.set_xlabel('Days in Test Set', fontsize=11, fontweight='bold')
ax4.set_ylabel('Residual Error ($)', fontsize=11, fontweight='bold')
ax4.set_title('Prediction Residuals Over Time', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Cumulative Error
ax5 = fig.add_subplot(gs[2, 1])
cumulative_error = np.cumsum(np.abs(errors))
ax5.plot(test_indices, cumulative_error, color='darkblue', linewidth=2.5, marker='o', markersize=3)
ax5.fill_between(test_indices, cumulative_error, alpha=0.3, color='blue')
ax5.set_xlabel('Days in Test Set', fontsize=11, fontweight='bold')
ax5.set_ylabel('Cumulative Absolute Error ($)', fontsize=11, fontweight='bold')
ax5.set_title('Cumulative Error Growth', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

plt.savefig('stock_prediction_results.png', dpi=300, bbox_inches='tight')
print("    ✓ Saved: stock_prediction_results.png")
plt.close()

# ============================================================================
# Feature Importance
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
sorted_features = [feature_columns[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_columns)))
bars = ax.barh(range(len(sorted_features)), sorted_importances, color=colors, edgecolor='black', linewidth=1.5)

ax.set_yticks(range(len(sorted_features)))
ax.set_yticklabels(sorted_features, fontsize=11, fontweight='bold')
ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Gradient Boosting Model - Feature Importance Ranking', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
    ax.text(val + 0.003, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("    ✓ Saved: feature_importance.png")
plt.close()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n[8] Additional Statistics...\n")

print("  TEST SET PRICE STATISTICS:")
print(f"    Actual Price   - Min: ${y_test.min():.2f}, Max: ${y_test.max():.2f}, Mean: ${y_test.mean():.2f}")
print(f"    Predicted Price- Min: ${y_pred.min():.2f}, Max: ${y_pred.max():.2f}, Mean: ${y_pred.mean():.2f}")

print(f"\n  PREDICTION ERROR STATISTICS:")
print(f"    Mean Error (Bias):   ${errors.mean():.2f}")
print(f"    Std Error:           ${errors.std():.2f}")
print(f"    Min Error:           ${errors.min():.2f}")
print(f"    Max Error:           ${errors.max():.2f}")
print(f"    Median Error:        ${np.median(errors):.2f}")

print(f"\n  PREDICTION ACCURACY BREAKDOWN:")
print(f"    Within $1:   {(np.abs(errors) <= 1).sum():3d}/{len(errors)} ({(np.abs(errors) <= 1).sum()/len(errors)*100:5.1f}%)")
print(f"    Within $2:   {(np.abs(errors) <= 2).sum():3d}/{len(errors)} ({(np.abs(errors) <= 2).sum()/len(errors)*100:5.1f}%)")
print(f"    Within $5:   {(np.abs(errors) <= 5).sum():3d}/{len(errors)} ({(np.abs(errors) <= 5).sum()/len(errors)*100:5.1f}%)")
print(f"    Within 2%:   {(np.abs(errors)/y_test <= 0.02).sum():3d}/{len(errors)} ({(np.abs(errors)/y_test <= 0.02).sum()/len(errors)*100:5.1f}%)")

# ============================================================================
# Export Results
# ============================================================================
results_df = pd.DataFrame({
    'Day': range(1, len(y_test) + 1),
    'Actual_Price': y_test.round(2),
    'Predicted_Price': y_pred.round(2),
    'Error_Dollar': errors.round(2),
    'Error_Percentage': (errors / y_test * 100).round(2),
    'Abs_Error': np.abs(errors).round(2)
})

results_df.to_csv('prediction_results.csv', index=False)
print("\n    ✓ Saved: prediction_results.csv")

print("\n" + "="*70)
print("✓ ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated Output Files:")
print("  1. stock_prediction_results.png - 5 detailed prediction plots")
print("  2. feature_importance.png      - Feature contribution analysis")
print("  3. prediction_results.csv      - Detailed prediction data")
print("\n" + "="*70)
