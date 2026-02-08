#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd      # For data manipulation and tables
import seaborn as sns    # For high-level visualization (Heatmaps)
import matplotlib.pyplot as plt  # For basic plotting control
import numpy as np
import sys
get_ipython().system('{sys.executable} -m pip install tslearn')
import tslearn
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance# For mathematical operations


print("Libraries loaded successfully!")


# In[31]:


df = pd.read_csv('traffic.csv')
df.head()


# In[4]:


# Convert the 'DateTime' column to actual date/time objects
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Verify the conversion
print(df.dtypes)


# In[5]:


# Extract the hour (0-23)
df['hour'] = df['DateTime'].dt.hour

# Extract the day of the week (Monday=0, Sunday=6)
df['weekday_num'] = df['DateTime'].dt.weekday

# Extract the month
df['month'] = df['DateTime'].dt.month

# Extract the specific date
df['date_only'] = df['DateTime'].dt.date

# Create a binary indicator for weekends (1 if Sat/Sun, 0 if Mon-Fri)
df['is_weekend'] = df['weekday_num'].apply(lambda x: 1 if x >= 5 else 0)

# Display the updated table
df.head()


# In[9]:


#To verify the number of unique junctions and total datarows.
print(f"Total rows: {len(df)}")
print(f"Unique Junctions: {df['Junction'].unique()}")


# In[10]:


# Group by Junction, Day of Week, and Hour to get the average traffic
# weekday_num was created in Phase 1 (0=Monday, 6=Sunday)
df_weekly = df.groupby(['Junction', 'weekday_num', 'hour'])['Vehicles'].mean().reset_index()

# Map numbers to names for better labels on your graph
days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
df_weekly['day_name'] = df_weekly['weekday_num'].map(days)


# In[12]:


df_weekly.head()


# In[14]:


#function to make a pivot table for a specific junction
def get_junction_pivot(junction_id):
    junction_data = df_weekly[df_weekly['Junction'] == junction_id]
    pivot = junction_data.pivot(index='hour', columns='weekday_num', values='Vehicles')
    # Rename columns to actual day names
    pivot.columns = [days[i] for i in pivot.columns]
    return pivot
pivot_j1=get_junction_pivot(1)
pivot_j2=get_junction_pivot(2)
pivot_j3=get_junction_pivot(3)
pivot_j4=get_junction_pivot(4)


# In[21]:


day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

#'figure' that will hold 4 sub-plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # 2 rows, 2 columns
axes = axes.flatten() # Flattens the 2x2 grid into a simple list of 4 spots

for i in range(1, 5): # This runs for Junctions 1, 2, 3, and 4
    # 1. Filter and Pivot
    junction_data = df_weekly[df_weekly['Junction'] == i]
    pivot = junction_data.pivot(index='hour', columns='day_name', values='Vehicles')
    pivot = pivot[day_order]

    # 2. Plot in the specific spot (axes[0], axes[1], etc.)
    sns.heatmap(pivot, cmap='RdYlBu_r', ax=axes[i-1])
    axes[i-1].set_title(f'Average Vehicles - Junction {i}')
    axes[i-1].set_xlabel('Weekday')
    axes[i-1].set_ylabel('Hour')

plt.tight_layout() # Prevents the graphs from overlapping
plt.show()


# In[23]:


# starting question2 
# 1. Average traffic for each Junction by Hour
junction_hourly = df.groupby(['Junction', 'hour'])['Vehicles'].mean().reset_index()

# 2. Average traffic for the WHOLE network (All 4 junctions added together)
network_total = df.groupby(['DateTime', 'hour'])['Vehicles'].sum().reset_index()
network_hourly = network_total.groupby('hour')['Vehicles'].mean().reset_index()

# View the peak hour for the network
network_peak_hour = network_hourly.loc[network_hourly['Vehicles'].idxmax()]
print(f"The single busiest hour for the whole network is: {network_peak_hour['hour']}:00")


# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Prepare Data
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['hour'] = df['DateTime'].dt.hour

# --- Question 2: Individual Junction Peaks ---
# Calculate average vehicle count for each hour per junction
junction_hourly = df.groupby(['Junction', 'hour'])['Vehicles'].mean().reset_index()

def find_peaks(data):
    # Defining standard windows: AM (7am-12pm) and PM (4pm-10pm)
    am_window = data[(data['hour'] >= 7) & (data['hour'] <= 12)]
    pm_window = data[(data['hour'] >= 16) & (data['hour'] <= 22)]

    am_peak = am_window.loc[am_window['Vehicles'].idxmax()]
    pm_peak = pm_window.loc[pm_window['Vehicles'].idxmax()]

    return am_peak['hour'], am_peak['Vehicles'], pm_peak['hour'], pm_peak['Vehicles']

# Store results for table
q2_results = []
for j in range(1, 5):
    j_data = junction_hourly[junction_hourly['Junction'] == j]
    am_h, am_v, pm_h, pm_v = find_peaks(j_data)
    q2_results.append({
        'Junction': j,
        'AM Peak Hour': int(am_h),
        'AM Volume': round(am_v, 2),
        'PM Peak Hour': int(pm_h),
        'PM Volume': round(pm_v, 2)
    })

# --- Question 3: Network-Level Peaks ---
# Aggregate (Sum) all vehicles across the 4 junctions for every timestamp
network_data = df.groupby('DateTime')['Vehicles'].sum().reset_index()
network_data['hour'] = network_data['DateTime'].dt.hour
network_hourly = network_data.groupby('hour')['Vehicles'].mean().reset_index()

# Find Network AM/PM Peaks
am_h_net, am_v_net, pm_h_net, pm_v_net = find_peaks(network_hourly)

# --- OUTPUT RESULTS ---
print("--- Question 2 Results (Intersection Peaks) ---")
print(pd.DataFrame(q2_results))

print("\n--- Question 3 Results (Network Peaks) ---")
print(f"Network AM Peak: {int(am_h_net)}:00 (Avg: {round(am_v_net, 2)} vehicles)")
print(f"Network PM Peak: {int(pm_h_net)}:00 (Avg: {round(pm_v_net, 2)} vehicles)")

# --- VISUALIZATION (Quantitative Evidence) ---
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
colors = sns.color_palette("muted", 4)

for i in range(1, 5):
    j_data = junction_hourly[junction_hourly['Junction'] == i]
    plt.plot(j_data['hour'], j_data['Vehicles'], label=f'Junction {i}', color=colors[i-1], alpha=0.6)

# Plot the Network Total as a bold dashed line
plt.plot(network_hourly['hour'], network_hourly['Vehicles'], label='WHOLE NETWORK', 
         color='black', linewidth=3, linestyle='--')

# Mark the peaks
plt.scatter([am_h_net, pm_h_net], [am_v_net, pm_v_net], color='yellow', 
            edgecolor='black', s=150, zorder=5, label='Network Peaks')

plt.title('Hourly Traffic Analysis: Individual vs. Network Peaks', fontsize=14)
plt.xlabel('Hour of Day')
plt.ylabel('Average Vehicle Count')
plt.xticks(range(24))
plt.legend()
plt.show()


# In[27]:


# 1. Get total daily traffic for the whole network
daily_traffic = df.groupby(df['DateTime'].dt.date)['Vehicles'].sum().reset_index()
daily_traffic.columns = ['Date', 'Total_Vehicles']
daily_traffic['Date'] = pd.to_datetime(daily_traffic['Date'])
daily_traffic['day_of_week'] = daily_traffic['Date'].dt.day_name()

# 2. Calculate the typical (Mean) and Variation (Std Dev) for EACH weekday
weekday_stats = daily_traffic.groupby('day_of_week')['Total_Vehicles'].agg(['mean', 'std']).reset_index()

# 3. Merge stats back to find outliers
daily_traffic = daily_traffic.merge(weekday_stats, on='day_of_week')

# 4. Define the Z-Score: How many standard deviations away is this day?
daily_traffic['z_score'] = (daily_traffic['Total_Vehicles'] - daily_traffic['mean']) / daily_traffic['std']

# 5. Define "Special Event Day" as Z-Score > 2 (The top 2.5% of busy days)
special_events = daily_traffic[daily_traffic['z_score'] > 2].sort_values(by='z_score', ascending=False)

print(f"Detected {len(special_events)} Special Event Days.")
print(special_events[['Date', 'day_of_week', 'Total_Vehicles', 'z_score']].head(10))


# In[29]:


#Anamoly plot for q4
import matplotlib.pyplot as plt

# Plotting the daily traffic
plt.figure(figsize=(15, 6))
plt.plot(daily_traffic['Date'], daily_traffic['Total_Vehicles'], color='gray', alpha=0.5, label='Normal Traffic')

# Overlay the special events in red
plt.scatter(special_events['Date'], special_events['Total_Vehicles'], color='red', label='Special Event Days', zorder=5)

# Adding a threshold line (e.g., Mean + 2SD)
plt.title('Detection of Special Event Days (Anomalies)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Total Network Vehicles')
plt.legend()
plt.show()


# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# 1. RE-DEFINE DATA (Ensuring 'X' is created in this step)
# Make sure 'df' is already loaded from your csv
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['date'] = df['DateTime'].dt.date
df['hour'] = df['DateTime'].dt.hour

# Pivot the data: We focus on Junction 1 (the highest volume junction)
# Each row = one day, Each column = one hour (0-23)
pivot_df = df[df['Junction'] == 1].pivot(index='date', columns='hour', values='Vehicles').dropna()

# --- THIS IS THE 'X' VARIABLE YOU WERE MISSING ---
# We scale the data so DTW compares the SHAPE, not the total volume
scaler = TimeSeriesScalerMeanVariance()
X = scaler.fit_transform(pivot_df.values) 

# 2. APPLY DTW CLUSTERING
# metric="dtw" is what allows for the time-warping "elastic" matching
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5, random_state=42, n_jobs=-1)
cluster_labels = model.fit_predict(X)

# 3. VISUALIZE THE RESULTS
plt.figure(figsize=(16, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)

    # Plot individual days in the cluster (faded gray)
    cluster_data = X[cluster_labels == i]
    for series in cluster_data[:15]: # Show first 15 days to see the pattern
        plt.plot(series.ravel(), color='gray', alpha=0.2)

    # Plot the 'Centroid' (The average "shape" of the cluster in red)
    plt.plot(model.cluster_centers_[i].ravel(), color='red', linewidth=2.5)
    plt.title(f'Cluster {i} Pattern')
    plt.xlabel('Hour (0-23)')
    plt.ylabel('Normalized Traffic')

plt.tight_layout()
plt.show()

# 4. REPORT EVIDENCE: Print how many days are in each cluster
print("Days per cluster:")
print(pd.Series(cluster_labels).value_counts())


# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# 1. Data Preparation
# Load your dataset
df = pd.read_csv('traffic.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['date'] = df['DateTime'].dt.date
df['hour'] = df['DateTime'].dt.hour

# Pivot data for Junction 1 (representing the network profile)
# Rows = Dates, Columns = Hours 0-23
pivot_df = df[df['Junction'] == 1].pivot(index='date', columns='hour', values='Vehicles').dropna()

# 2. Scaling
# We use Mean-Variance scaling to focus on the 'shape' of the day
scaler = TimeSeriesScalerMeanVariance()
X = scaler.fit_transform(pivot_df.values)

# 3. DTW Clustering
# Metric='dtw' allows for temporal misalignment (elastic matching)
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5, random_state=42, n_jobs=-1)
cluster_labels = model.fit_predict(X)

# 4. Identification of the Anomalous Cluster
# Attach labels back to our dates
pivot_df['cluster'] = cluster_labels

# Identify which cluster has the fewest days or the most erratic centroid
# In most runs, the 'Anomalous' cluster is the one with the least frequent label
anomalous_cluster_id = pd.Series(cluster_labels).value_counts().idxmin()
anomalous_days = pivot_df[pivot_df['cluster'] == anomalous_cluster_id].index

# 5. Output for Report Appendix
print(f"--- Question 5: DTW Anomalous Days (Cluster {anomalous_cluster_id}) ---")
print(f"Total days identified as temporal anomalies: {len(anomalous_days)}")
print("\nList of Dates:")
for d in anomalous_days:
    print(d)

# 6. Visualization
plt.figure(figsize=(6, 4))
cluster_center = model.cluster_centers_[anomalous_cluster_id].ravel()
plt.plot(cluster_center, color='red', linewidth=3)
plt.title(f'Anomalous Temporal Signature (Cluster {anomalous_cluster_id})')
plt.xlabel('Hour of Day')
plt.ylabel('Normalized Traffic')
plt.grid(True, alpha=0.3)
plt.show()


# In[43]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 1. Feature Engineering
df['month'] = df['DateTime'].dt.month
df['day'] = df['DateTime'].dt.day
df['day_of_week'] = df['DateTime'].dt.weekday
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# BONUS FEATURE: 1-Hour Lag (Previous hour's traffic for that junction)
# Motivation: Traffic is sequential; the best predictor of now is what happened 1 hour ago.
df = df.sort_values(['Junction', 'DateTime'])
df['lag_1'] = df.groupby('Junction')['Vehicles'].shift(1).fillna(0)

# 2. Data Selection
features = ['Junction', 'hour', 'day_of_week', 'month', 'day', 'is_weekend', 'lag_1']
X = df[features]
y = df['Vehicles']

# 3. Preprocessing & Splitting
# MLPs are highly sensitive to feature scaling!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Custom MLP Architecture (8 layers, 15 neurons each)
mlp_layers = tuple([15] * 8)
mlp = MLPRegressor(hidden_layer_sizes=mlp_layers, max_iter=500, activation='relu', solver='adam', random_state=42)

# 5. Linear Regression (Baseline)
lr = LinearRegression()

# 6. Training and Evaluation
models = {"MLP (8 Layers, 15 Neurons)": mlp, "Linear Regression": lr}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds))
    }

# Display Results
results_df = pd.DataFrame(results).T
print(results_df)


# In[45]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 1. Feature Engineering (Input features: Junction, hour, day, month, weekend, etc.)
df['month'] = df['DateTime'].dt.month
df['day'] = df['DateTime'].dt.day
df['day_of_week'] = df['DateTime'].dt.weekday
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# BONUS FEATURE: 1-Hour Lag (Corrected for latest Pandas version)
df = df.sort_values(['Junction', 'DateTime'])
df['traffic_lag_1'] = df.groupby('Junction')['Vehicles'].shift(1).bfill()

# 2. Input/Output Selection
features = ['Junction', 'hour', 'day_of_week', 'month', 'day', 'is_weekend', 'traffic_lag_1']
X = df[features]
y = df['Vehicles']

# 3. Preprocessing (Assumption: Standard Scaling is required for MLP convergence)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (Assumption: 80% Training, 20% Validation)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. MLP Design (Harshita: 8 layers, 15 neurons each)
mlp_layers = tuple([15] * 8)
mlp_model = MLPRegressor(
    hidden_layer_sizes=mlp_layers, 
    max_iter=1000, 
    activation='relu', 
    solver='adam', 
    random_state=42
)

# 5. Linear Regression (Baseline comparison)
lr_model = LinearRegression()

# 6. Training and Evaluation
results = []
for name, model in [("MLP (8x15 Layers)", mlp_model), ("Linear Regression", lr_model)]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results.append({"Model": name, "MAE": mae, "R2": r2, "RMSE": rmse})

# Display Quantitative Evaluation
performance_table = pd.DataFrame(results)
print(performance_table)


# In[ ]:





# In[46]:


from sklearn.metrics import mean_absolute_error, r2_score

# Function to train and return metrics
def train_and_evaluate(feature_list, X_df, y_df):
    X_subset = X_df[feature_list]
    # Re-scale for fair comparison
    X_scaled = StandardScaler().fit_transform(X_subset)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_df, test_size=0.2, random_state=42)

    # Custom 8x15 MLP
    model = MLPRegressor(hidden_layer_sizes=(15,)*8, max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return mean_absolute_error(y_test, preds), r2_score(y_test, preds)

# Baseline Features (provided in prompt)
baseline_features = ['Junction', 'hour', 'day_of_week', 'month', 'day', 'is_weekend']
# Enhanced Features (Bonus)
enhanced_features = baseline_features + ['traffic_lag_1']

# Comparison
mae_base, r2_base = train_and_evaluate(baseline_features, df, df['Vehicles'])
mae_enh, r2_enh = train_and_evaluate(enhanced_features, df, df['Vehicles'])

print(f"Baseline (Without Lag) -> MAE: {mae_base:.2f}, R2: {r2_base:.2f}")
print(f"Enhanced (With Lag)     -> MAE: {mae_enh:.2f}, R2: {r2_enh:.2f}")


# In[49]:


# 3. Models (Harshita's Architecture: 8 layers, 15 neurons)
mlp = MLPRegressor(hidden_layer_sizes=([15] * 8), max_iter=1000, random_state=42)
lr = LinearRegression()

mlp.fit(X_train, y_train)
lr.fit(X_train, y_train)

mlp_preds = mlp.predict(X_test)
lr_preds = lr.predict(X_test)

# --- GRAPH GENERATION (All 6 Graphs) ---
plt.figure(figsize=(18, 12))
sns.set_style("whitegrid")

# Graph 1: MLP Loss Curve
plt.subplot(2, 3, 1)
plt.plot(mlp.loss_curve_, color='blue')
plt.title('1. MLP Training Loss Curve')
plt.xlabel('Iterations'); plt.ylabel('Loss')

# Graph 2: MLP Predicted vs Actual
plt.subplot(2, 3, 2)
plt.scatter(y_test, mlp_preds, alpha=0.2, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('2. MLP: Predicted vs Actual')

# Graph 3: Linear Regression Predicted vs Actual
plt.subplot(2, 3, 3)
plt.scatter(y_test, lr_preds, alpha=0.2, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('3. LR: Predicted vs Actual')

# Graph 4: Residuals (Error Distribution)
plt.subplot(2, 3, 4)
residuals = y_test - mlp_preds
sns.histplot(residuals, kde=True, color='purple')
plt.title('4. Error (Residual) Distribution')

# Graph 5: Feature Importance (Fixed Warning)
plt.subplot(2, 3, 5)
importances = np.mean(np.abs(mlp.coefs_[0]), axis=1)
sns.barplot(x=features, y=importances, hue=features, palette='viridis', legend=False)
plt.xticks(rotation=45)
plt.title('5. Feature Importance')

# Graph 6: Metrics Comparison (Fixed Warning)
plt.subplot(2, 3, 6)
mae_mlp = mean_absolute_error(y_test, mlp_preds)
mae_lr = mean_absolute_error(y_test, lr_preds)
model_names = ['MLP (8x15)', 'Lin Reg']
mae_vals = [mae_mlp, mae_lr]
sns.barplot(x=model_names, y=mae_vals, hue=model_names, palette='coolwarm', legend=False)
plt.title('6. MAE Comparison (Lower is Better)')

plt.tight_layout()
plt.show()

print(f"Final Results:\nMLP MAE: {mae_mlp:.2f} | LR MAE: {mae_lr:.2f}")


# In[ ]:




