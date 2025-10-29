# 🎓 Prometheus & Grafana Quick Guide

## ✅ Step-by-Step Instructions

---

## 📊 PROMETHEUS Tutorial (http://localhost:9090)

### What is Prometheus?
Prometheus shows RAW metrics data. It's where the data is stored.

### Step 1: Open Prometheus
1. Open: http://localhost:9090
2. You should see a query box at the top

### Step 2: Make Some Predictions First!
**IMPORTANT:** You need to generate data first!

1. Go to Streamlit: http://localhost:8501
2. Click "DDoS Attack" → Click "Analyze Traffic"
3. Click "Web Attack" → Click "Analyze Traffic"  
4. Click "Normal Traffic" → Click "Analyze Traffic"
5. Make at least 5-10 predictions

### Step 3: Query Metrics in Prometheus

Now go back to Prometheus (http://localhost:9090)

**Query 1: Total Predictions**
1. In the query box, type: `predictions_total`
2. Click the blue "Execute" button
3. You should see:
   ```
   predictions_total{prediction="Attack"} 7
   predictions_total{prediction="Normal"} 3
   ```
4. Click the "Graph" tab to see it as a chart!

**Query 2: API Requests**
1. Clear the query box
2. Type: `api_requests_total`
3. Click "Execute"
4. Shows how many API calls to each endpoint

**Query 3: Expert Weights (Who's doing the work?)**
1. Type: `expert_gating_weight`
2. Click "Execute"
3. Shows:
   - Tabular Expert (FT-Transformer): ~98%
   - Temporal Expert (CNN): ~2%

**Query 4: Model Status**
1. Type: `model_loaded`
2. Click "Execute"
3. Should show: `1` (model is loaded)

**Query 5: Request Rate (attacks per second)**
1. Type: `rate(predictions_total{prediction="Attack"}[1m])`
2. Click "Execute"
3. Click "Graph" tab
4. Shows attack detection rate over time!

---

## 🎨 GRAFANA Tutorial (http://localhost:3000)

### What is Grafana?
Grafana makes BEAUTIFUL dashboards from Prometheus data.

### Step 1: Login to Grafana
1. Open: http://localhost:3000
2. Login:
   - Username: `admin`
   - Password: `admin`
3. Click "Skip" when asked to change password

### Step 2: Add Prometheus as Data Source
1. Click the **⚙️ (gear icon)** on the left sidebar
2. Click **"Data sources"**
3. Click **"Add data source"** (blue button)
4. Click **"Prometheus"** (first option)
5. In the URL field, enter: `http://prometheus:9090`
6. Scroll down and click **"Save & Test"**
7. You should see green message: "Successfully queried the Prometheus API"

### Step 3: Create Your First Dashboard

**Create Dashboard:**
1. Click **➕ (plus icon)** on left sidebar
2. Click **"Create Dashboard"**
3. Click **"Add visualization"**

**Panel 1 - Total Predictions (Big Number):**
1. In "Query" field, type: `sum(predictions_total)`
2. On the right side, change:
   - Panel title: "Total Predictions"
   - Visualization type: "Stat" (top right dropdown)
3. Click **"Apply"** (top right)

**Add Another Panel:**
1. Click **"Add" → "Visualization"** (top right)
2. In "Query" field, type: `predictions_total`
3. On the right side, change:
   - Panel title: "Attack vs Normal"
   - Visualization type: "Pie chart"
4. Scroll down on right, find "Legend" section
5. Set Legend values: `{{prediction}}`
6. Click **"Apply"**

**Add Third Panel - Request Rate:**
1. Click **"Add" → "Visualization"**
2. Query: `rate(api_requests_total[1m])`
3. Panel title: "API Request Rate"
4. Visualization: "Time series" (line graph)
5. Legend: `{{endpoint}} - {{method}}`
6. Click **"Apply"**

**Add Fourth Panel - Expert Weights:**
1. Click **"Add" → "Visualization"**
2. Query: `expert_gating_weight`
3. Panel title: "Expert Contribution"
4. Visualization: "Pie chart"
5. Legend: `{{expert}}`
6. Click **"Apply"**

**Save Dashboard:**
1. Click **💾 (save icon)** at top
2. Name: "MoE Cybersecurity Monitor"
3. Click **"Save"**

**Enable Auto-Refresh:**
1. Top right corner, click the refresh dropdown
2. Select **"5s"** or **"10s"**
3. Now it updates automatically!

---

## 🎬 Live Demo - See It All Work!

**Do This:**

1. **Open 3 Browser Tabs:**
   - Tab 1: Streamlit (http://localhost:8501)
   - Tab 2: Prometheus (http://localhost:9090)
   - Tab 3: Grafana (http://localhost:3000)

2. **In Streamlit (Tab 1):**
   - Select "DDoS Attack"
   - Click "Analyze Traffic"
   - Wait 5 seconds

3. **In Prometheus (Tab 2):**
   - Type query: `predictions_total`
   - Click "Execute"
   - You should see the Attack counter went up by 1!

4. **In Grafana (Tab 3):**
   - Watch your dashboards
   - The numbers should update automatically
   - Pie chart changes color

5. **Repeat:**
   - Make more predictions in Streamlit
   - Watch Prometheus and Grafana update!

---

## 🐛 Troubleshooting

### "No data in Prometheus"
**Solution:**
1. Make predictions in Streamlit first!
2. Wait 10-15 seconds (Prometheus scrapes every 10s)
3. Query: `predictions_total` and click Execute

### "Grafana shows 'No data'"
**Solution:**
1. Check data source is added (Settings → Data Sources)
2. URL must be: `http://prometheus:9090` (NOT localhost!)
3. Make predictions to generate data
4. Change time range: Top right, click time range, select "Last 15 minutes"

### "Prometheus says no data queried yet"
**Solution:**
1. Type a query first! (e.g., `predictions_total`)
2. Click the blue "Execute" button
3. Make sure you made predictions in Streamlit

---

## 📊 What Each Metric Means

**predictions_total**
- **What**: Counts how many predictions of each type
- **Use**: See attack vs normal ratio
- **Example**: `predictions_total{prediction="Attack"} 15`

**api_requests_total**
- **What**: Counts API calls to each endpoint
- **Use**: Monitor API usage
- **Example**: `api_requests_total{endpoint="/predict",method="POST"} 20`

**expert_gating_weight**
- **What**: Shows which expert (FT-Transformer vs CNN) is being used
- **Use**: Understand model behavior
- **Example**: 
  - `expert_gating_weight{expert="Tabular Expert"} 0.98` (98%)
  - `expert_gating_weight{expert="Temporal Expert"} 0.02` (2%)

**model_loaded**
- **What**: 1 if model loaded, 0 if failed
- **Use**: Health check
- **Example**: `model_loaded 1`

---

## 🎯 Your Action Plan

**Right Now:**

1. ✅ Make 10 predictions in Streamlit
2. ✅ Go to Prometheus → Query `predictions_total` → Click Execute
3. ✅ Go to Grafana → Add Prometheus data source → Create dashboard
4. ✅ Watch it update in real-time!

**You'll know it's working when:**
- Prometheus shows numbers when you query
- Grafana charts update automatically
- Making a prediction in Streamlit updates the metrics

**Need help? Common queries to copy-paste:**
```
predictions_total
api_requests_total  
expert_gating_weight
model_loaded
rate(predictions_total[1m])
```

Good luck! 🚀
