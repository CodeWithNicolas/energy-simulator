# Energy Management System Simulator

A Python-based energy management system simulator featuring Model Predictive Control (MPC) for optimal battery dispatch in residential solar+storage systems.

## Quick Start

**Prerequisites**: You need to provide a `slpe.csv` file containing load profile data. The CSV should be semicolon-separated (`;`) and contain hourly energy consumption data. Supported formats:
- 24 values (1 day, tiled to full year)
- 8760 values (full year, hourly)
- Can use comma or dot as decimal separator

1. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib plotly cvxpy requests
   ```

2. **Run basic simulation**:
   ```python
   python simulator.py
   ```

3. **Generate MPC animation**:
   ```python
   python mpc_plots.py
   ```

## What You'll See

- **Static plots**: Daily/weekly/monthly energy consumption and PV production  
- **MPC animation**: Real-time visualization of predictions

