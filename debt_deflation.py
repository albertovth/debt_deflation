import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Streamlit sliders for the selected parameters
st.title('Macroeconomic Model for Debt Deflation')

st.markdown('''
You can use this app to model and visualize [debt deflation](https://en.wikipedia.org/wiki/Debt_deflation), in general terms for an economy. Debt deflation is understood as the process through which total debt decreases in the economy, leading to deflationary effects on growth, prices, employment, and wages. 

This model offers a simplified and discrete approach to more complex models, like Steve Keen's (2011) [monetary Minsky model](https://keenomics.s3.amazonaws.com/debtdeflation_media/papers/PaperPrePublicationProof.pdf), which is based on differential equations. In this model, the differential equations are replaced by growth rates calculated on macroeconomic variables. The result is a less smooth visualization, but one that can be easier to grasp intuitively.

The app includes a series of sliders to adjust various preset parameters. Most of the presets are calibrated to regressions for the Norwegian economy, using the latest available data. However, this does not apply to the variable called **threshold_periods_for_financial_collapse**. This variable is exogenous to the other variables in the model, and can be adjusted to simulate different scenarios.

The number of periods leading to financial collapse is difficult to predict, and it is uncertain whether a collapse will occur within the analyzed period. This depends on several factors, including financial fragility, diminishing returns on debt, debt servicing costs, and confidence. Establishing an estimate for **threshold_periods_for_financial_collapse** should be done on the basis of careful analysis of financial conditions in each individual economy, based on research like performed by [Irving Fisher (1933)](https://doi.org/10.2307/1907327), [Steve Keen (2001)](https://www.bloomsbury.com/uk/debunking-economics-9781848139954/), [Hyman Minsky (1986)](https://www.levyinstitute.org/publications/stabilizing-an-unstable-economy), central banks and projects on economic downturns.
It is important to add as a disclaimer that there is no model that can with complete accuracy predict financial shocks and debt deflation. This is because of the inherent complexity of the processes involved, ranging from structural economic relations to psychological aspects of economies. The purpose of this simple illustration is therefore not to predict, but to simplify and illustrate complex ideas graphically. 
''')

c = st.slider('Private consumption as proportion of total debt stock', min_value=0.05, max_value=0.30, value=0.12, step=0.01)
gc = st.slider('Government consumption as proportion of total debt stock', min_value=0.2, max_value=0.8, value=0.5, step=0.01)
i = st.slider('Total gross investment as proportion of total debt stock', min_value=0.05, max_value=0.30, value=0.09, step=0.01)
ai_investment_ratio = st.slider('AI autonomous investment ratio as proportion of GDP', min_value=0.01, max_value=0.10, value=0.05, step=0.01)
tax_rate = st.slider('Tax rate as proportion of GDP', min_value=0.2, max_value=0.6, value=0.4, step=0.01)
k = st.slider('Proportion of GDP from net exports', min_value=0.01, max_value=0.10, value=0.01, step=0.01)
x = st.slider('Export total debt relation', min_value=0.01, max_value=0.10, value=0.04, step=0.01)
private_debt_ratio_initial = st.slider('Initial private debt ratio as proportion of GDP', min_value=1.0, max_value=3.0, value=2.3, step=0.1)
government_debt_ratio_initial = st.slider('Initial government debt ratio as proportion of GDP', min_value=0.1, max_value=1.0, value=0.45, step=0.05)
debt_growth_rate = st.slider('Debt growth rate (initial)', min_value=0.01, max_value=0.10, value=0.0457, step=0.001)
threshold_periods_for_financial_collapse = st.slider('Threshold periods for financial collapse', min_value=1, max_value=30, value=15, step=1)
periods = st.slider('Select the number of periods (years) to simulate:', min_value=10, max_value=100, value=20, step=1)

# Parameters calibrated
A = 0               # Autonomous consumption
debt_influence_factor = 0.8  # Reduce the influence of debt on consumption
gi = 0.35            # Government propensity to invest
initial_debt_to_gdp_ratio = 2.75  # Total debt starts at 2.75 times GDP
debt_to_gdp_threshold = 3.25  # Maximum allowed debt-to-GDP ratio

# Time steps and years (extended to 40 years)
time_steps = periods
years = np.arange(2024, 2024 + time_steps)

# Employment equation (with a lag)
E = np.zeros(time_steps)
wage_share = np.zeros(time_steps)
productivity_growth = np.zeros(time_steps)

beta_E = 0.4  # Employment sensitivity to GDP growth
alpha = 0.5   # Wage share sensitivity to employment
gamma = 0.4   # Wage share sensitivity to productivity growth
delta = 0.07  # Productivity growth sensitivity to GDP growth

Y = np.zeros(time_steps)  # GDP
C = np.zeros(time_steps)  # Private consumption
G = np.zeros(time_steps)  # Government spending
I = np.zeros(time_steps)  # Investment
X_M = np.zeros(time_steps)  # Net exports
total_debt = np.zeros(time_steps)  # Total debt stock (private + government)
private_income = np.zeros(time_steps)  # Private income
government_income = np.zeros(time_steps)  # Government income (taxes)
ai_investment = np.zeros(time_steps)  # AI autonomous investment

# Debt-to-GDP ratio tracking
debt_to_gdp_ratio = np.zeros(time_steps)

# Rate of growth arrays for Y, C, G, I, Debt
Ro_Y = np.zeros(time_steps)
Ro_C = np.zeros(time_steps)
Ro_G = np.zeros(time_steps)
Ro_I = np.zeros(time_steps)
Ro_Debt = np.zeros(time_steps)
Ro_TotalDebt = np.zeros(time_steps)

# Pre-initialization step for year -1 to set the flow of income (GDP)
Y_minus1 = 579  # Set a pre-initial GDP value (slightly lower than Y[0])

# Initial values for year 2023 (base year)
total_debt_0 = initial_debt_to_gdp_ratio * 579
private_debt_0 = (private_debt_ratio_initial / initial_debt_to_gdp_ratio) * total_debt_0
government_debt_0 = (government_debt_ratio_initial / initial_debt_to_gdp_ratio) * total_debt_0

C_0 = 9 + c * private_debt_0
G_0 = 12 + gc * government_debt_0
I_0 = 6 + i * private_debt_0
X_M_0 = 3 + x * total_debt_0

Y[0] = C_0 + G_0 + I_0 + X_M_0

total_debt[0] = initial_debt_to_gdp_ratio * 579
private_income[0] = (1 - tax_rate) * Y[0]
government_income[0] = tax_rate * Y[0]
ai_investment[0] = ai_investment_ratio * Y[0]

# Initial growth rates set to 0 for 2024
Ro_Y[0] = 0
Ro_C[0] = 0
Ro_G[0] = 0
Ro_I[0] = 0
Ro_Debt[0] = 0
Ro_TotalDebt[0] = 0

# Initialize variables
threshold_exceed_counter = 0
collapse_triggered = False
growth_failure_counter = 0
growth_benchmark = 0.1

# Model loop for time steps starting from 2024 (year 1)
for t in range(1, time_steps):
    Y_reference = Y[t-1] if t > 1 else Y_minus1

    private_income[t] = (1 - tax_rate) * Y_reference
    government_income[t] = tax_rate * Y_reference

    C[t] = A + c * (private_debt_ratio_initial / initial_debt_to_gdp_ratio) * total_debt[t-1]
    G[t] = gc * (government_debt_ratio_initial / initial_debt_to_gdp_ratio) * total_debt[t-1]
    I[t] = i * (private_debt_ratio_initial / initial_debt_to_gdp_ratio) * total_debt[t-1] + ai_investment_ratio * Y[t-1]
    X_M[t] = x * total_debt[t-1] + k * Y[t-1]

    Y[t] = C[t] + G[t] + I[t] + X_M[t]

    # Update debt stock with a growth rate, and monitor for collapse trigger based on either counter
    if not collapse_triggered:
        if (total_debt[t-1] / Y_reference) < debt_to_gdp_threshold:
            total_debt[t] = total_debt[t-1] * (1 + debt_growth_rate)
        else:
            total_debt[t] = total_debt[t-1]  # Debt stock stagnates after reaching the threshold
            threshold_exceed_counter += 1

        if Ro_Y[t] < growth_benchmark:
            growth_failure_counter += 1  # Increment failure counter if growth falls below benchmark
        else:
            growth_failure_counter = 0  # Reset counter if growth recovers

        if growth_failure_counter >= threshold_periods_for_financial_collapse or threshold_exceed_counter >= threshold_periods_for_financial_collapse or (growth_failure_counter + threshold_exceed_counter) >= threshold_periods_for_financial_collapse:
            collapse_triggered = True
            debt_growth_rate = -0.0457  # Set the permanent collapse debt growth rate

    else:
        total_debt[t] = total_debt[t-1] * (1 + debt_growth_rate)

    debt_to_gdp_ratio[t] = total_debt[t] / Y[t]

    # Calculate the rate of growth (Ro) for GDP, Consumption, Investment, Government Spending, and Debt
    if C[t-1] > 0:
        Ro_C[t] = (C[t] - C[t-1]) / C[t-1] * 100
    else:
        Ro_C[t] = 0

    if Y[t-1] > 0:
        Ro_Y[t] = (Y[t] - Y[t-1]) / Y[t-1] * 100
    else:
        Ro_Y[t] = 0

    if G[t-1] > 0:
        Ro_G[t] = (G[t] - G[t-1]) / G[t-1] * 100
    else:
        Ro_G[t] = 0

    if I[t-1] > 0:
        Ro_I[t] = (I[t] - I[t-1]) / I[t-1] * 100
    else:
        Ro_I[t] = 0

    if total_debt[t-1] > 0:
        Ro_Debt[t] = (total_debt[t] - total_debt[t-1]) / total_debt[t-1] * 100
    else:
        Ro_Debt[t] = 0

    Ro_TotalDebt[t] = Ro_Debt[t]

    employment_floor = 0.7
    E[t] = min(1, max(employment_floor, beta_E * Ro_Y[t-1]))

    # Productivity growth based on lagged GDP growth (no cap needed here)
    productivity_growth[t] = delta * Ro_Y[t-1]

    wage_floor = 0.45
    # Wage share as a function of lagged employment and productivity growth, capped at 1 (100%)
    wage_share[t] = min(1, max(wage_floor, alpha * E[t-1] + gamma * productivity_growth[t-1]))

# Plot the main results (GDP, Consumption, Investment, etc.)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years[1:], Y[1:], label='GDP (Y)')
ax.plot(years[1:], C[1:], label='Consumption (C)')
ax.plot(years[1:], G[1:], label='Government Spending (G)')
ax.plot(years[1:], I[1:], label='Investment (I)')
ax.plot(years[1:], X_M[1:], label='Net Exports (X-M)')
ax.plot(years[1:], ai_investment[1:], label='AI Investment (AI)')
ax.set_title('Macroeconomic Model for Debt Deflation (Starting from 2024)')
ax.set_xlabel('Year')
ax.set_ylabel('Monetary Units')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Plot GDP and Total Debt Stock on the same chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years[1:], Y[1:], label='GDP (Y)')
ax.plot(years[1:], total_debt[1:], label='Total Debt Stock')
ax.set_title('GDP and Total Debt Stock Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Monetary Units')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Plot the debt-to-GDP ratio
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years[1:], debt_to_gdp_ratio[1:], label='Debt-to-GDP Ratio')
ax.set_title('Debt-to-GDP Ratio Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Debt-to-GDP Ratio')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Plot the rates of growth (Ro) including Debt Stock Growth starting from 2024
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years[1:], Ro_Y[1:], label='Rate of Growth of GDP (Ro_Y)')
ax.plot(years[1:], Ro_C[1:], label='Rate of Growth of Consumption (Ro_C)')
ax.plot(years[1:], Ro_G[1:], label='Rate of Growth of Government Spending (Ro_G)')
ax.plot(years[1:], Ro_I[1:], label='Rate of Growth of Investment (Ro_I)')
ax.plot(years[1:], Ro_Debt[1:], label='Rate of Growth of Debt Stock (Ro_Debt)')
ax.set_title('Rate of Growth of GDP, Consumption, Investment, Government Spending, and Debt Stock (2024 Onward)')
ax.set_xlabel('Year')
ax.set_ylabel('Rate of Growth (%)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Plot wage share and employment over time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years[1:], wage_share[1:], label='Wage Share')
ax.plot(years[1:], E[1:], label='Employment Rate')
ax.set_title('Wage Share and Employment Rate Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Closing the loop in Wage Share vs. Employment
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(E[1:], wage_share[1:], color='blue', marker='o', label='Wage Share vs. Employment')
ax.plot([E[-1], E[1]], [wage_share[-1], wage_share[1]], 'k--')  # Connect last and first points
ax.set_title('Wage Share vs. Employment Rate (Closed Loop)')
ax.set_xlabel('Employment Rate')
ax.set_ylabel('Wage Share')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Plot GDP Growth and Debt Growth on the same chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years[1:], Ro_Y[1:], label='Rate of Growth of GDP (Ro_Y)', color='blue')
ax.plot(years[1:], Ro_TotalDebt[1:], label='Rate of Growth of Total Debt (Ro_TotalDebt)', color='red')
ax.set_title('Rate of Growth of GDP and Total Debt')
ax.set_xlabel('Year')
ax.set_ylabel('Rate of Growth (%)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Plot the rate of growth of Consumption over time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years[1:], Ro_C[1:], label='Rate of Growth of Consumption (Ro_C)')
ax.set_title('Rate of Growth of Consumption (2024 Onward)')
ax.set_xlabel('Year')
ax.set_ylabel('Rate of Growth (%)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Plot wage share and employment over time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years[1:], wage_share[1:], label='Wage Share')
ax.plot(years[1:], E[1:], label='Employment Rate')
ax.set_title('Wage Share and Employment Rate Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Closing the loop in Wage Share vs. Employment
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(E[1:], wage_share[1:], color='blue', marker='o', label='Wage Share vs. Employment')
ax.plot([E[-1], E[1]], [wage_share[-1], wage_share[1]], 'k--')  # Connect last and first points
ax.set_title('Wage Share vs. Employment Rate (Closed Loop)')
ax.set_xlabel('Employment Rate')
ax.set_ylabel('Wage Share')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# 3D Plot with closed loop
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(E[1:], wage_share[1:], total_debt[1:], color='red', marker='o', label='Wage Share, Employment, and Total Debt')
ax.plot([E[-1], E[1]], [wage_share[-1], wage_share[1]], [total_debt[-1], total_debt[1]], 'k--')  # Close loop in 3D
ax.set_title('Wage Share, Employment, and Total Debt (Closed Loop)')
ax.set_xlabel('Employment Rate')
ax.set_ylabel('Wage Share')
ax.set_zlabel('Total Debt')
st.pyplot(fig)
