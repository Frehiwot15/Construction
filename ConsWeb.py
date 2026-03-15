import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
st.set_page_config(page_title="Advanced Construction Risk", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700;800&display=swap');

body {font-family: 'Quicksand', sans-serif;}

.box {
  width: 140px;
  height: auto;
  float: left;
  transition: .5s linear;
  position: relative;
  display: block;
  overflow: hidden;
  padding: 15px;
  text-align: center;
  margin: 0 5px;
  background: transparent;
  text-transform: uppercase;
  font-weight: 900;
}

.box:before {
  position: absolute;
  content: '';
  left: 0;
  bottom: 0;
  height: 4px;
  width: 100%;
  border-bottom: 4px solid transparent;
  border-left: 4px solid transparent;
  box-sizing: border-box;
  transform: translateX(100%);
}

.box:after {
  position: absolute;
  content: '';
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  border-top: 4px solid transparent;
  border-right: 4px solid transparent;
  box-sizing: border-box;
  transform: translateX(-100%);
}

.box:hover {
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
}

.box:hover:before {
  border-color: #262626;
  height: 100%;
  transform: translateX(0);
  transition: .3s transform linear, .3s height linear .3s;
}

.box:hover:after {
  border-color: #262626;
  height: 100%;
  transform: translateX(0);
  transition: .3s transform linear, .3s height linear .5s;
}

button {
  color: black;
  text-decoration: none;
  cursor: pointer;
  outline: none;
  border: none;
  background: transparent;
}

</style>
""", unsafe_allow_html=True)

st.title("🏗️ Advanced Construction Scheduling Risk Analytics ")
st.write("Monte Carlo Simulation with Risk Contribution & Critical Activity Detection")

# -----------------------------
# USER INPUT PANEL
# -----------------------------
st.sidebar.header("Project Configuration")

num_floors = st.sidebar.slider("Number of Floors", 1, 60, 0)
simulations = st.sidebar.slider("Simulation Runs", 100, 300, 500)
planned_duration = st.sidebar.number_input("Planned Duration (Days)", 100, 500, 180)

st.sidebar.subheader("Phase Duration Inputs (Days)")

def phase_input(name, default):
    opt = st.sidebar.number_input(f"{name} Optimistic", value=default-2)
    ml = st.sidebar.number_input(f"{name} Most Likely", value=default)
    pess = st.sidebar.number_input(f"{name} Pessimistic", value=default+3)
    return (opt, ml, pess)

phases = {
    "Site Preparation": phase_input("Site Preparation", 5),
    "Foundation": phase_input("Foundation", 20),
    "Structure": phase_input("Structure", 18),
    "MEP": phase_input("MEP Installation", 12),
    "Interior": phase_input("Interior Finishing",30),
    "Floor Work": phase_input("Per Floor Construction", 2)
}

# -----------------------------
# SIMULATION ENGINE
# -----------------------------
def triangular(opt, ml, pess, size):
    return np.random.triangular(opt, ml, pess, size) # size parametr is for number of iteration

def run_simulation():
    total_results = []
    contribution_tracker = {phase: [] for phase in phases}

    for _ in range(simulations):

        phase_times = {}

        for phase in phases:
            if phase == "Floor Work":
                phase_times[phase] = triangular(*phases[phase], num_floors).sum()
            else:
                phase_times[phase] = triangular(*phases[phase], 1)[0]

        total_time = sum(phase_times.values())
        total_results.append(total_time)

        for phase in phase_times:
            contribution_tracker[phase].append(phase_times[phase])

    return np.array(total_results), contribution_tracker

# -----------------------------
# RUN BUTTON
# -----------------------------
run_col1, run_col2, run_col3 = st.columns(3)
with run_col2:
    run_clicked = st.button("Simulate Risk")

if run_clicked:
    results, contribution = run_simulation()

    # -----------------------------
    # KPI METRICS
    # -----------------------------
    avg = np.mean(results)
    p50 = np.percentile(results, 50)
    p80 = np.percentile(results, 80)
    p90 = np.percentile(results, 90)
    delay_prob = np.sum(results > planned_duration) / simulations

    st.subheader("📊 Executive Risk Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Expected Duration", f"{avg:.1f} Days")
    col2.metric("P50 Confidence", f"{p50:.1f}")
    col3.metric("P80 Confidence", f"{p80:.1f}")
    col4.metric("P90 Confidence", f"{p90:.1f}")
    col5.metric("Delay Probability", f"{delay_prob*100:.1f}%")

    st.session_state.last_avg = avg
    st.session_state.last_p50 = p50
    st.session_state.last_p80 = p80
    st.session_state.last_p90 = p90
    st.session_state.last_results = results

    # -----------------------------
    # COMPLETION DISTRIBUTION
    # -----------------------------
    st.subheader("Project Completion Distribution")

    fig = px.histogram(results, nbins=40, title="Completion Time Probability Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # RISK CONTRIBUTION ANALYSIS
    # -----------------------------
    st.subheader("Risk Contribution Analysis")

    contribution_avg = {
        phase: np.mean(contribution[phase])
        for phase in contribution
    }

    contrib_df = pd.DataFrame(
        list(contribution_avg.items()),
        columns=["Phase", "Average Duration"]
    ).sort_values("Average Duration", ascending=False)

    fig2 = px.bar(contrib_df, x="Phase", y="Average Duration", title="Phase Risk Contribution (Tornado Style)")
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # CRITICAL ACTIVITY DETECTION
    # -----------------------------
    st.subheader("Critical Activity Identification")

    std_dev = {
        phase: np.std(contribution[phase])
        for phase in contribution
    }
    mean_phase = {
        phase: np.mean(contribution[phase])
        for phase in contribution
    }
    cv = {
        phase: (std_dev[phase] / mean_phase[phase]) if mean_phase[phase] != 0 else 0
        for phase in contribution
    }

    std_df = pd.DataFrame(
        [
            {
                "Phase": phase,
                "Risk Variability": std_dev[phase],
                "Average Duration": mean_phase[phase],
                "CV (Std/Mean)": cv[phase],
            }
            for phase in contribution
        ]
    ).sort_values("CV (Std/Mean)", ascending=False)

    st.dataframe(std_df)

    most_sensitive_phase = std_df.iloc[0]["Phase"]
    st.info(f"⚠️ Most Risk Sensitive Phase: {most_sensitive_phase}")
    st.session_state.last_avg = avg
    st.session_state.last_results = results

# -----------------------------
# MULTI PROJECT COMPARISON (SCENARIO STORAGE)
# -----------------------------
st.subheader("Scenario Comparison & Storage")

scenario_name = st.text_input("Save Scenario Name", key="scenario_name")

if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}

save_col, show_col = st.columns(2)
with save_col:
    save_clicked = st.button("Save Scenario")
with show_col:
    show_scenarios = st.button("Show Scenarios")

if save_clicked:
    clean_name = scenario_name.strip()
    if not clean_name:
        st.error("Enter a scenario name before saving.")
    elif "last_avg" not in st.session_state:
        st.error("Run simulation first before saving a scenario.")
    else:
        st.session_state.scenarios[clean_name] = st.session_state.last_avg
        st.success(f"Scenario '{clean_name}' Saved")

if show_scenarios and st.session_state.scenarios:
    scenario_df = pd.DataFrame(
        list(st.session_state.scenarios.items()),
        columns=["Scenario", "Expected Duration"]
    )
    st.dataframe(scenario_df)
    st.bar_chart(scenario_df.set_index("Scenario"))

    # Download comparison as Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        scenario_df.to_excel(writer, index=False, sheet_name="Scenarios")
        summary_df = pd.DataFrame({
            "Metric": ["Last Expected Duration", "P50", "P80", "P90"],
            "Value": [
                st.session_state.get("last_avg", "N/A"),
                st.session_state.get("last_p50", "N/A"),
                st.session_state.get("last_p80", "N/A"),
                st.session_state.get("last_p90", "N/A")
            ]
        })
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
    excel_buffer.seek(0)
    st.download_button("Export Scenario Comparison", excel_buffer, "scenario_comparison.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if "last_results" in st.session_state:
    df = pd.DataFrame(st.session_state.last_results, columns=["Completion Time"])
    csv = df.to_csv(index=False).encode()
    st.download_button("Export Simulation Data", csv, "risk_results.csv", "text/csv")
