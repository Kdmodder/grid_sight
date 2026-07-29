"""
GridSight - Phase-Aware Risk Forecasting UI
Streamlit interface for cumulative delay and cost overrun prediction
"""
import requests
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import random

# 
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "champion_model.pkl")
# FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_list.json")
# PHASE_PATH = os.path.join(BASE_DIR, "models", "phase_mapping.json")
# 

# Page configuration
st.set_page_config(
    page_title="GridSight - Cost & Delay Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
# 
# # Load champion model once (cached)
# @st.cache_resource
# 
# 
# def load_model():
#     """Load the trained champion model safely (local + Render)."""
#     try:
#         BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#         MODEL_PATH = os.path.join(BASE_DIR, "models", "champion_model.pkl")
# 
#         with open(MODEL_PATH, "rb") as f:
#             champion = pickle.load(f)
# 
#         return champion
# 
#     except Exception as e:
#         st.error(f"❌ Error loading model: {e}")
#         st.stop()
# 
# 
# # Load model
# champion = load_model()
# delay_model = champion['delay_model']
# cost_model = champion['cost_model']
# encoders = champion['encoders_and_features']

# ============================================================================
# AUTOFILL HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_test_data():
    """Load test_data.csv for autofill functionality"""
    try:
        df = pd.read_csv('test_data.csv')
        return df
    except Exception as e:
        st.error(f"Could not load test_data.csv: {e}")
        return None

def get_random_project_for_phase(phase_num):
    """Get a random project that has data filled till the selected phase"""
    df = load_test_data()
    if df is None or df.empty:
        return None
    
    # Filter projects that have data till the selected phase
    filtered_df = df[df['Phase_Num'] == phase_num]
    
    if filtered_df.empty:
        return None
    
    # Select a random row
    random_row = filtered_df.sample(n=1).iloc[0]
    return random_row

def autofill_from_test_data(phase_num):
    """Autofill session state with data from test_data.csv for the selected phase"""
    project = get_random_project_for_phase(phase_num)
    
    if project is None:
        st.warning(f"No test data available for Phase {phase_num}")
        return False
    
    # Helper function to safely get value with NaN handling
    def safe_get(value, default):
        if pd.isna(value):
            return default
        return value
    
    # Helper function to convert string date to datetime.date object
    def parse_date(date_str):
        if pd.isna(date_str):
            return datetime.strptime('01-01-2024', '%d-%m-%Y').date()
        try:
            return datetime.strptime(str(date_str), '%d-%m-%Y').date()
        except:
            try:
                return datetime.strptime(str(date_str), '%Y-%m-%d').date()
            except:
                return datetime.strptime('01-01-2024', '%d-%m-%Y').date()
    
    # Helper function to normalize terrain type
    def normalize_terrain(terrain):
        terrain_str = str(terrain)
        # Map 'Flat' to 'Plain', and handle other variations
        if pd.isna(terrain) or terrain_str == 'nan':
            return 'Mixed'
        terrain_map = {
            'Flat': 'Plain',
            'Plain': 'Plain',
            'Hilly': 'Hilly',
            'Forest': 'Forest',
            'Mixed': 'Mixed'
        }
        return terrain_map.get(terrain_str, 'Mixed')
    
    # Map project data to session state - Metadata with NaN handling
    metadata = {
        'project_type': safe_get(project.get('Project_Type'), 'Line'),
        'voltage_level': safe_get(project.get('Voltage_Level'), '400kV'),
        'contract_type': safe_get(project.get('Contract_Type'), 'EPC'),
        'terrain_type': normalize_terrain(project.get('Terrain_Type')),
        'project_size': float(safe_get(project.get('Project_Size'), 200)),
        'total_budget_cr': float(safe_get(project.get('Total_Budget_Cr'), 300)),
        'planned_duration': int(safe_get(project.get('Planned_Duration'), 24)),
        'project_location': safe_get(project.get('Project_Location'), 'Unknown'),
        'start_date': parse_date(project.get('Start_Date'))
    }
    st.session_state['saved_metadata'] = metadata
    st.session_state['metadata_saved'] = True
    
    # Map phase data - get all rows for this project up to the selected phase
    df = load_test_data()
    project_id = project['Project_ID']
    project_rows = df[(df['Project_ID'] == project_id) & (df['Phase_Num'] <= phase_num)].sort_values('Phase_Num')
    
    phases_data = {}
    
    # Engineering Phase (Phase 1)
    if phase_num >= 1:
        p1 = project_rows[project_rows['Phase_Num'] == 1]
        if not p1.empty:
            p1 = p1.iloc[0]
            phases_data[1] = {
                'planned_days': int(safe_get(p1.get('Engg_Planned_Days'), 90)),
                'actual_days': int(safe_get(p1.get('Engg_Actual_Days'), 90)),
                'planned_cost': float(safe_get(p1.get('Engg_Planned_Cost_Cr'), 30)),
                'actual_cost': float(safe_get(p1.get('Engg_Actual_Cost_Cr'), 30)),
                'completion': float(safe_get(p1.get('Engg_%Complete'), 100))
            }
    
    # ROW Phase (Phase 2)
    if phase_num >= 2:
        p2 = project_rows[project_rows['Phase_Num'] == 2]
        if not p2.empty:
            p2 = p2.iloc[0]
            phases_data[2] = {
                'planned_days': int(safe_get(p2.get('ROW_Planned_Days'), 150)),
                'actual_days': int(safe_get(p2.get('ROW_Actual_Days'), 150)),
                'planned_cost': float(safe_get(p2.get('ROW_Planned_Cost_Cr'), 50)),
                'actual_cost': float(safe_get(p2.get('ROW_Actual_Cost_Cr'), 50)),
                'completion': float(safe_get(p2.get('ROW_%Complete'), 100))
            }
    
    # Supply Phase (Phase 3)
    if phase_num >= 3:
        p3 = project_rows[project_rows['Phase_Num'] == 3]
        if not p3.empty:
            p3 = p3.iloc[0]
            phases_data[3] = {
                'planned_days': int(safe_get(p3.get('Supply_Planned_Days'), 75)),
                'actual_days': int(safe_get(p3.get('Supply_Actual_Days'), 75)),
                'planned_cost': float(safe_get(p3.get('Supply_Planned_Cost_Cr'), 80)),
                'actual_cost': float(safe_get(p3.get('Supply_Actual_Cost_Cr'), 80)),
                'completion': float(safe_get(p3.get('Supply_%Complete'), 100))
            }
    
    # Civil Phase (Phase 4)
    if phase_num >= 4:
        p4 = project_rows[project_rows['Phase_Num'] == 4]
        if not p4.empty:
            p4 = p4.iloc[0]
            phases_data[4] = {
                'planned_days': int(safe_get(p4.get('Civil_Planned_Days'), 100)),
                'actual_days': int(safe_get(p4.get('Civil_Actual_Days'), 100)),
                'planned_cost': float(safe_get(p4.get('Civil_Planned_Cost_Cr'), 50)),
                'actual_cost': float(safe_get(p4.get('Civil_Actual_Cost_Cr'), 50)),
                'completion': float(safe_get(p4.get('Civil_%Complete'), 100))
            }
    
    # Erection Phase (Phase 5)
    if phase_num >= 5:
        p5 = project_rows[project_rows['Phase_Num'] == 5]
        if not p5.empty:
            p5 = p5.iloc[0]
            phases_data[5] = {
                'planned_days': int(safe_get(p5.get('Erection_Planned_Days'), 120)),
                'actual_days': int(safe_get(p5.get('Erection_Actual_Days'), 120)),
                'planned_cost': float(safe_get(p5.get('Erection_Planned_Cost_Cr'), 60)),
                'actual_cost': float(safe_get(p5.get('Erection_Actual_Cost_Cr'), 60)),
                'completion': float(safe_get(p5.get('Erection_%Complete'), 100))
            }
    
    # Stringing Phase (Phase 6)
    if phase_num >= 6:
        p6 = project_rows[project_rows['Phase_Num'] == 6]
        if not p6.empty:
            p6 = p6.iloc[0]
            phases_data[6] = {
                'planned_days': int(safe_get(p6.get('Stringing_Planned_Days'), 80)),
                'actual_days': int(safe_get(p6.get('Stringing_Actual_Days'), 80)),
                'planned_cost': float(safe_get(p6.get('Stringing_Planned_Cost_Cr'), 35)),
                'actual_cost': float(safe_get(p6.get('Stringing_Actual_Cost_Cr'), 35)),
                'completion': float(safe_get(p6.get('Stringing_%Complete'), 100))
            }
    
    # Testing Phase (Phase 7)
    if phase_num >= 7:
        p7 = project_rows[project_rows['Phase_Num'] == 7]
        if not p7.empty:
            p7 = p7.iloc[0]
            phases_data[7] = {
                'planned_days': int(safe_get(p7.get('Testing_Planned_Days'), 45)),
                'actual_days': int(safe_get(p7.get('Testing_Actual_Days'), 45)),
                'planned_cost': float(safe_get(p7.get('Testing_Planned_Cost_Cr'), 20)),
                'actual_cost': float(safe_get(p7.get('Testing_Actual_Cost_Cr'), 20)),
                'completion': float(safe_get(p7.get('Testing_%Complete'), 100))
            }
    
    st.session_state['saved_phases'] = phases_data
    
    return True

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
# def predict_for_project(project_data):
#     """
#     Make predictions for a single project.
#     Returns: predicted delay (days) and cost overrun (Cr)
#     """
#     # Convert to DataFrame
#     df = pd.DataFrame([project_data])
#     
#     # Create derived features
#     if 'Project_Location' in df.columns and 'State_Encoded' not in df.columns:
#         df['State'] = df['Project_Location'].apply(
#             lambda x: str(x).split(',')[0] if pd.notna(x) else 'Unknown'
#         )
#         if 'State' in encoders['label_encoders']:
#             try:
#                 df['State_Encoded'] = encoders['label_encoders']['State'].transform(df['State'])
#             except ValueError:
#                 df['State_Encoded'] = 0
#         else:
#             df['State_Encoded'] = 0
#         df = df.drop(['State'], axis=1, errors='ignore')
#     
#     if 'Start_Date' in df.columns and 'Days_Since_Start' not in df.columns:
#         df['Start_Date'] = pd.to_datetime(df['Start_Date'], format='%d-%m-%Y', errors='coerce')
#         df['Days_Since_Start'] = (datetime.now() - df['Start_Date']).dt.days
#         df = df.drop(['Start_Date'], axis=1, errors='ignore')
#     
#     # Drop unused columns
#     df = df.drop(['Project_Location', 'DOCO_Date'], axis=1, errors='ignore')
#     
#     # Encode categorical features
#     label_encoders = encoders['label_encoders']
#     categorical_features = encoders['categorical_features']
#     
#     for col in categorical_features:
#         if col in df.columns and col in label_encoders:
#             mask = df[col].notna()
#             if mask.sum() > 0:
#                 try:
#                     df.loc[mask, col] = label_encoders[col].transform(df.loc[mask, col].astype(str))
#                 except ValueError:
#                     df.loc[mask, col] = 0
#             df[col] = df[col].astype(float)
#     
#     # Convert all to numeric
#     for col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     
#     # Extract features
#     X = df[encoders['all_features']]
#     
#     # Make predictions
#     delay_pred = delay_model.predict(X)[0]
#     cost_pred = cost_model.predict(X)[0]
#     
#     return {
#         'predicted_delay_days': round(delay_pred, 2),
#         'predicted_cost_overrun_cr': round(cost_pred, 2)
#     }
# 
# # Save feedback to CSV
def save_feedback(project_data, prediction, is_correct):
    """Save user feedback for future retraining."""
    feedback_path = 'Data/new_feedback_data.csv'
    
    # Prepare feedback row
    feedback_row = project_data.copy()
    feedback_row['Predicted_Delay'] = prediction['predicted_delay_days']
    feedback_row['Predicted_Cost'] = prediction['predicted_cost_overrun_cr']
    feedback_row['User_Feedback'] = 'Correct' if is_correct else 'Incorrect'
    feedback_row['Feedback_Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Append to feedback file
    df = pd.DataFrame([feedback_row])
    
    if os.path.exists(feedback_path) and os.path.getsize(feedback_path) > 0:
        df.to_csv(feedback_path, mode='a', header=False, index=False)
    else:
        df.to_csv(feedback_path, mode='w', header=True, index=False)

# Helper function to determine risk level
def get_risk_level(delay, cost):
    """Classify risk level based on delay and cost."""
    if delay < 10 and cost < 5:
        return "🟢 LOW", "green"
    elif delay < 30 and cost < 15:
        return "🟡 MEDIUM", "orange"
    else:
        return "🔴 HIGH", "red"

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Header
st.title("⚡ GridSight - Risk Forecasting System")
st.markdown("**Phase-Aware Cumulative Delay & Cost Overrun Prediction**")
st.markdown("---")

# Sidebar - Information
with st.sidebar:
    st.header("ℹ️ About GridSight")
    st.markdown("""
    GridSight predicts **cumulative risk** at any project milestone:
    
    - **Cumulative Delay** (days)
    - **Cumulative Cost Overrun** (₹ Cr)
    
    **How it works:**
    1. Define project metadata
    2. Select current phase
    3. Enter actual values for completed phases
    4. Get risk prediction
    
    All predictions are relative to the **original project plan**.
    """)
    
    st.markdown("---")
#     st.markdown(f"**Model Version:** {champion.get('dataset_version', 'v1')}")
#     st.markdown(f"**Last Updated:** {champion.get('created_at', 'N/A')}")

# Initialize session state for storing inputs
if 'project_data' not in st.session_state:
    st.session_state.project_data = {}
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'metadata_saved' not in st.session_state:
    st.session_state.metadata_saved = False
if 'saved_metadata' not in st.session_state:
    st.session_state.saved_metadata = {}
if 'saved_phases' not in st.session_state:
    st.session_state.saved_phases = {}
if 'phase_predictions' not in st.session_state:
    st.session_state.phase_predictions = []

# ============================================================================
# STEP 1: PROJECT METADATA
# ============================================================================

st.subheader("📋 Step 1: Project Metadata")
st.caption("Project metadata defines the baseline risk characteristics.")

# Load saved metadata if exists
if st.session_state.metadata_saved and st.session_state.saved_metadata:
    st.success("✅ Metadata saved and loaded")
    project_type = st.session_state.saved_metadata.get('project_type', "Line")
    voltage_level = st.session_state.saved_metadata.get('voltage_level', "400kV")
    contract_type = st.session_state.saved_metadata.get('contract_type', "TBCB")
    terrain_type = st.session_state.saved_metadata.get('terrain_type', "Hilly")
    project_size = st.session_state.saved_metadata.get('project_size', 100.0)
    total_budget_cr = st.session_state.saved_metadata.get('total_budget_cr', 200.0)
    planned_duration = st.session_state.saved_metadata.get('planned_duration', 24)
    project_location = st.session_state.saved_metadata.get('project_location', "Telangana, India")
    start_date = st.session_state.saved_metadata.get('start_date', datetime(2020, 1, 1))
else:
    # Default values
    project_type = "Line"
    voltage_level = "400kV"
    contract_type = "TBCB"
    terrain_type = "Hilly"
    project_size = 100.0
    total_budget_cr = 200.0
    planned_duration = 24
    project_location = "Telangana, India"
    start_date = datetime(2020, 1, 1)

col1, col2, col3 = st.columns(3)

with col1:
    project_type = st.selectbox(
        "Project Type",
        options=["Line", "Substation", "HVDC"],
        index=["Line", "Substation", "HVDC"].index(project_type),
        help="Type of transmission infrastructure project"
    )
    
    voltage_level = st.selectbox(
        "Voltage Level",
        options=["765kV", "400kV", "220kV", "132kV"],
        index=["765kV", "400kV", "220kV", "132kV"].index(voltage_level),
        help="Operating voltage of the transmission line/substation"
    )

with col2:
    contract_type = st.selectbox(
        "Contract Type",
        options=["TBCB", "EPC", "Item Rate", "Turnkey"],
        index=["TBCB", "EPC", "Item Rate", "Turnkey"].index(contract_type),
        help="Type of contract agreement"
    )
    
    terrain_type = st.selectbox(
        "Terrain Type",
        options=["Hilly", "Plain", "Forest", "Mixed"],
        index=["Hilly", "Plain", "Forest", "Mixed"].index(terrain_type),
        help="Primary terrain type along the project route"
    )

with col3:
    project_size = st.number_input(
        "Project Size (CKM)",
        min_value=0.0,
        max_value=1000.0,
        value=float(project_size),
        step=10.0,
        help="Circuit kilometers"
    )
    
    total_budget_cr = st.number_input(
        "Total Budget (₹ Cr)",
        min_value=0.0,
        max_value=10000.0,
        value=float(total_budget_cr),
        step=10.0,
        help="Total approved project budget in Crores"
    )

planned_duration = st.number_input(
    "Planned Duration (Months)",
    min_value=1,
    max_value=120,
    value=int(planned_duration),
    step=1,
    help="Original planned project duration"
)

# Project location (for state encoding)
project_location = st.text_input(
    "Project Location",
    value=str(project_location),
    help="State/Region of the project"
)

# Start date
start_date = st.date_input(
    "Project Start Date",
    value=start_date,
    help="Original project start date"
)

# Save metadata button
if st.button("💾 Save Project Metadata", type="secondary", use_container_width=True):
    st.session_state.saved_metadata = {
        'project_type': project_type,
        'voltage_level': voltage_level,
        'contract_type': contract_type,
        'terrain_type': terrain_type,
        'project_size': project_size,
        'total_budget_cr': total_budget_cr,
        'planned_duration': planned_duration,
        'project_location': project_location,
        'start_date': start_date
    }
    st.session_state.metadata_saved = True
    st.success("✅ Project metadata saved successfully!")
    st.rerun()

st.markdown("---")

# ============================================================================
# STEP 2: CURRENT PHASE SELECTION
# ============================================================================

st.subheader("🎯 Step 2: Current Project Phase")
st.caption("Select the current milestone to determine which phase inputs are required.")

# Phase definitions
phases = [
    ("1", "Planning/Engineering"),
    ("2", "Approvals & ROW"),
    ("3", "Procurement/Supply"),
    ("4", "Site Preparation/Civil"),
    ("5", "Erection"),
    ("6", "Stringing"),
    ("7", "Testing/Commissioning")
]

phase_options = [f"Phase {p[0]}: {p[1]}" for p in phases]

current_phase_display = st.radio(
    "Current Phase",
    options=phase_options,
    horizontal=True,
    help="Select the current project phase"
)

# Extract phase number
current_phase = int(current_phase_display.split(":")[0].split()[1])

# Progress indicator
progress_pct = (current_phase / 7) * 100
st.progress(progress_pct / 100)
st.caption(f"Progress: Phase {current_phase} of 7 ({progress_pct:.0f}%)")

st.markdown("---")

# ============================================================================
# STEP 3: PHASE-WISE INPUT (DYNAMIC)
# ============================================================================

st.subheader("📊 Step 3: Phase-Wise Actuals")
st.caption("Enter values only for phases that are completed.")

# Phase input mapping
phase_prefixes = {
    1: "Engg",
    2: "ROW",
    3: "Supply",
    4: "Civil",
    5: "Erection",
    6: "Stringing",
    7: "Testing"
}

phase_data = {}

# Show inputs only for completed phases
for phase_num in range(1, current_phase + 1):
    prefix = phase_prefixes.get(phase_num, "")
    phase_name = phases[phase_num - 1][1]
    
    # Load saved phase data if exists
    if phase_num in st.session_state.saved_phases:
        saved_data = st.session_state.saved_phases[phase_num]
    else:
        saved_data = {
            'planned_days': 90,
            'actual_days': 100,
            'planned_cost': 25.0,
            'actual_cost': 30.0,
            'completion': 100 if phase_num < current_phase else 50
        }
    
    with st.expander(f"📌 Phase {phase_num}: {phase_name}", expanded=(phase_num == current_phase)):
        # Show saved indicator
        if phase_num in st.session_state.saved_phases:
            st.success(f"✅ Phase {phase_num} data saved")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Planned Values**")
            planned_days = st.number_input(
                f"Planned Days (Phase {phase_num})",
                min_value=0,
                max_value=365,
                value=int(saved_data['planned_days']),
                step=5,
                key=f"planned_days_{phase_num}"
            )
            
            planned_cost = st.number_input(
                f"Planned Cost - ₹ Cr (Phase {phase_num})",
                min_value=0.0,
                max_value=3000.0,
                value=float(saved_data['planned_cost']),
                step=5.0,
                key=f"planned_cost_{phase_num}"
            )
        
        with col2:
            st.markdown("**Actual Values**")
            actual_days = st.number_input(
                f"Actual Days (Phase {phase_num})",
                min_value=0,
                max_value=500,
                value=int(saved_data['actual_days']),
                step=5,
                key=f"actual_days_{phase_num}"
            )
            
            actual_cost = st.number_input(
                f"Actual Cost - ₹ Cr (Phase {phase_num})",
                min_value=0.0,
                max_value=3000.0,
                value=float(saved_data['actual_cost']),
                step=5.0,
                key=f"actual_cost_{phase_num}"
            )
        
        # Completion percentage
        completion = st.slider(
            f"Completion % (Phase {phase_num})",
            min_value=0,
            max_value=100,
            value=int(saved_data['completion']),
            step=5,
            key=f"completion_{phase_num}"
        )
        
        # Save button for this phase
        if st.button(f"💾 Save Phase {phase_num} Data", key=f"save_phase_{phase_num}", use_container_width=True):
            st.session_state.saved_phases[phase_num] = {
                'planned_days': planned_days,
                'actual_days': actual_days,
                'planned_cost': planned_cost,
                'actual_cost': actual_cost,
                'completion': completion
            }
            st.success(f"✅ Phase {phase_num} data saved!")
            st.rerun()
        
        # Store phase data for prediction
        phase_data[phase_num] = {
            'planned_days': planned_days,
            'actual_days': actual_days,
            'planned_cost': planned_cost,
            'actual_cost': actual_cost,
            'completion': completion
        }

st.markdown("---")

# ============================================================================
# AUTOFILL SECTION
# ============================================================================

st.subheader("🔄 Quick Test with Sample Data")

col_autofill1, col_autofill2 = st.columns([3, 1])

with col_autofill1:
    use_autofill = st.toggle(
        f"Autofill data till Phase {current_phase} from test_data.csv",
        help="Load random project data from test_data.csv matching your selected phase"
    )

with col_autofill2:
    if use_autofill:
        if st.button("🎲 Load Random Data", type="secondary", use_container_width=True):
            with st.spinner(f"Loading random project data for Phase {current_phase}..."):
                if autofill_from_test_data(current_phase):
                    st.success(f"✅ Autofilled data from test_data.csv for Phase {current_phase}")
                    st.rerun()
                else:
                    st.error("❌ Failed to load test data")

st.markdown("---")

# ============================================================================
# STEP 4: PREDICT BUTTON
# ============================================================================

st.subheader("🔮 Step 4: Generate Prediction")

# Assemble input data
def assemble_project_data():
    """Assemble all inputs into the format expected by the model."""
    data = {
        'Project_Type': project_type,
        'Voltage_Level': voltage_level,
        'Contract_Type': contract_type,
        'Terrain_Type': terrain_type,
        'Project_Size': project_size,
        'Total_Budget_Cr': total_budget_cr,
        'Planned_Duration': planned_duration,
        'Project_Location': project_location,
        'Start_Date': start_date.strftime('%d-%m-%Y'),
        'Phase_Num': current_phase,
        'Phase_Name': phases[current_phase - 1][1],
        'DOCO_Date': None,
        'Unnamed: 62': None
    }
    
    # Add phase-specific data
    for phase_num, prefix in phase_prefixes.items():
        if phase_num <= current_phase:
            # Completed or current phase
            pd = phase_data.get(phase_num, {})
            data[f'{prefix}_%Complete'] = pd.get('completion', 100)
            data[f'{prefix}_Planned_Days'] = pd.get('planned_days', 0)
            data[f'{prefix}_Actual_Days'] = pd.get('actual_days', 0)
            data[f'{prefix}_Planned_Cost_Cr'] = pd.get('planned_cost', 0)
            data[f'{prefix}_Actual_Cost_Cr'] = pd.get('actual_cost', 0)
            
            # Phase-specific fields (set defaults)
            if prefix == 'ROW':
                data['Forest_Clearance'] = 1
                data['Statutory_Approvals'] = 'Complete'
            elif prefix == 'Engg':
                data['DPR_Status'] = 'Approved'
                data['Design_Changes'] = 0
            elif prefix == 'Supply':
                data['Orders_Placed'] = 1
                data['Equip_Delivery_Status'] = 'OnTime'
            elif prefix == 'Civil':
                data['Foundations_Complete'] = 1 if pd.get('completion', 0) == 100 else 0
                data['Site_Access'] = 1
            elif prefix == 'Erection':
                data['Towers_Erected_Pct'] = pd.get('completion', 0)
                data['Structures_Complete'] = 1 if pd.get('completion', 0) == 100 else 0
            elif prefix == 'Stringing':
                data['Conductor_Strung_Pct'] = pd.get('completion', 0)
            elif prefix == 'Testing':
                data['Trial_Run_Status'] = 'Complete' if pd.get('completion', 0) == 100 else 'Pending'
        else:
            # Future phase - set to None (NaN)
            data[f'{prefix}_%Complete'] = None
            data[f'{prefix}_Planned_Days'] = None
            data[f'{prefix}_Actual_Days'] = None
            data[f'{prefix}_Planned_Cost_Cr'] = None
            data[f'{prefix}_Actual_Cost_Cr'] = None
            
            # Phase-specific fields
            if prefix == 'ROW':
                data['Forest_Clearance'] = None
                data['Statutory_Approvals'] = None
            elif prefix == 'Engg':
                data['DPR_Status'] = None
                data['Design_Changes'] = None
            elif prefix == 'Supply':
                data['Orders_Placed'] = None
                data['Equip_Delivery_Status'] = None
            elif prefix == 'Civil':
                data['Foundations_Complete'] = None
                data['Site_Access'] = None
            elif prefix == 'Erection':
                data['Towers_Erected_Pct'] = None
                data['Structures_Complete'] = None
            elif prefix == 'Stringing':
                data['Conductor_Strung_Pct'] = None
            elif prefix == 'Testing':
                data['Trial_Run_Status'] = None
    
    # Dummy values for cumulative targets (not used in prediction, but needed for CSV structure)
    data['Cumulative_Delay_Days'] = 0
    data['Cumulative_Cost_Overrun_Cr'] = 0
    data['Project_ID'] = f"USER_INPUT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return data

# Predict button
if st.button("🎯 Predict Cumulative Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing project data..."):
        try:
            # Check if metadata is saved
            if not st.session_state.metadata_saved:
                st.warning("⚠️ Please save project metadata first!")
            else:
                # Use saved metadata
                project_type = st.session_state.saved_metadata['project_type']
                voltage_level = st.session_state.saved_metadata['voltage_level']
                contract_type = st.session_state.saved_metadata['contract_type']
                terrain_type = st.session_state.saved_metadata['terrain_type']
                project_size = st.session_state.saved_metadata['project_size']
                total_budget_cr = st.session_state.saved_metadata['total_budget_cr']
                planned_duration = st.session_state.saved_metadata['planned_duration']
                project_location = st.session_state.saved_metadata['project_location']
                start_date = st.session_state.saved_metadata['start_date']
                
                # Use saved phase data
                phase_data = {}
                for phase_num in range(1, current_phase + 1):
                    if phase_num in st.session_state.saved_phases:
                        phase_data[phase_num] = st.session_state.saved_phases[phase_num]
                    else:
                        st.warning(f"⚠️ Phase {phase_num} data not saved. Please save it first!")
                        phase_data = None
                        break
                
                if phase_data:
                    # Assemble data
                    project_input = assemble_project_data()
                    st.session_state.project_data = project_input
                    
                    # Make prediction
                    API_URL = os.getenv(
                            "PREDICTION_API_URL",
                            "http://gridsight-predictor.kserve.svc.cluster.local/v1/models/gridsight:predict"
                            )
                    response = requests.post(
                            API_URL,
                            json={
                                "instances": [project_input]
                                },
                            timeout=30
                          )
                    response.raise_for_status()
                    prediction = response.json()["predictions"][0]

                    st.session_state.prediction_result = prediction
                    st.session_state.prediction_made = True
                    
                    # Calculate cumulative planned values
                    total_planned_days = sum([phase_data[p]['planned_days'] for p in range(1, current_phase + 1)])
                    total_planned_cost = sum([phase_data[p]['planned_cost'] for p in range(1, current_phase + 1)])
                    
                    # Store phase-wise prediction
                    phase_prediction = {
                        'phase': current_phase,
                        'phase_name': phases[current_phase - 1][1],
                        'planned_days': total_planned_days,
                        'predicted_delay': prediction['predicted_delay_days'],
                        'actual_timeline': total_planned_days + prediction['predicted_delay_days'],
                        'planned_cost': total_planned_cost,
                        'predicted_overrun': prediction['predicted_cost_overrun_cr'],
                        'predicted_total_cost': total_planned_cost + prediction['predicted_cost_overrun_cr']
                    }
                    
                    # Update or append phase prediction
                    existing_idx = next((i for i, p in enumerate(st.session_state.phase_predictions) 
                                        if p['phase'] == current_phase), None)
                    if existing_idx is not None:
                        st.session_state.phase_predictions[existing_idx] = phase_prediction
                    else:
                        st.session_state.phase_predictions.append(phase_prediction)
                    
                    st.success("✅ Prediction completed successfully!")
        
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.session_state.prediction_made = False

st.markdown("---")

# ============================================================================
# STEP 5: OUTPUT INTERPRETATION
# ============================================================================

if st.session_state.prediction_made and st.session_state.prediction_result:
    st.subheader("📈 Step 5: Risk Assessment Results")
    
    prediction = st.session_state.prediction_result
    
    # Display predictions in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📅 Estimated Cumulative Delay",
            value=f"{prediction['predicted_delay_days']:.2f} days",
            help="Total delay relative to original plan"
        )
    
    with col2:
        st.metric(
            label="💰 Estimated Cumulative Cost Overrun",
            value=f"₹ {prediction['predicted_cost_overrun_cr']:.2f} Cr",
            help="Total cost overrun relative to original budget"
        )
    
    with col3:
        risk_label, risk_color = get_risk_level(
            prediction['predicted_delay_days'],
            prediction['predicted_cost_overrun_cr']
        )
        st.metric(
            label="⚠️ Overall Risk Level",
            value=risk_label
        )
    
    # Phase-wise tracking table
    if st.session_state.phase_predictions:
        st.markdown("---")
        st.subheader("📊 Phase-Wise Tracking")
        st.caption("Historical predictions and progress tracking across phases")
        
        # Create DataFrame for display
        tracking_df = pd.DataFrame(st.session_state.phase_predictions)
        
        # Format display
        display_df = tracking_df.copy()
        display_df['Phase'] = display_df['phase'].astype(str) + " - " + display_df['phase_name']
        display_df['Planned Timeline (days)'] = display_df['planned_days'].round(0).astype(int)
        display_df['Predicted Delay (days)'] = display_df['predicted_delay'].round(2)
        display_df['Actual Timeline (days)'] = display_df['actual_timeline'].round(2)
        display_df['Planned Cost (₹ Cr)'] = display_df['planned_cost'].round(2)
        display_df['Predicted Overrun (₹ Cr)'] = display_df['predicted_overrun'].round(2)
        display_df['Predicted Total Cost (₹ Cr)'] = display_df['predicted_total_cost'].round(2)
        
        # Select columns to display
        display_cols = ['Phase', 'Planned Timeline (days)', 'Predicted Delay (days)', 
                       'Actual Timeline (days)', 'Planned Cost (₹ Cr)', 
                       'Predicted Overrun (₹ Cr)', 'Predicted Total Cost (₹ Cr)']
        
        st.dataframe(
            display_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("💡 **Note**: Values update as you progress through phases. Each row shows cumulative prediction at that milestone.")
    
    # Explanation
    st.info("""
    **📌 What do these values mean?**
    
    These values represent the **estimated deviation from the original project plan** as of the selected phase.
    
    - **Cumulative Delay**: Total additional days beyond the planned schedule
    - **Cumulative Cost Overrun**: Total additional cost beyond the approved budget
    - **Risk Level**: Overall project risk classification based on delay and cost
    
    These are **risk estimates**, not arithmetic calculations. They consider:
    - Project DNA (type, size, terrain, contract)
    - Historical phase performance
    - Sequential phase dependencies
    - Forward-fill memory of past delays
    """)
    
    st.markdown("---")
    
    # ============================================================================
    # STEP 6: USER FEEDBACK
    # ============================================================================
    
    st.subheader("💬 Step 6: User Feedback")
    st.caption("Help improve the model by validating predictions against actual outcomes.")
    
    st.markdown("**Was this prediction reasonable based on actual project outcome?**")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("✅ Prediction was Correct", use_container_width=True):
            save_feedback(
                st.session_state.project_data,
                st.session_state.prediction_result,
                is_correct=True
            )
            st.success("✅ Feedback saved! Thank you for helping improve the model.")
            st.balloons()
    
    with col2:
        if st.button("❌ Prediction was Incorrect", use_container_width=True):
            save_feedback(
                st.session_state.project_data,
                st.session_state.prediction_result,
                is_correct=False
            )
            st.warning("⚠️ Feedback noted. This will be reviewed during retraining.")
    
    with col3:
        st.caption("Your feedback is saved to `Data/new_feedback_data.csv` and will be used for future model retraining.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>GridSight - Phase-Aware Risk Forecasting System | Powered by XGBoost</p>
    <p>This is an inference-only interface. Model retraining is performed separately.</p>
</div>
""", unsafe_allow_html=True)
