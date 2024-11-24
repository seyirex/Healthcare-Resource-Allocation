import streamlit as st
import pandas as pd
from ortools.linear_solver import pywraplp
from pycaret.regression import load_model, predict_model
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth 
from streamlit_authenticator.utilities import LoginError
import yaml
from yaml.loader import SafeLoader


st.set_page_config(page_title="Healthcare Resource Allocation Tool", page_icon="🧑‍⚕️")

# Load the configuration file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Load the pre-trained model
model = load_model("./model/modelCCP_11-23-2024")

# Initialize the authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Creating a login widget
try:
    authenticator.login()
except LoginError as e:
    st.error(e)


def predict(model, input_df):
    """
    Predict the number of visits using the pre-trained model.

    Args:
        model: Trained PyCaret model.
        input_df (pd.DataFrame): Input data for prediction.

    Returns:
        float: Predicted number of visits.
    """
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df["prediction_label"].iloc[0]
    return predictions


def run():
    """
    Main function to run the Streamlit app.
    """

    if st.session_state["authentication_status"]:
        # Welcome message
        st.sidebar.success(f"Welcome, {st.session_state['name']}!")

        # Logout button
        authenticator.logout("Logout", "sidebar")
            # App title
        st.title("🏥 Healthcare Resource Allocation and Optimization")
        
        # Description
        st.markdown("""
        This application predicts patient visits and optimizes healthcare resource allocation across regions. 
        You can input details manually or upload a file for batch predictions. 
        The app also provides an optimized resource allocation plan based on predicted demand.
        """)

        # Create tabs for navigation
        tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])

        # Tab 1: Single Prediction
        with tab1:
            with st.form("prediction_form"):
                st.header("Enter Patient and Regional Details")
                st.markdown("Use this form to input details for a single prediction.")

                # Row 1: Visit Type and Facility Type
                col1, col2 = st.columns(2)
                with col1:
                    visit_type = st.selectbox(
                        "Visit Type",
                        ["out-patient", "in-patient"],
                        help="Specify if the patient visit is in-patient or out-patient.",
                    )
                with col2:
                    facility_type = st.selectbox(
                        "Facility Type",
                        ["hospital", "clinic", "laboratory"],
                        help="Type of healthcare facility.",
                    )

                # Row 2: Sex and Region
                col3, col4 = st.columns(2)
                with col3:
                    sex = st.selectbox("Sex", ["male", "female"], help="Patient's gender.")
                with col4:
                    region = st.selectbox(
                        "Region",
                        [
                            "north-central",
                            "north-east",
                            "north-west",
                            "south-east",
                            "south-south",
                            "south-west",
                        ],
                        help="Region where the healthcare facility is located.",
                    )

                # Row 3: Age and Length of Stay
                col5, col6 = st.columns(2)
                with col5:
                    age = st.slider("Age", 0, 100, 30, help="Patient's age.")
                with col6:
                    length_of_stay = st.slider(
                        "Length of Stay (days)", 0, 100, 0, help="Length of stay in days."
                    )

                # Row 4: Age Group and Region Count
                col7, col8 = st.columns(2)
                with col7:
                    age_group = st.selectbox(
                        "Age Group",
                        ["child", "young adult", "adult", "senior"],
                        help="Age category based on the patient's age.",
                    )
                with col8:
                    region_count = st.slider(
                        "Region Count",
                        1,
                        100,
                        1,
                        help="The number of visits made by the patient in the selected region.",
                    )

                # Row 5: Beds Available and Staff Available
                st.header("Regional Resource Details")
                col9, col10 = st.columns(2)
                with col9:
                    beds_available = st.number_input(
                        "Beds Available",
                        0,
                        10000,
                        200,
                        help="Number of beds available in the region. Must be a non-negative value.",
                    )
                with col10:
                    staff_available = st.number_input(
                        "Staff Available",
                        0,
                        1000,
                        50,
                        help="Number of healthcare staff available in the region. Must be a non-negative value.",
                    )

                # Submit button
                submitted = st.form_submit_button("Predict and Optimize")

            if submitted:
                if beds_available < 0 or staff_available < 0:
                    st.error(
                        "Invalid input: Beds and staff availability must be non-negative."
                    )
                else:
                    # Prediction step
                    input_data = pd.DataFrame(
                        {
                            "visit_type": [visit_type],
                            "facility_type": [facility_type],
                            "sex": [sex],
                            "region": [region],
                            "age": [age],
                            "length_of_stay": [length_of_stay],
                            "age_group": [age_group],
                            "sex_binary": [1 if sex == "female" else 0],
                            "region_count": [region_count],
                        }
                    )

                    predicted_visits = predict(model, input_data)
                    st.success(f"Predicted Total Visits: {predicted_visits:.2f}")

                    # Optimization
                    solver = pywraplp.Solver.CreateSolver("GLOP")
                    x = solver.NumVar(0, beds_available, "x")  # Beds allocated
                    unmet_demand = solver.NumVar(0, solver.infinity(), "unmet_demand")

                    # Objective: Minimize unmet demand
                    solver.Objective().SetCoefficient(unmet_demand, 1)
                    solver.Objective().SetMinimization()

                    # Constraints
                    solver.Add(x <= beds_available)
                    solver.Add(x <= staff_available * 5)
                    solver.Add(unmet_demand >= predicted_visits - x)

                    # Solve
                    status = solver.Solve()
                    if status == pywraplp.Solver.OPTIMAL:
                        st.header("Optimization Results")
                        st.write(f"Beds Allocated: {x.solution_value()}")
                        st.write(f"Unmet Demand: {unmet_demand.solution_value()}")

                        # Visualization
                        st.header("Resource Allocation")
                        fig, ax = plt.subplots()
                        regions = ["Allocated Beds", "Unmet Demand"]
                        values = [x.solution_value(), unmet_demand.solution_value()]
                        ax.bar(regions, values, color=["green", "red"])
                        ax.set_ylabel("Value")
                        ax.set_title("Resource Allocation Overview")
                        st.pyplot(fig)
                    else:
                        st.error("No optimal solution found.")

        # Tab 2: Batch Prediction
        with tab2:
            st.header("Batch Prediction")
            st.markdown(
                "Upload a CSV file containing patient and regional details for batch predictions."
            )

            # File upload
            uploaded_file = st.file_uploader(
                "Choose a file",
                type="csv",
                help="Upload a CSV file with the required columns.",
            )

            if uploaded_file:
                batch_data = pd.read_csv(uploaded_file)
                st.write("Uploaded Data:", batch_data.head())

                # Check required columns
                required_columns = [
                    "visit_type",
                    "facility_type",
                    "sex",
                    "region",
                    "age",
                    "length_of_stay",
                    "age_group",
                    "sex_binary",
                    "region_count",
                ]
                if all(col in batch_data.columns for col in required_columns):
                    # Predict for all rows
                    batch_data["predicted_visits"] = batch_data.apply(
                        lambda row: predict(model, pd.DataFrame([row])), axis=1
                    )

                    st.header("Batch Predictions")
                    st.write(batch_data)

                    # Visualization
                    st.header("Predicted Visits by Region")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    batch_data.groupby("region")["predicted_visits"].sum().plot(
                        kind="bar", ax=ax, color="skyblue"
                    )
                    ax.set_ylabel("Predicted Visits")
                    ax.set_title("Predicted Visits by Region")
                    st.pyplot(fig)

                    # Allow download of results
                    st.download_button(
                        label="Download Predictions",
                        data=batch_data.to_csv(index=False),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(
                        f"Uploaded file must contain the following columns: {', '.join(required_columns)}"
                    )

        
    elif st.session_state["authentication_status"] is False:
        st.error("Invalid username or password")
        
    elif st.session_state["authentication_status"] is None:
        
        st.warning("Please enter your username and password")


if __name__ == "__main__":
    run()
