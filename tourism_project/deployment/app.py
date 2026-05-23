import numpy as np
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
 
st.set_page_config(
    page_title="Wellness Tourism Package — Purchase Predictor",
    layout="wide",
)
 

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="bpinto16/wellness-tourism-model",
        filename="best_wellness_model.joblib",
    )
    return joblib.load(model_path)

model = load_model()
 

st.title("Wellness Tourism Package — Purchase Predictor")
st.write(
    """
    This tool predicts whether a customer is likely to purchase the
    **Wellness Tourism Package** before a sales call is made.
    Fill in the customer's details below and click **Predict**.
    """
)
st.divider()
 

# INPUT FORM
st.subheader("Customer details")
col1, col2, col3 = st.columns(3)
 
with col1:
    age = st.slider(
        "Age", min_value=18, max_value=61, value=36,
        help="Customer age (dataset range: 18–61)"
    )
    gender = st.selectbox(
        "Gender", options=["Female", "Male"]
    )
    marital_status = st.selectbox(
        "Marital status", options=["Divorced", "Married", "Single"]
    )
    occupation = st.selectbox(
        "Occupation", options=["Large Business", "Salaried", "Small Business"]
    )
 
with col2:
    designation = st.selectbox(
        "Designation",
        options=["Executive", "Manager", "Senior Manager", "AVP", "VP"],
        help="Customer's role in their organisation"
    )
    monthly_income = st.number_input(
        "Monthly income (₹)", min_value=1000, max_value=98678,
        value=22400, step=500,
        help="Gross monthly income in INR"
    )
    city_tier = st.selectbox(
        "City tier",
        options=[1, 2, 3],
        help="1 = best infrastructure (metro), 3 = lowest"
    )
    type_of_contact = st.selectbox(
        "Type of contact",
        options=["Company Invited", "Self Enquiry"],
        help="How the customer was contacted"
    )
 
with col3:
    passport = st.selectbox(
        "Holds passport?", options=["Yes", "No"],
        help="Whether the customer has a valid passport"
    )
    own_car = st.selectbox(
        "Owns a car?", options=["Yes", "No"]
    )
    number_of_person_visiting = st.selectbox(
        "Number of persons visiting",
        options=[1, 2, 3, 4, 5],
        help="Total people accompanying the customer"
    )
    number_of_children_visiting = st.selectbox(
        "Children visiting (under 5)",
        options=[0, 1, 2, 3]
    )
 
st.divider()
st.subheader("Travel profile")
col4, col5 = st.columns(2)
 
with col4:
    number_of_trips = st.slider(
        "Average trips per year", min_value=1, max_value=7, value=3,
        help="Capped at 7 (95th percentile) during training"
    )
    preferred_property_star = st.selectbox(
        "Preferred hotel star rating", options=[3, 4, 5]
    )
 
with col5:
    product_pitched = st.selectbox(
        "Product pitched",
        options=["Basic", "Deluxe", "King", "Standard", "Super Deluxe"],
        help="The tourism package pitched to this customer"
    )
 
st.divider()
st.subheader("Sales interaction")
col6, col7 = st.columns(2)
 
with col6:
    duration_of_pitch = st.slider(
        "Duration of pitch (minutes)", min_value=5, max_value=127, value=15
    )
    number_of_followups = st.selectbox(
        "Number of follow-ups", options=[1, 2, 3, 4, 5, 6]
    )
 
with col7:
    pitch_satisfaction_score = st.selectbox(
        "Pitch satisfaction score (1=low, 5=high)", options=[1, 2, 3, 4, 5]
    )
 
st.divider()
 

# PREDICTION
THRESHOLD = 0.40   # matches classification threshold used in train.py
 
if st.button("Predict Purchase Likelihood", use_container_width=True):
 
    #  Build raw input DataFrame
    raw_input = pd.DataFrame([{
        "Age"                      : age,
        "TypeofContact"            : type_of_contact,
        "CityTier"                 : city_tier,
        "DurationOfPitch"          : float(duration_of_pitch),
        "Occupation"               : occupation,
        "Gender"                   : gender,
        "NumberOfPersonVisiting"   : number_of_person_visiting,
        "NumberOfFollowups"        : float(number_of_followups),
        "ProductPitched"           : product_pitched,
        "PreferredPropertyStar"    : float(preferred_property_star),
        "MaritalStatus"            : marital_status,
        "NumberOfTrips"            : float(min(number_of_trips, 7)),  # enforce training cap
        "Passport"                 : 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore"   : pitch_satisfaction_score,
        "OwnCar"                   : 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting" : float(number_of_children_visiting),
        "Designation"              : designation,
        "MonthlyIncome_log"        : np.log1p(monthly_income),  # match preparation.py transform
    }])
 
    purchase_probability = model.predict_proba(raw_input)[0, 1]
    will_purchase = purchase_probability >= THRESHOLD
 
    # Display result
    st.subheader("Prediction result")
 
    result_col1, result_col2 = st.columns(2)
 
    with result_col1:
        if will_purchase:
            st.success(
                f"**Likely to purchase**\n\n"
                f"This customer has a **{purchase_probability:.1%}** probability "
                f"of buying the Wellness Tourism Package."
            )
        else:
            st.warning(
                f"**Unlikely to purchase**\n\n"
                f"This customer has a **{purchase_probability:.1%}** probability "
                f"of buying the Wellness Tourism Package."
            )
 
    with result_col2:
        st.metric(
            label="Purchase probability",
            value=f"{purchase_probability:.1%}",
            delta=f"{purchase_probability - THRESHOLD:+.1%} vs threshold ({THRESHOLD:.0%})",
        )
        st.caption(
            f"Decision threshold: {THRESHOLD:.0%}"
        )
 
    # Interpretation guide 
    st.divider()
    st.subheader("How to use this result")
 
    if will_purchase:
        st.markdown(
            """
            **Recommended action:** Prioritise this customer for a direct sales call.
            - Confirm travel dates and group size
            - Highlight wellness benefits aligned with their profile
            - Follow up within 48 hours for best conversion
            """
        )
    else:
        st.markdown(
            """
            **Recommended action:** Add to nurture campaign rather than direct call.
            - Include in email / WhatsApp wellness content series
            - Re-evaluate after 2–3 additional touchpoints
            - Consider pitching a lower-tier product first (e.g. Basic package)
            """
        )
