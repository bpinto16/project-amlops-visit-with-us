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
        repo_id="bpinto16/wellness-tourism-mlflow-model",
        filename="best_wellness_mlflow_model.joblib",
    )
    return joblib.load(model_path)

model = load_model()

st.title("Wellness Tourism Package — Purchase Predictor")
st.write(
    """
    Predict whether a customer is likely to purchase the **Wellness Tourism Package**
    before a sales call is made. Fill in the details below and click **Predict**.
    """
)
st.divider()

# Customer demographics
st.subheader("Customer demographics")
c1, c2, c3, c4 = st.columns(4)

with c1:
    age = st.number_input(
        "Age", min_value=18, max_value=61, value=36, step=1,
        help="Enter a number between 18 and 61"
    )

with c2:
    gender = st.selectbox("Gender", options=["Female", "Male"])

with c3:
    marital_status = st.selectbox(
        "Marital status", options=["Divorced", "Married", "Single"]
    )

with c4:
    occupation = st.selectbox(
        "Occupation", options=["Large Business", "Salaried", "Small Business"]
    )

c5, c6, c7, c8 = st.columns(4)

with c5:
    designation = st.selectbox(
        "Designation",
        options=["Executive", "Manager", "Senior Manager", "AVP", "VP"],
        help="Customer's role in their organisation"
    )

with c6:
    monthly_income = st.number_input(
        "Monthly income (₹)", min_value=1000, max_value=98678,
        value=22400, step=500,
        help="Gross monthly income in INR"
    )

with c7:
    city_tier = st.selectbox(
        "City tier", options=[1, 2, 3],
        help="1 = metro (best), 3 = lowest"
    )

with c8:
    type_of_contact = st.selectbox(
        "Type of contact",
        options=["Company Invited", "Self Enquiry"],
        help="How the customer was contacted"
    )

st.divider()

# Travel profile
st.subheader("Travel profile")
t1, t2, t3, t4 = st.columns(4)

with t1:
    number_of_trips = st.number_input(
        "Avg trips per year", min_value=1, max_value=7, value=3, step=1,
        help="Enter a number between 1 and 7 (capped at 95th percentile)"
    )

with t2:
    preferred_property_star = st.selectbox(
        "Preferred hotel stars", options=[3, 4, 5]
    )

with t3:
    product_pitched = st.selectbox(
        "Product pitched",
        options=["Basic", "Deluxe", "King", "Standard", "Super Deluxe"],
        help="Package pitched to this customer"
    )

with t4:
    passport = st.selectbox(
        "Holds passport?", options=["Yes", "No"],
        help="Whether the customer has a valid passport"
    )

t5, t6, t7, t8 = st.columns(4)

with t5:
    number_of_person_visiting = st.selectbox(
        "Persons visiting", options=[1, 2, 3, 4, 5],
        help="Total people including the customer"
    )

with t6:
    number_of_children_visiting = st.selectbox(
        "Children visiting (under 5)", options=[0, 1, 2, 3]
    )

with t7:
    own_car = st.selectbox("Owns a car?", options=["Yes", "No"])

with t8:
    st.empty()  

st.divider()

# Sales interaction
st.subheader("Sales interaction")
s1, s2, s3 = st.columns(3)

with s1:
    duration_of_pitch = st.slider(
        "Duration of pitch (minutes)", min_value=5, max_value=127, value=15
    )

with s2:
    number_of_followups = st.number_input(
        "Number of follow-ups", min_value=1, max_value=6, value=3, step=1,
        help="Enter a number between 1 and 6"
    )

with s3:
    pitch_satisfaction_score = st.slider(
        "Pitch satisfaction score", min_value=1, max_value=5, value=3,
        help="1 = very dissatisfied, 5 = very satisfied"
    )

st.divider()

# Prediction 
THRESHOLD = 0.40

if st.button("Predict Purchase Likelihood", use_container_width=True):

    raw_input = pd.DataFrame([{
        "Age"                      : int(age),
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
        "NumberOfTrips"            : float(min(number_of_trips, 7)),
        "Passport"                 : 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore"   : pitch_satisfaction_score,
        "OwnCar"                   : 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting" : float(number_of_children_visiting),
        "Designation"              : designation,
        "MonthlyIncome_log"        : np.log1p(monthly_income),
    }])

    purchase_probability = model.predict_proba(raw_input)[0, 1]
    will_purchase = purchase_probability >= THRESHOLD

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
        st.caption(f"Decision threshold: {THRESHOLD:.0%}")

    st.divider()
    st.subheader("Recommended action")

    if will_purchase:
        st.markdown(
            """
            **Prioritise for a direct sales call.**
            - Confirm travel dates and group size
            - Highlight wellness benefits aligned with their profile
            - Follow up within 48 hours for best conversion
            """
        )
    else:
        st.markdown(
            """
            **Add to nurture campaign. Do not call yet.**
            - Include in email / WhatsApp wellness content series
            - Re-evaluate after 2 or 3 additional touchpoints
            - Consider pitching a lower tier product first (eg: Basic package)
            """
        )
