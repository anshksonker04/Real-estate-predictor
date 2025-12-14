import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
# We use @st.cache_resource so the model loads only once, making the app faster
@st.cache_resource
def load_model():
    return joblib.load('real_estate_pipeline_v1.pkl')

model = load_model()

# 2. App Title and Description
st.title("🏡 Real Estate Price Predictor")
st.write("Enter the property details below to get an estimated sale price.")

# 3. Create Input Fields for the User
# We divide the layout into two columns for a better look
col1, col2 = st.columns(2)

with col1:
    list_year = st.number_input("Listing Year", min_value=2000, max_value=2030, value=2021, step=1)
    assessed_value = st.number_input("Assessed Value ($)", min_value=1000, value=300000, step=1000)
    town = st.text_input("Town Name", value="Stamford")

with col2:
    # Note: These options should match what your model saw during training
    prop_type = st.selectbox("Property Type", ["Residential", "Commercial", "Vacant Land", "Apartments"])
    res_type = st.selectbox("Residential Type", ["Single Family", "Condo", "Two Family", "Three Family"])

# 4. Prediction Logic
if st.button("Predict Price"):
    # Create a dataframe from inputs (must match the training columns exactly!)
    input_data = pd.DataFrame({
        'list_year': [list_year],
        'town': [town],
        'assessed_value': [assessed_value],
        'property_type': [prop_type],
        'residential_type': [res_type]
    })

    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Sale Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

# 5. Optional: Show the input data for debugging
if st.checkbox("Show Input Data"):
    st.write(input_data)