import streamlit as st
import pandas as pd
import joblib
import time


st.title("Welcome")


model_xgb = joblib.load("final_xgb_model.joblib")
pipeline = joblib.load("pipeline.joblib")
necessary_col = joblib.load("necessary_columns.joblib")
star = ["---", 0, 1, 2, 3, 4, 5]




age = st.text_input("Age: ", 0)
age = int(age)

# --------------------------------------------------------------

flight_distance = st.text_input("Flight Distance: ", 0)
flight_distance = int(flight_distance)

# ---------------------------------------------------------------

gender = st.selectbox(("Add Gender: "),
                      ("---", "Male", "Female"))

if gender == "--":
    gender = str(" ")

# ----------------------------------------------------------------

customer_type = st.selectbox(("Add Customer Type: "),
                      ("---", "Loyal Customer", "disloyal Customer"))

if customer_type == "---":
    customer_type = str(" ")

# -------------------------------------------------------------------

travel_type = st.selectbox(("Add Travel Type: "),
                      ("---", "Personal Travel", "Business Travel"))

if travel_type == "---":
    travel_type = str(" ")

# -------------------------------------------------------------------

travel_class = st.selectbox(("Add Travel Class: "),
                      ("---", "Eco", "Eco Plus", "Business"))

if travel_class == "---":
    travel_class = str(" ")

# --------------------------------------------------------------------

wifi_service = st.selectbox(("Add Inflight wifi service: "),
                            star)

if wifi_service == "---":
    wifi_service = str(" ")

# ----------------------------------------------------------------------

departure_arrival_time = st.selectbox(("Add Departure/Arrival time convenient: "),
                            star)

if departure_arrival_time == "---":
    departure_arrival_time = str(" ")

# ----------------------------------------------------------------------

gate_location = st.selectbox(("Add Gate location: "),
                            star)

if gate_location == "---":
    gate_location = str(" ")

# ----------------------------------------------------------------------

food_drink = st.selectbox(("Add Food and drink: "),
                            star)

if food_drink == "---":
    food_drink = str(" ")

# ----------------------------------------------------------------------

online_boarding	 = st.selectbox(("Add Online boarding: "),
                                star)

if online_boarding == "---":
    online_boarding = str(" ")

# ------------------------------------------------------------------------

seat_comfort = st.selectbox(("Add Seat comfort: "),
                                star)

if seat_comfort == "---":
    seat_comfort = str(" ")

# ----------------------------------------------------------------------

inflight_entertainment = st.selectbox(("Add Inflight entertainment: "),
                                star)

if inflight_entertainment == "---":
    inflight_entertainment = str(" ")

# ------------------------------------------------------------------------

on_board_service = st.selectbox(("Add On-board service: "),
                                star)

if on_board_service == "---":
    on_board_service = str(" ")

# -----------------------------------------------------------------------

leg_room_service = st.selectbox(("Add Leg room service: "),
                                star)

if leg_room_service == "---":
    leg_room_service = str(" ")

# -----------------------------------------------------------------------
	
baggage_handling = st.selectbox(("Add Baggage handling	: "),
                                star)

if baggage_handling == "---":
    baggage_handling = str(" ")

# ------------------------------------------------------------------------

checkin_service	 = st.selectbox(("Add Checkin service: "),
                                star)

if checkin_service == "---":
    checkin_service = str(" ")

# ----------------------------------------------------------------------------

inflight_service = st.selectbox(("Add Inflight service	: "),
                                star)

if inflight_service == "---":
    inflight_service = str(" ")

# -------------------------------------------------------------------------

datapoint = {"Age": [age],
             "Flight Distance": [flight_distance],
             "Gender": [gender],
             "Customer Type": [customer_type],
             "Type of Travel": [travel_type],
             "Class": [travel_class],
             "Inflight wifi service": [wifi_service],
             "Departure/Arrival time convenient": [departure_arrival_time],
             "Gate location": [gate_location],
             "Food and drink": [food_drink],
             "Online boarding": [online_boarding],
             "Seat comfort": [seat_comfort],
             "Inflight entertainment": [inflight_entertainment],
             "On-board service": [on_board_service],
             "Leg room service": [leg_room_service],
             "Baggage handling": [baggage_handling],
             "Checkin service": [checkin_service],
             "Inflight service": [inflight_service]}    

datapoint_df = pd.DataFrame(datapoint)

cat_index = list(range(6, 18))
datapoint_df.iloc[:,cat_index] = datapoint_df.iloc[:,cat_index].astype('object')

end_point = pipeline.transform(datapoint_df)
# Extract categorical encoder details

categorical_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
onehot_encoded_cols = categorical_encoder.get_feature_names_out(datapoint_df.select_dtypes(include="object").columns)

# Combine numerical and encoded categorical columns
all_feature_names = list(datapoint_df.select_dtypes(exclude="object").columns) + list(onehot_encoded_cols)

# Create the final DataFrame
normal_endpoint = pd.DataFrame(end_point, columns=all_feature_names)
final_endpoint = normal_endpoint[necessary_col]

if st.button("Qiymət təxmini: "):
    progress_text = "Prediction . . . "
    my_bar = st.sidebar.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete+1, text=progress_text)
    my_bar.empty()

    predict = model_xgb.predict(final_endpoint)
    st.write(predict)             



