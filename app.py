import streamlit as st
import pandas as pd
import requests
import pickle
import xgboost as xgb
import datetime

# Set page title
st.set_page_config(page_title="Flight Delay Prediction")

# Page header
st.header("Flight Delay Prediction")

st.markdown("Maximum 7 Days before your Flight")

# Input section
st.subheader("Enter Flight Details")

# Airline selection
airline_mapping = {
    "American Airlines": "AA",
    "Alaska Airlines": "AS",
    "JetBlue Airways": "B6",
    "Delta Air Lines": "DL",
    "Frontier Airlines": "F9",
    "Spirit Airlines": "NK",
    "United Airlines": "UA",
    "Southwest Airlines": "WN"
}
airline_options = list(airline_mapping.keys())
selected_airline = st.selectbox("Select Airline", airline_options)
airline_code = airline_mapping[selected_airline]

# Load the merged airport data
merged_data = pd.read_csv('./merged_airport_data.csv')

# Origin airport selection
origin = st.selectbox("Select Origin Airport", merged_data['Origin Airport Code'].unique())

# Destination airport selection
destination = st.selectbox("Select Destination Airport", merged_data['Dest Airport Code'].unique())

# Retrieve airport information based on user selection
selected_route = merged_data[(merged_data['Origin Airport Code'] == origin) & (merged_data['Dest Airport Code'] == destination)]

month = st.selectbox("Select Month", ["January", "February", "March", "April", "May", "June",
                                       "July", "August", "September", "October", "November", "December"])

selected_date = st.date_input("Select Date", min_value=datetime.date.today(), max_value=datetime.date.today() + datetime.timedelta(days=7))

day_of_week = selected_date.strftime("%A")

# Departure time selection
departure_time = st.selectbox("Select Departure Time", ["00:00", "01:00", "02:00", "04:00", "05:00", "06:00",
                                                         "07:00", "08:00", "09:00", "10:00", "11:00", "12:00",
                                                         "13:00", "14:00", "15:00", "16:00", "17:00", "18:00",
                                                         "19:00", "20:00", "21:00", "22:00", "23:00"])
def get_weather_value(weather_data, selected_date, departure_time):
    selected_datetime = datetime.datetime.combine(selected_date, datetime.datetime.strptime(departure_time, "%H:%M").time())
    
    closest_value = None
    min_time_diff = datetime.timedelta.max
    
    for value in weather_data['values']:
        valid_time = value['validTime'].split('/')[0]
        valid_datetime = datetime.datetime.strptime(valid_time, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        
        time_diff = abs(selected_datetime - valid_datetime)
        
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_value = value['value']
    
    return closest_value


if not selected_route.empty:
    origin_lat = int(selected_route['Origin Latitude'].values[0])
    origin_lon = int(selected_route['Origin Longitude'].values[0])
    dest_lat = int(selected_route['Dest Latitude'].values[0])
    dest_lon = int(selected_route['Dest Longitude'].values[0])
    fly_distance = int(selected_route['Fly Distance'].values[0])
    
    origin_grid_url = f"https://api.weather.gov/points/{origin_lat},{origin_lon}"
    dest_grid_url = f"https://api.weather.gov/points/{dest_lat},{dest_lon}"
    
    origin_grid_data = requests.get(origin_grid_url).json()
    dest_grid_data = requests.get(dest_grid_url).json()
    
    origin_grid_x = origin_grid_data['properties']['gridX']
    origin_grid_y = origin_grid_data['properties']['gridY']
    dest_grid_x = dest_grid_data['properties']['gridX']
    dest_grid_y = dest_grid_data['properties']['gridY']
    
    origin_weather_url = f"https://api.weather.gov/gridpoints/{origin_grid_data['properties']['gridId']}/{origin_grid_x},{origin_grid_y}"
    dest_weather_url = f"https://api.weather.gov/gridpoints/{dest_grid_data['properties']['gridId']}/{dest_grid_x},{dest_grid_y}"
    
    origin_weather_data = requests.get(origin_weather_url).json()
    dest_weather_data = requests.get(dest_weather_url).json()
    
    # Extract relevant weather information for origin airport
# Extract relevant weather information for origin airport
    origin_precip = get_weather_value(origin_weather_data['properties']['quantitativePrecipitation'], selected_date, departure_time)
    origin_wind_speed = get_weather_value(origin_weather_data['properties']['windSpeed'], selected_date, departure_time)
    origin_snowfall = get_weather_value(origin_weather_data['properties']['snowfallAmount'], selected_date, departure_time)

    # Extract relevant weather information for destination airport
    dest_precip = get_weather_value(dest_weather_data['properties']['quantitativePrecipitation'], selected_date, departure_time)
    dest_wind_speed = get_weather_value(dest_weather_data['properties']['windSpeed'], selected_date, departure_time)
    dest_snowfall = get_weather_value(dest_weather_data['properties']['snowfallAmount'], selected_date, departure_time)

    airline_columns = [f"Airline_{code}" for code in airline_mapping.values()]
    origin_columns = [f"Origin_{code}" for code in merged_data['Origin Airport Code'].unique()]
    dest_columns = [f"Dest_{code}" for code in merged_data['Dest Airport Code'].unique()]
    month_columns = [f"Month_{i}" for i in range(1, 13)]
    day_of_week_columns = [f"DayOfWeek_{i}" for i in range(1, 8)]
    hour_columns = [f"Hour_{i}" for i in range(24)]

    # Create a dictionary to store the data
    data = {
        'Fly Distance': [fly_distance],
        'AWND': [origin_wind_speed],
        'PRCP': [origin_precip],
        'SNOW': [origin_snowfall],
        'AWND_des': [dest_wind_speed],
        'PRCP_des': [dest_precip],
        'SNOW_des': [dest_snowfall]
    }
    for column in ['AWND', 'PRCP', 'SNOW', 'AWND_des', 'PRCP_des', 'SNOW_des']:
        data[column] = [0 if x is None else x for x in data[column]]

    # Add airline columns
    data.update({col: [1 if airline_code == col.split("_")[1] else 0] for col in airline_columns})

    # Add origin airport columns
    data.update({col: [1 if origin == col.split("_")[1] else 0] for col in origin_columns})

    # Add destination airport columns
    data.update({col: [1 if destination == col.split("_")[1] else 0] for col in dest_columns})

    # Add month columns
    data.update({col: [1 if month == pd.to_datetime(col, format='Month_%m').strftime('%B') else 0] for col in month_columns})

    # Create a dictionary to map day of the week column names to their corresponding day names
    day_of_week_mapping = {
        'DayOfWeek_1': 'Monday',
        'DayOfWeek_2': 'Tuesday',
        'DayOfWeek_3': 'Wednesday',
        'DayOfWeek_4': 'Thursday',
        'DayOfWeek_5': 'Friday',
        'DayOfWeek_6': 'Saturday',
        'DayOfWeek_7': 'Sunday'
    }

    # Add day of week columns
    data.update({col: [1 if day_of_week == day_of_week_mapping[col] else 0] for col in day_of_week_columns})

    # Add hour columns
    data.update({col: [1 if departure_time == f"{int(col.split('_')[1]):02d}:00" else 0] for col in hour_columns})

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    df = df.reset_index(drop=True)
    print(df.shape)
    st.write(df.iloc[:, 0])
    with open('xgb_classifier.pkl', 'rb') as file:
        xgb_classifier = pickle.load(file)
        

    if st.button("Predict"):
        st.dataframe(df)

        delay_probability = xgb_classifier.predict_proba(df)[:, 1][0]
        st.write(f"The probability of flight delay is: {delay_probability:.2%}")

else:
    st.write("This Route Doesn't Exist. Please Enter Again :(")

