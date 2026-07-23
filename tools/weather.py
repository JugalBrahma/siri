import os
import requests
from langchain_core.tools import tool

# Make sure your API key is set in your environment variables
# or replace os.getenv with your hardcoded key string.
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

@tool
def get_current_weather(lat: float, lon: float) -> str:
    """
    Fetches the current weather data for a given latitude and longitude 
    using the OpenWeather API.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "units": "metric",  # Defaults to Celsius
        "appid": OPENWEATHER_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Pull out a neat summary string for the agent to read
        weather_info = data.get("weather", [{}])[0]
        desc = weather_info.get("description", "N/A")
        temp = data.get("main", {}).get("temp", "N/A")
        
        return f"The current temperature is {temp}°C with {desc}."
    except Exception as e:
        return f"Failed to fetch weather data: {str(e)}"

# Put your tool inside a list
tools = [get_current_weather]