import os
import requests
from typing import Optional
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv(override=True)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


BASE_URL = "https://api.openweathermap.org/data/2.5"


class _WeatherInputError(Exception):
    """A problem with the tool input or config; its message goes straight to the model."""


def _fetch(endpoint: str, location: Optional[str], lat: Optional[float], lon: Optional[float],
           example_city: str, **extra) -> dict:
    """Call an OpenWeather endpoint by city name or coordinates and return the JSON."""
    api_key = os.getenv("OPENWEATHER_API_KEY") or OPENWEATHER_API_KEY
    if not api_key:
        raise _WeatherInputError("Error: OPENWEATHER_API_KEY is not configured in .env")

    params = {"units": "metric", "appid": api_key, **extra}  # metric = Celsius
    if location and location.strip():
        params["q"] = location.strip()
    elif lat is not None and lon is not None:
        params["lat"] = lat
        params["lon"] = lon
    else:
        raise _WeatherInputError(
            f"Error: Please specify a city name (e.g. '{example_city}') or coordinates (lat, lon)."
        )

    response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


@tool
def get_current_weather(
    location: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> str:
    """
    Fetches the current weather data for a city name (e.g. 'San Francisco', 'Tokyo', 'London')
    or coordinates (latitude and longitude) using the OpenWeather API.

    Args:
        location (str, optional): The name of the city or location (e.g. 'San Francisco', 'Paris').
        lat (float, optional): Latitude coordinate if city name is not provided.
        lon (float, optional): Longitude coordinate if city name is not provided.
    """
    try:
        data = _fetch("weather", location, lat, lon, example_city="San Francisco")

        city_name = data.get("name", location or "Requested Location")
        weather_info = data.get("weather", [{}])[0]
        desc = weather_info.get("description", "N/A")
        main = data.get("main", {})
        temp = main.get("temp", "N/A")
        feels_like = main.get("feels_like", "N/A")
        humidity = main.get("humidity", "N/A")
        wind = data.get("wind", {}).get("speed", "N/A")

        return f"Weather in {city_name}: {temp}°C (feels like {feels_like}°C), {desc}, Humidity: {humidity}%, Wind: {wind} m/s."
    except _WeatherInputError as e:
        return str(e)
    except Exception as e:
        return f"Failed to fetch weather data: {str(e)}"


@tool
def get_weather_forecast(
    location: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> str:
    """
    Fetches a multi-day weather forecast for a city name (e.g. 'Tokyo', 'London')
    or coordinates (latitude and longitude) using the OpenWeather 5-day/3-hour API.

    Use this tool when the user asks about the forecast, upcoming weather, or
    weather for the next few days.

    Args:
        location (str, optional): The name of the city or location.
        lat (float, optional): Latitude coordinate if city name is not provided.
        lon (float, optional): Longitude coordinate if city name is not provided.
    """
    try:
        # cnt=5 entries ≈ 15 hours of forecast data
        data = _fetch("forecast", location, lat, lon, example_city="Tokyo", cnt=5)

        city_name = data.get("city", {}).get("name", location or "Requested Location")
        forecasts = data.get("list", [])
        if not forecasts:
            return f"No forecast data available for {city_name}."

        lines = [f"Forecast for {city_name}:"]
        for entry in forecasts:
            dt_txt = entry.get("dt_txt", "N/A")
            main = entry.get("main", {})
            temp = main.get("temp", "N/A")
            desc = entry.get("weather", [{}])[0].get("description", "N/A")
            humidity = main.get("humidity", "N/A")
            lines.append(f"  {dt_txt}: {temp}°C, {desc}, Humidity: {humidity}%")

        return "\n".join(lines)
    except _WeatherInputError as e:
        return str(e)
    except Exception as e:
        return f"Failed to fetch forecast data: {str(e)}"


tools = [get_current_weather, get_weather_forecast]