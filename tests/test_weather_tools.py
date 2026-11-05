import unittest
from unittest.mock import MagicMock, patch

from tools.weather import get_current_weather, get_weather_forecast


def _response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


@patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"})
class TestWeatherTools(unittest.TestCase):
    def test_current_weather_by_city(self):
        payload = {
            "name": "Paris",
            "weather": [{"description": "light rain"}],
            "main": {"temp": 12, "feels_like": 10, "humidity": 80},
            "wind": {"speed": 3},
        }
        with patch("tools.weather.requests.get", return_value=_response(payload)) as get:
            result = get_current_weather.invoke({"location": " Paris "})
        self.assertEqual(
            result,
            "Weather in Paris: 12°C (feels like 10°C), light rain, Humidity: 80%, Wind: 3 m/s.",
        )
        url = get.call_args.args[0]
        params = get.call_args.kwargs["params"]
        self.assertTrue(url.endswith("/weather"))
        self.assertEqual(params["q"], "Paris")
        self.assertEqual(params["units"], "metric")

    def test_forecast_by_coordinates(self):
        payload = {
            "city": {"name": "Tokyo"},
            "list": [{"dt_txt": "2026-10-27 12:00:00", "main": {"temp": 20, "humidity": 50},
                      "weather": [{"description": "clear sky"}]}],
        }
        with patch("tools.weather.requests.get", return_value=_response(payload)) as get:
            result = get_weather_forecast.invoke({"lat": 35.6, "lon": 139.7})
        self.assertIn("Forecast for Tokyo:", result)
        self.assertIn("2026-10-27 12:00:00: 20°C, clear sky, Humidity: 50%", result)
        url = get.call_args.args[0]
        params = get.call_args.kwargs["params"]
        self.assertTrue(url.endswith("/forecast"))
        self.assertEqual((params["lat"], params["lon"], params["cnt"]), (35.6, 139.7, 5))

    def test_forecast_with_no_entries(self):
        with patch("tools.weather.requests.get", return_value=_response({"city": {"name": "X"}, "list": []})):
            self.assertEqual(get_weather_forecast.invoke({"location": "X"}), "No forecast data available for X.")

    def test_missing_location(self):
        with patch("tools.weather.requests.get") as get:
            self.assertEqual(
                get_current_weather.invoke({}),
                "Error: Please specify a city name (e.g. 'San Francisco') or coordinates (lat, lon).",
            )
            self.assertEqual(
                get_weather_forecast.invoke({}),
                "Error: Please specify a city name (e.g. 'Tokyo') or coordinates (lat, lon).",
            )
            get.assert_not_called()

    def test_missing_api_key(self):
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": ""}), patch("tools.weather.OPENWEATHER_API_KEY", None):
            self.assertEqual(
                get_current_weather.invoke({"location": "Paris"}),
                "Error: OPENWEATHER_API_KEY is not configured in .env",
            )

    def test_request_failure(self):
        with patch("tools.weather.requests.get", side_effect=ConnectionError("down")):
            self.assertEqual(get_current_weather.invoke({"location": "Paris"}), "Failed to fetch weather data: down")
            self.assertEqual(get_weather_forecast.invoke({"location": "Paris"}), "Failed to fetch forecast data: down")


if __name__ == "__main__":
    unittest.main()
