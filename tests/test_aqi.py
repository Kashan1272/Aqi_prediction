from aqi_predictor.aqi import category, pm10_to_us_aqi, pm25_to_us_aqi


def test_aqi_breakpoints():
    assert pm25_to_us_aqi(5.0) <= 50
    assert pm25_to_us_aqi(40.0) > 100
    assert pm10_to_us_aqi(100) > 50


def test_categories():
    assert category(40) == "Good"
    assert category(175) == "Unhealthy"
    assert category(350) == "Hazardous"
