"""Open-Meteo에서 코디 추천에 사용할 현재 날씨와 시간별 예보를 조회한다."""

import json
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict


class WeatherInfo(BaseModel):
    """추천을 원하는 장소와 시간대의 날씨. 온도 단위는 섭씨다."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    temperature_c: float
    feels_like_c: float | None = None
    conditions: str | None = None


_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_WEATHER_FIELDS = ("temperature_2m", "apparent_temperature", "weather_code")
_WEATHER_CONDITIONS = {
    0: "맑음",
    1: "대체로 맑음",
    2: "부분적으로 흐림",
    3: "흐림",
    45: "안개",
    48: "안개와 서리",
    51: "약한 이슬비",
    53: "이슬비",
    55: "강한 이슬비",
    56: "약한 어는 이슬비",
    57: "강한 어는 이슬비",
    61: "약한 비",
    63: "비",
    65: "강한 비",
    66: "약한 어는 비",
    67: "강한 어는 비",
    71: "약한 눈",
    73: "눈",
    75: "강한 눈",
    77: "싸락눈",
    80: "약한 소나기",
    81: "소나기",
    82: "강한 소나기",
    85: "약한 눈 소나기",
    86: "강한 눈 소나기",
    95: "뇌우",
    96: "약한 우박을 동반한 뇌우",
    97: "강한 뇌우",
    99: "강한 우박을 동반한 뇌우",
}


class WeatherLookupError(RuntimeError):
    """날씨 API 호출이 실패했거나 응답을 해석할 수 없다."""


def _fetch_json(url: str, params: dict[str, str | float | int]) -> dict:
    try:
        with urlopen(f"{url}?{urlencode(params)}", timeout=8) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, ValueError) as exc:
        raise WeatherLookupError("날씨 조회 서비스에 연결하거나 응답을 읽을 수 없습니다.") from exc
    if not isinstance(payload, dict) or payload.get("error"):
        raise WeatherLookupError("날씨 조회 서비스에서 정상 응답을 받지 못했습니다.")
    return payload


def get_weather(
    location: str,
    *,
    at: datetime | None = None,
    country_code: str | None = None,
) -> WeatherInfo:
    """지역명의 첫 검색 결과를 사용해 현재 날씨 또는 지정 시간의 예보를 반환한다.

    at은 시간대가 포함된 datetime이며, 해당 시각을 포함하는 시간의 예보를 사용한다.
    country_code에는 KR 같은 ISO 국가 코드로 지역 검색 범위를 지정할 수 있다.
    지역을 찾지 못하거나 지정 시각이 예보 범위를 벗어나면 ValueError를 발생시킨다.
    """
    location = location.strip()
    if len(location) < 2:
        raise ValueError("조회할 지역명을 두 글자 이상 입력해 주세요.")
    if at is not None and at.utcoffset() is None:
        raise ValueError("예보 시각에는 시간대를 포함해 주세요.")

    search_params: dict[str, str | float | int] = {"name": location, "count": 1, "language": "ko"}
    if country_code is not None:
        code = country_code.strip().upper()
        if len(code) != 2 or not code.isascii() or not code.isalpha():
            raise ValueError("국가 코드는 KR처럼 영문 두 글자로 입력해 주세요.")
        search_params["countryCode"] = code
    matches = _fetch_json(_GEOCODING_URL, search_params).get("results", [])
    if not isinstance(matches, list):
        raise WeatherLookupError("지역 검색 응답의 형식이 올바르지 않습니다.")
    if not matches:
        raise ValueError(f"지역을 찾을 수 없습니다. ({location})")

    try:
        params: dict[str, str | float | int] = {
            "latitude": matches[0]["latitude"],
            "longitude": matches[0]["longitude"],
            "temperature_unit": "celsius",
            "timezone": "GMT",
        }
    except (KeyError, TypeError) as exc:
        raise WeatherLookupError("지역 검색 응답에 좌표가 없습니다.") from exc

    fields = ",".join(_WEATHER_FIELDS)
    if at is None:
        params["current"] = fields
    else:
        params.update(hourly=fields, forecast_days=16)
    forecast = _fetch_json(_FORECAST_URL, params)

    try:
        if at is None:
            values = forecast["current"]
        else:
            hour_utc = at.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
            hour_key = hour_utc.strftime("%Y-%m-%dT%H:%M")
            hourly = forecast["hourly"]
            try:
                index = hourly["time"].index(hour_key)
            except ValueError:
                raise ValueError("요청한 시각이 시간별 예보 범위를 벗어났습니다.") from None
            values = {field: hourly[field][index] for field in _WEATHER_FIELDS}
        temperature, feels_like, weather_code = (values[field] for field in _WEATHER_FIELDS)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise WeatherLookupError("날씨 조회 응답에 필요한 정보가 없습니다.") from exc

    try:
        return WeatherInfo(
            temperature_c=temperature,
            feels_like_c=feels_like,
            conditions=_WEATHER_CONDITIONS.get(weather_code),
        )
    except (TypeError, ValueError) as exc:
        raise WeatherLookupError("날씨 조회 응답에 유효한 기온 또는 날씨 상태가 없습니다.") from exc
