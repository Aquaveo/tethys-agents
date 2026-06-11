"""Real-HTTP chained-tools demo: USGS streamflow analyst.

Three @tool functions hit the USGS Water Services API live (no API key,
no auth, JSON response) and return short string summaries the LLM can
chain together to compare current conditions against the past week.

The same chained-tools shape applies to GEOGloWS-based tools that wrap
``initial_conditions_adjustment_q.get_reach_location`` and friends — once
``geoglows``/``geopandas``/``boto3`` are installed in the environment, add
them as additional @tool functions alongside these USGS ones.

Run:
    python examples/forecast_real_demo.py
"""

import requests

from tethys_agents.react_agent import ReactAgent
from tethys_agents.tool import tool

USGS_BASE = "https://waterservices.usgs.gov/nwis"
DISCHARGE_PARAM = "00060"  # cubic feet per second
HTTP_TIMEOUT = 30


def _fetch_iv_payload(site_no: str, period: str = "") -> dict:
    """Internal helper: fetch USGS Instantaneous Values JSON for a site."""
    site_no = str(site_no).strip()
    params = f"format=json&sites={site_no}&parameterCd={DISCHARGE_PARAM}"
    if period:
        params += f"&period={period}"
    url = f"{USGS_BASE}/iv/?{params}"
    resp = requests.get(url, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@tool
def get_usgs_site_info(site_no: str) -> str:
    """Return the name, latitude, and longitude for a USGS streamflow gauge site."""
    payload = _fetch_iv_payload(site_no)
    series = payload["value"]["timeSeries"]
    if not series:
        return f"USGS site {site_no} not found or has no discharge data."
    info = series[0]["sourceInfo"]
    name = info["siteName"]
    geo = info["geoLocation"]["geogLocation"]
    return (
        f"USGS site {site_no} ({name}) is at "
        f"latitude {geo['latitude']:.4f}, longitude {geo['longitude']:.4f}."
    )


@tool
def get_current_streamflow(site_no: str) -> str:
    """Return the most recent instantaneous discharge measurement (cubic feet per second)."""
    payload = _fetch_iv_payload(site_no)
    series = payload["value"]["timeSeries"]
    if not series or not series[0]["values"][0]["value"]:
        return f"No recent discharge measurements for USGS site {site_no}."
    latest = series[0]["values"][0]["value"][-1]
    flow_cfs = float(latest["value"])
    when = latest["dateTime"]
    return (
        f"Most recent discharge at USGS site {site_no} is "
        f"{flow_cfs:,.0f} cubic feet per second, measured at {when}."
    )


@tool
def get_recent_flow_statistics(site_no: str) -> str:
    """Return mean, max, and min instantaneous discharge over the last 7 days (cubic feet per second)."""
    payload = _fetch_iv_payload(site_no, period="P7D")
    series = payload["value"]["timeSeries"]
    if not series or not series[0]["values"][0]["value"]:
        return f"No 7-day discharge history for USGS site {site_no}."
    raw_values = series[0]["values"][0]["value"]
    numeric = [
        float(v["value"]) for v in raw_values if v["value"] not in ("", None)
    ]
    if not numeric:
        return (
            f"USGS site {site_no} returned only missing values for the past 7 days."
        )
    mean_cfs = sum(numeric) / len(numeric)
    return (
        f"Over the past 7 days at USGS site {site_no}, discharge averaged "
        f"{mean_cfs:,.0f} cubic feet per second, ranging from "
        f"{min(numeric):,.0f} cfs to {max(numeric):,.0f} cfs "
        f"across {len(numeric)} 15-minute readings."
    )


if __name__ == "__main__":
    agent = ReactAgent(
        tools=[
            get_usgs_site_info,
            get_current_streamflow,
            get_recent_flow_statistics,
        ],
        model="qwen3:latest",
        system_prompt=(
            "You are a hydrology analyst helping with US river streamflow data. "
            "Use the available tools to query the USGS Water Services API. "
            "When the user asks about current flow at a site, also fetch the "
            "recent statistics so you can put the current reading in context "
            "(above or below typical recent conditions). Keep answers concise "
            "and quote specific numbers from the tool observations."
        ),
    )

    answer = agent.run(
        user_msg=(
            "What is the streamflow right now at USGS site 11447650 "
            "(Sacramento River at Freeport, CA)? Where is it located, and "
            "how does the current flow compare to the last week's average?"
        ),
        max_rounds=5,
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER:")
    print("=" * 70)
    print(answer)
