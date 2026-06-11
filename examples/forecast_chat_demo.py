"""LLM-driven hydrology forecast Q&A — minimal chained-tools smoke demo.

Three @tool functions mirror the intent of `initial_conditions_adjustment_q.py`
(river location, current forecast, return-period thresholds) without pulling
in the heavy GEOGloWS/geopandas/plotly/statsmodels stack. The data returned is
synthetic-but-realistic so the LLM can chain the tools and reason over their
outputs the same way it would on real GEOGloWS data.

Run:
    OLLAMA_HOST=http://localhost:11434 python examples/forecast_chat_demo.py
"""

from tethys_agents.react_agent import ReactAgent
from tethys_agents.tool import tool


# ---- Tools (synthetic data; same shape as the notebook helpers would return) ----

# A small lookup table covering a couple of GEOGloWS-style 9-digit river IDs.
# Replace with real `wf.fetch_geoglows_data(...)` calls once geoglows is installed.
_RIVER_DB = {
    760013500: {
        "lat": 6.2518,
        "lon": -75.5636,
        "name": "Magdalena River (Caracoli)",
        "mean_flow_m3s": 412.3,
        "peak_flow_m3s": 1487.6,
        "horizon_days": 15,
        "return_periods": {
            2: 1235.4,
            5: 1789.1,
            10: 2103.8,
            25: 2487.2,
            50: 2769.5,
            100: 3045.7,
        },
    },
    560014200: {
        "lat": -1.4521,
        "lon": -78.0067,
        "name": "Pastaza River (Andes)",
        "mean_flow_m3s": 95.8,
        "peak_flow_m3s": 312.4,
        "horizon_days": 15,
        "return_periods": {
            2: 280.1,
            5: 410.6,
            10: 498.3,
            25: 605.7,
            50: 686.4,
            100: 768.0,
        },
    },
}


def _lookup(river_id: int) -> dict:
    if river_id not in _RIVER_DB:
        raise ValueError(
            f"River ID {river_id} not in demo database. "
            f"Known IDs: {list(_RIVER_DB.keys())}"
        )
    return _RIVER_DB[river_id]


@tool
def get_river_location(river_id: int) -> str:
    """Return the latitude and longitude for a GEOGloWS 9-digit river ID."""
    r = _lookup(river_id)
    return (
        f"River {river_id} ({r['name']}) is at "
        f"latitude {r['lat']:.4f}, longitude {r['lon']:.4f}."
    )


@tool
def get_forecast_summary(river_id: int) -> str:
    """Return mean and peak flow from the current 15-day GEOGloWS forecast."""
    r = _lookup(river_id)
    return (
        f"Forecast for river {river_id} over the next {r['horizon_days']} days: "
        f"mean flow {r['mean_flow_m3s']:.1f} m^3/s, "
        f"peak flow {r['peak_flow_m3s']:.1f} m^3/s."
    )


@tool
def get_return_period_thresholds(river_id: int) -> str:
    """Return the 2/5/10/25/50/100-year return-period flow thresholds (m^3/s)."""
    r = _lookup(river_id)
    parts = [f"{rp}-year: {flow:.1f}" for rp, flow in r["return_periods"].items()]
    return (
        f"Return-period thresholds for river {river_id} (m^3/s): "
        + "; ".join(parts) + "."
    )


# ---- Run the agent ----

if __name__ == "__main__":
    agent = ReactAgent(
        tools=[
            get_river_location,
            get_forecast_summary,
            get_return_period_thresholds,
        ],
        model="qwen3:latest",
        system_prompt=(
            "You are a hydrology analyst helping with river forecasts. "
            "Use the available tools to look up information for the user's "
            "river. When the user asks about a forecast peak, also fetch "
            "the return-period thresholds and compare them to give context."
        ),
    )

    answer = agent.run(
        user_msg=(
            "For river 760013500, where is it located, what does the "
            "current 15-day forecast look like, and how does the peak "
            "compare to the historical return-period thresholds?"
        ),
        max_rounds=5,
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER:")
    print("=" * 70)
    print(answer)
