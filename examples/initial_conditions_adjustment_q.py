"""Reusable workflow for forecast bias correction and initialization adjustment.

This file was converted from a Colab notebook export. Import it to reuse the
individual data-fetching, bias-correction, plotting, and AR(1) adjustment
functions, or run it directly to execute the original interactive workflow.
"""

from __future__ import annotations

import datetime as dt
import io
import math
import warnings
from dataclasses import dataclass
from typing import Any

import boto3
import geoglows
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pytz
import requests
import urllib3
from botocore import UNSIGNED
from botocore.config import Config
from timezonefinder import TimezoneFinder


METADATA_TABLE_URL = (
    "http://geoglows-v2.s3-us-west-2.amazonaws.com/tables/"
    "package-metadata-table.parquet"
)
DEFAULT_RETURN_PERIODS = (2, 5, 10, 25, 50, 100)
DEFAULT_IDEAM_STATION_ID = "0026177030"
DEFAULT_IDEAM_VARIABLE = "Q_MEDIA_D"
DEFAULT_IDEAM_FEWS_URL = (
    "https://fews.ideam.gov.co/visorfews/data/series/jsonQ/"
    f"{DEFAULT_IDEAM_STATION_ID}.json"
)
RETURN_PERIOD_COLORS = {
    "2 Year": "rgba(254, 240, 1, .4)",
    "5 Year": "rgba(253, 154, 1, .4)",
    "10 Year": "rgba(255, 56, 5, .4)",
    "25 Year": "rgba(255, 0, 0, .4)",
    "50 Year": "rgba(128, 0, 106, .4)",
    "100 Year": "rgba(128, 0, 246, .4)",
}


@dataclass
class WorkflowResult:
    river_id: int
    latitude: float
    longitude: float
    timezone_offset_hours: float
    daily_df: pd.DataFrame
    ensembles: pd.DataFrame
    stats_df: pd.DataFrame
    records: pd.DataFrame
    observed_discharge: pd.DataFrame
    observed_realtime: pd.DataFrame
    observed_sensors: pd.DataFrame
    simulated_return_periods: pd.DataFrame
    corrected_historical: pd.DataFrame
    corrected_return_periods: pd.DataFrame
    corrected_ensembles: pd.DataFrame
    corrected_stats: pd.DataFrame
    fixed_records: pd.DataFrame
    corrected_ensembles_ar1: pd.DataFrame
    corrected_stats_ar1: pd.DataFrame
    location_figure: go.Figure
    raw_forecast_figure: go.Figure
    corrected_forecast_figure: go.Figure
    ar1_forecast_figure: go.Figure


def configure_notebook_output(renderer: str = "colab") -> None:
    """Configure notebook-friendly Plotly output and warnings."""
    pio.renderers.default = renderer
    warnings.filterwarnings("ignore")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    warnings.filterwarnings(
        "ignore",
        category=urllib3.exceptions.InsecureRequestWarning,
    )


def display_dataframe(df: pd.DataFrame) -> None:
    """Display a DataFrame as HTML when IPython is available."""
    from IPython.core.display import HTML
    from IPython.display import display

    display(HTML(df.to_html()))


def validate_river_id(river_id: int | str) -> int:
    """Validate and return a GEOGloWS 9-digit river ID."""
    try:
        river_id = int(river_id)
    except ValueError as exc:
        raise ValueError("river_id must be an integer.") from exc

    if not (10**8 <= river_id <= 10**9 - 1):
        raise ValueError("river_id must be a 9-digit integer.")

    return river_id


def gumbel_1(std: float, xbar: float, rp: int | float) -> float:
    """Return the Gumbel Type I flow value for a return period."""
    return -math.log(-math.log(1 - (1 / rp))) * std * 0.7797 + xbar - (0.45 * std)


def normalize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp negative values to zero and normalize the index to daily timestamps."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df[df < 0] = 0
    df.index = pd.to_datetime(df.index.to_series().dt.strftime("%Y-%m-%d"))
    return df


def normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp negative values to zero and normalize the index to second precision."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df[df < 0] = 0
    df.index = pd.to_datetime(df.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S"))
    return df


def fetch_metadata_table(metadata_table_url: str = METADATA_TABLE_URL) -> pd.DataFrame:
    """Load the GEOGloWS reach metadata table."""
    return pd.read_parquet(metadata_table_url)


def get_reach_location(
    river_id: int,
    metadata_table: pd.DataFrame | None = None,
) -> tuple[float, float]:
    """Return latitude and longitude for a GEOGloWS reach."""
    river_id = validate_river_id(river_id)
    metadata_table = metadata_table if metadata_table is not None else fetch_metadata_table()
    match = metadata_table.loc[metadata_table["LINKNO"] == river_id, ["lat", "lon"]]

    if match.empty:
        raise ValueError(f"River ID {river_id} was not found in the metadata table.")

    latitude, longitude = match.iloc[0]
    return float(latitude), float(longitude)


def build_reach_geodataframe(
    river_id: int,
    latitude: float,
    longitude: float,
) -> gpd.GeoDataFrame:
    """Build a one-point GeoDataFrame for the requested reach."""
    df = pd.DataFrame(
        {
            "Longitude": [longitude],
            "Latitude": [latitude],
            "River_ID": [str(river_id)],
        }
    )
    return gpd.GeoDataFrame(
        data=df,
        geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
    ).set_crs("epsg:4326")


def plot_reach_location(
    river_id: int,
    latitude: float,
    longitude: float,
    zoom: int = 5,
    height: int = 500,
) -> go.Figure:
    """Create a map for the selected reach."""
    geo_df = build_reach_geodataframe(river_id, latitude, longitude)
    fig = px.scatter_mapbox(
        geo_df,
        lat=geo_df.geometry.y,
        lon=geo_df.geometry.x,
        hover_data=["River_ID"],
        color_discrete_sequence=["red"],
        zoom=zoom,
        height=height,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "ESRI",
                "source": [
                    "https://services.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ],
            },
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "ESRI",
                "source": [
                    "https://services.arcgisonline.com/ArcGIS/rest/services/"
                    "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
                ],
            },
        ],
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    return fig


def get_timezone_offset_hours(latitude: float, longitude: float) -> float:
    """Return the current UTC offset for a latitude/longitude."""
    timezone_str = TimezoneFinder().timezone_at(lng=longitude, lat=latitude)
    if timezone_str is None:
        raise ValueError(f"Could not find timezone for {latitude=}, {longitude=}.")

    timezone = pytz.timezone(timezone_str)
    local_now = dt.datetime.now(pytz.utc).astimezone(timezone)
    return local_now.utcoffset().total_seconds() / 3600


def fetch_geoglows_data(river_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch retrospective, ensemble, stats, and record GEOGloWS datasets."""
    river_id = validate_river_id(river_id)
    daily_df = normalize_daily_index(geoglows.data.retro_daily(river_id))
    ensembles = normalize_datetime_index(geoglows.data.forecast_ensembles(river_id))
    stats_df = normalize_datetime_index(geoglows.data.forecast_stats(river_id))
    records = normalize_datetime_index(geoglows.data.forecast_records(river_id))
    return daily_df, ensembles, stats_df, records


def calculate_return_periods(
    flow_df: pd.DataFrame,
    river_id: int,
    return_periods: tuple[int, ...] = DEFAULT_RETURN_PERIODS,
) -> pd.DataFrame:
    """Calculate return-period values from annual maximum flow."""
    max_annual_flow = flow_df.groupby(flow_df.index.year).max()
    values = max_annual_flow.values.flatten()
    mean_value = np.mean(values)
    std_value = np.std(values)
    return_period_values = [gumbel_1(std_value, mean_value, rp) for rp in return_periods]

    data = {"rivid": [river_id]}
    data.update({rp: [value] for rp, value in zip(return_periods, return_period_values)})
    return pd.DataFrame(data=data).set_index("rivid").T


def fetch_ideam_daily(
    id_estacion: str,
    variable: str = "NV_MEDIA_D",
    only_approved: bool = False,
) -> pd.DataFrame:
    """Download a daily IDEAM time series from S3."""
    s3 = boto3.client(
        "s3",
        endpoint_url="https://datos.ideam.gov.co/",
        config=Config(signature_version=UNSIGNED),
        verify=False,
    )
    prefix = (
        "observaciones/historicos/parquet/"
        f"label={variable}/id_estacion={id_estacion}/"
    )
    resp = s3.list_objects_v2(Bucket="s3-estacionesideam", Prefix=prefix)
    files = resp.get("Contents", [])
    if not files:
        raise FileNotFoundError(
            f"No hay datos en S3 para estación {id_estacion} variable {variable}"
        )

    dfs = []
    for item in files:
        obj = s3.get_object(Bucket="s3-estacionesideam", Key=item["Key"])
        dfs.append(pd.read_parquet(io.BytesIO(obj["Body"].read())))

    df = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=["fecha"], keep="first")
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    if only_approved:
        df = df[df["nivel_aprobacion"] != "Preliminar"].copy()

    value_column = "Streamflow (m3/s)" if variable.startswith("Q") else "Water Level (cm)"
    return (
        df[["fecha", "valor"]]
        .rename(columns={"fecha": "Datetime", "valor": value_column})
        .set_index("Datetime")
    )


def _parse_fews_section(section: dict[str, Any], value_key: str) -> pd.DataFrame:
    """Parse an IDEAM FEWS JSON section into a discharge DataFrame."""
    records = [item for item in section["data"] if value_key in item]
    df = pd.DataFrame(records)
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%Y/%m/%d %H:%M")
    df = df.rename(columns={"Fecha": "Datetime", value_key: "Discharge (m3/s)"})
    df = df.sort_values("Datetime").reset_index(drop=True)
    df.set_index("Datetime", inplace=True)
    df.index = pd.to_datetime(df.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S"))
    return df


def fetch_ideam_fews_data(url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return observed and sensor IDEAM FEWS discharge series."""
    response = requests.get(url, timeout=30, verify=False)
    response.raise_for_status()
    payload = response.json()
    return _parse_fews_section(payload["Qobs"], "Qobs"), _parse_fews_section(
        payload["Qsen"],
        "Qsen",
    )


def fetch_observed_discharge(
    station_id: str = DEFAULT_IDEAM_STATION_ID,
    variable: str = DEFAULT_IDEAM_VARIABLE,
    fews_url: str = DEFAULT_IDEAM_FEWS_URL,
    timezone_offset_hours: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch historic and real-time IDEAM observations."""
    observed_discharge = normalize_daily_index(fetch_ideam_daily(station_id, variable=variable))
    df_obs, df_sen = fetch_ideam_fews_data(fews_url)
    df_obs.index = df_obs.index - pd.Timedelta(hours=timezone_offset_hours)
    df_sen.index = df_sen.index - pd.Timedelta(hours=timezone_offset_hours)
    return observed_discharge, df_obs, df_sen


def _build_title(main_title: str, plot_titles: list[str] | None) -> str:
    if plot_titles:
        return main_title + "<br>" + "<br>".join(plot_titles)
    return main_title


def _return_period_trace(
    name: str,
    x_values: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp],
    y_values: tuple[float, float, float, float],
    color: str,
    visible: bool | str,
    fill: str = "toself",
) -> go.Scatter:
    return go.Scatter(
        name=name,
        x=x_values,
        y=y_values,
        legendgroup="returnperiods",
        fill=fill,
        visible=visible,
        line=dict(color=color, width=0),
    )


def add_return_periods_to_figure(
    fig: go.Figure,
    stats_df: pd.DataFrame,
    return_periods_df: pd.DataFrame,
    x_values: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp],
    max_visible: float,
    min_visible: float | None = None,
    value_precision: int = 0,
) -> go.Figure:
    """Add return-period threshold bands to a Plotly forecast figure."""
    r2 = round(float(return_periods_df.loc[2].values[0]), value_precision)
    visible = True if max_visible > r2 else "legendonly"
    fig.for_each_trace(
        lambda trace: trace.update(visible=True)
        if trace.name == "Maximum & Minimum Flow"
        else None
    )

    if min_visible is None:
        base_value = float(return_periods_df.loc[100].values[0]) * 0.05
    else:
        base_value = min_visible * 0.95

    values = {
        rp: round(float(return_periods_df.loc[rp].values[0]), value_precision)
        for rp in DEFAULT_RETURN_PERIODS
    }
    label_format = "{:.2f}" if value_precision else "{:.0f}"

    fig.add_trace(
        _return_period_trace(
            "Return Periods",
            x_values,
            (base_value, base_value, base_value, base_value),
            "rgba(0,0,0,0)",
            visible,
            fill="none",
        )
    )
    for lower, upper in zip(DEFAULT_RETURN_PERIODS[:-1], DEFAULT_RETURN_PERIODS[1:]):
        label = f"{lower} Year: {label_format.format(values[lower])}"
        color = RETURN_PERIOD_COLORS[f"{lower} Year"]
        fig.add_trace(
            _return_period_trace(
                label,
                x_values,
                (values[lower], values[lower], values[upper], values[upper]),
                color,
                visible,
            )
        )

    r100 = values[100]
    fig.add_trace(
        _return_period_trace(
            f"100 Year: {label_format.format(r100)}",
            x_values,
            (
                r100,
                r100,
                max(r100 + r100 * 0.05, max_visible),
                max(r100 + r100 * 0.05, max_visible),
            ),
            RETURN_PERIOD_COLORS["100 Year"],
            visible,
        )
    )
    fig["layout"]["xaxis"].update(autorange=True)
    if min_visible is not None:
        fig["layout"]["yaxis"].update(autorange=True)
    return fig


def add_series_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    name: str,
    color: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[go.Figure, pd.DataFrame]:
    """Add a single-column DataFrame as a line trace and return plotted rows."""
    data = df.copy()
    if start is not None:
        data = data.loc[data.index >= pd.to_datetime(start)]
    if end is not None:
        data = data.loc[data.index <= pd.to_datetime(end)]
    if len(data.index) == 0:
        return fig, data

    fig.add_trace(
        go.Scatter(
            name=name,
            x=data.index,
            y=data.iloc[:, 0].values,
            line=dict(color=color),
        )
    )
    return fig, data


def plot_forecast_with_context(
    stats_df: pd.DataFrame,
    return_periods_df: pd.DataFrame,
    title: str,
    plot_titles: list[str] | None = None,
    records: pd.DataFrame | None = None,
    historical: pd.DataFrame | None = None,
    observed_discharge: pd.DataFrame | None = None,
    observed_realtime: pd.DataFrame | None = None,
    records_name: str = "1st days forecasts",
    historical_name: str = "Retrospective Simulation",
    historical_color: str = "blue",
    value_precision: int = 0,
) -> go.Figure:
    """Plot forecast statistics with optional records, history, observations, and return periods."""
    fig = geoglows.plots.forecast_stats(df=stats_df, plot_titles=plot_titles)
    forecast_start = pd.to_datetime(stats_df.index[0])
    forecast_end = pd.to_datetime(stats_df.index[-1])
    context_start = forecast_start - dt.timedelta(days=15)
    context_end = forecast_start + dt.timedelta(days=2)

    x_values = (forecast_start, forecast_end, forecast_end, forecast_start)
    max_visible = float(stats_df.max().max())
    min_visible = float(stats_df.min().min())

    if records is not None:
        fig, plotted = add_series_trace(
            fig,
            records,
            records_name,
            "#FFA15A",
            start=context_start,
            end=context_end,
        )
        if len(plotted.index) > 0:
            x_values = (plotted.index[0], forecast_end, forecast_end, plotted.index[0])
            max_visible = max(float(plotted.max().max()), max_visible)
            min_visible = min(float(plotted.min().min()), min_visible)

    if historical is not None:
        fig, plotted = add_series_trace(
            fig,
            historical,
            historical_name,
            historical_color,
            start=context_start,
        )
        if len(plotted.index) > 0:
            x_values = (plotted.index[0], forecast_end, forecast_end, plotted.index[0])
            max_visible = max(float(plotted.max().max()), max_visible)
            min_visible = min(float(plotted.min().min()), min_visible)

    if observed_discharge is not None:
        fig, _ = add_series_trace(
            fig,
            observed_discharge,
            "Observed Discharge (Historic)",
            "green",
            start=context_start,
        )

    if observed_realtime is not None:
        fig, _ = add_series_trace(
            fig,
            observed_realtime,
            "Observed Discharge (FEWS)",
            "brown",
            start=context_start,
        )

    add_return_periods_to_figure(
        fig,
        stats_df,
        return_periods_df,
        x_values,
        max_visible,
        min_visible=min_visible if value_precision else None,
        value_precision=value_precision,
    )
    fig["layout"].update(title=_build_title(title, plot_titles))
    return fig


def correct_historical(
    simulated_data: pd.DataFrame,
    observed_data: pd.DataFrame,
) -> pd.DataFrame:
    """Bias-correct historical simulation against observed discharge."""
    corrected = geoglows.bias.correct_historical(simulated_data, observed_data)
    return normalize_daily_index(corrected)


def bias_correct_forecast_with_bounds(
    sim_hist: pd.DataFrame,
    forecast: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    """Bias-correct a forecast while preserving out-of-range scale factors."""
    monthly_simulated = sim_hist[sim_hist.index.month == forecast.index[0].month].dropna()
    min_simulated = monthly_simulated.min().values[0]
    max_simulated = monthly_simulated.max().values[0]

    min_factor_df = forecast.copy()
    max_factor_df = forecast.copy()
    forecast_for_correction = forecast.copy()

    for column in forecast.columns:
        tmp_array = np.ones(forecast[column].shape[0])
        tmp_array[forecast[column] < min_simulated] = 0
        min_factor = np.where(tmp_array == 0, forecast[column] / min_simulated, tmp_array)

        tmp_array = np.ones(forecast[column].shape[0])
        tmp_array[forecast[column] > max_simulated] = 0
        max_factor = np.where(tmp_array == 0, forecast[column] / max_simulated, tmp_array)

        clipped_forecast = forecast[column].copy()
        clipped_forecast.mask(clipped_forecast <= min_simulated, min_simulated, inplace=True)
        clipped_forecast.mask(clipped_forecast >= max_simulated, max_simulated, inplace=True)

        forecast_for_correction.update(
            pd.DataFrame(clipped_forecast, index=forecast.index, columns=[column])
        )
        min_factor_df.update(pd.DataFrame(min_factor, index=forecast.index, columns=[column]))
        max_factor_df.update(pd.DataFrame(max_factor, index=forecast.index, columns=[column]))

    corrected = geoglows.bias.correct_forecast(forecast_for_correction, sim_hist, observed)
    corrected = corrected.multiply(min_factor_df, axis=0)
    return corrected.multiply(max_factor_df, axis=0)


def calculate_forecast_stats(
    ensembles: pd.DataFrame,
    high_res_column: str = "ensemble_52",
) -> pd.DataFrame:
    """Calculate forecast-stat columns from corrected ensembles."""
    if high_res_column not in ensembles.columns:
        raise KeyError(f"{high_res_column!r} was not found in the ensemble columns.")

    ensemble = ensembles.copy()
    high_res_df = ensemble[high_res_column].to_frame()
    ensemble.drop(columns=[high_res_column], inplace=True)
    ensemble.dropna(inplace=True)
    high_res_df.dropna(inplace=True)

    max_df = ensemble.quantile(1.0, axis=1).to_frame().rename(columns={1.0: "flow_max"})
    p75_df = ensemble.quantile(0.75, axis=1).to_frame().rename(columns={0.75: "flow_75p"})
    p50_df = ensemble.quantile(0.50, axis=1).to_frame().rename(columns={0.50: "flow_med"})
    p25_df = ensemble.quantile(0.25, axis=1).to_frame().rename(columns={0.25: "flow_25p"})
    min_df = ensemble.quantile(0, axis=1).to_frame().rename(columns={0.0: "flow_min"})
    mean_df = ensemble.mean(axis=1).to_frame().rename(columns={0: "flow_avg"})
    high_res_df.rename(columns={high_res_column: "high_res"}, inplace=True)

    return pd.concat(
        [max_df, p75_df, mean_df, p50_df, p25_df, min_df, high_res_df],
        axis=1,
    )


def correct_forecast_records(
    records: pd.DataFrame,
    simulated_history: pd.DataFrame,
    observed_discharge: pd.DataFrame,
) -> pd.DataFrame:
    """Bias-correct forecast records by forecast month."""
    fixed_records = pd.DataFrame()
    for _, monthly_records in records.groupby(records.index.month):
        corrected_values = bias_correct_forecast_with_bounds(
            sim_hist=simulated_history,
            forecast=monthly_records.copy(),
            observed=observed_discharge,
        )
        fixed_records = pd.concat([fixed_records, corrected_values])

    fixed_records.sort_index(inplace=True)
    return fixed_records


def split_3h_and_1h_ensembles(corrected_ensemble_df: pd.DataFrame) -> dict[str, Any]:
    """Split corrected ensembles into 3-hour members and the hourly member."""
    valid_3h_mask = corrected_ensemble_df.iloc[:, :51].notna().any(axis=1)
    df_3h = corrected_ensemble_df.loc[valid_3h_mask, corrected_ensemble_df.columns[:51]]
    df_1h = corrected_ensemble_df.iloc[:, 51:]

    return {
        "time_3h": df_3h.index,
        "ens_3h": df_3h.to_numpy(dtype=np.float32),
        "time_1h": df_1h.index,
        "ens_1h": df_1h.to_numpy(dtype=np.float32),
    }


def download_bias_corrected_initialization_forecasts(
    river_id: int,
    observed_realtime: pd.DataFrame,
    simulated_history: pd.DataFrame,
    observed_discharge: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Download and bias-correct initialization forecasts for the observation window."""
    start_event = observed_realtime.index.min().strftime("%Y-%m-%d")
    end_event = observed_realtime.index.max().strftime("%Y-%m-%d")
    forecasts: dict[str, dict[str, Any]] = {}

    for day_date in pd.date_range(start=start_event, end=end_event, freq="D"):
        init_str = day_date.strftime("%Y%m%d")
        print(f"Downloading forecast for {init_str}")
        ensemble_df = normalize_datetime_index(
            geoglows.data.forecast_ensembles(river_id=river_id, date=init_str)
        )
        corrected_ensemble_df = bias_correct_forecast_with_bounds(
            sim_hist=simulated_history,
            forecast=ensemble_df,
            observed=observed_discharge,
        )
        forecasts[init_str] = split_3h_and_1h_ensembles(corrected_ensemble_df)

    return forecasts


def calculate_ar1_error_parameters(
    forecasts: dict[str, dict[str, Any]],
    observed_realtime: pd.DataFrame,
) -> tuple[pd.Series, float, float]:
    """Fit AR(1) parameters to high-resolution forecast errors."""
    from statsmodels.tsa.arima.model import ARIMA

    init_dates = sorted(forecasts.keys())
    init_times = pd.to_datetime(init_dates, format="%Y%m%d")

    records = []
    for t_obs, h_obs in observed_realtime.dropna().iloc[:, 0].items():
        valid = init_times <= t_obs
        if not valid.any():
            continue
        last_init = init_times[valid].max()
        forecast = forecasts[last_init.strftime("%Y%m%d")]
        m52 = pd.Series(forecast["ens_1h"].ravel(), index=forecast["time_1h"])
        if t_obs not in m52.index or pd.isna(m52.loc[t_obs]):
            continue
        records.append((t_obs, float(m52.loc[t_obs] - h_obs)))

    err = pd.Series(dict(records)).sort_index()
    if len(err) < 2:
        raise ValueError("At least two matched observation/forecast errors are required.")

    result = ARIMA(err.values, order=(1, 0, 0)).fit()
    return err, float(result.params[0]), float(result.params[1])


def apply_ar1_initial_condition_correction(
    corrected_ensembles: pd.DataFrame,
    forecasts: dict[str, dict[str, Any]],
    observed_realtime: pd.DataFrame,
) -> pd.DataFrame:
    """Apply an AR(1) initialization correction to corrected ensembles."""
    _, mu, phi = calculate_ar1_error_parameters(forecasts, observed_realtime)
    init_dates = sorted(forecasts.keys())
    init_times = pd.to_datetime(init_dates, format="%Y%m%d")

    observed_clean = observed_realtime.dropna()
    last_obs_time = observed_clean.index[-1]
    last_obs_value = float(observed_clean.iloc[-1, 0])
    init_for_current_error = init_times[init_times <= last_obs_time].max()
    forecast = forecasts[init_for_current_error.strftime("%Y%m%d")]
    m52_at_obs = pd.Series(
        forecast["ens_1h"].ravel(),
        index=forecast["time_1h"],
    ).loc[last_obs_time]
    current_error = float(m52_at_obs - last_obs_value)

    elapsed_hours = np.array(
        [(time - last_obs_time).total_seconds() / 3600 for time in corrected_ensembles.index]
    )
    error_hat = mu + (phi ** (elapsed_hours / 12.0)) * (current_error - mu)
    error_hat_df = pd.DataFrame(
        np.tile(error_hat.reshape(-1, 1), (1, corrected_ensembles.shape[1])),
        index=corrected_ensembles.index,
        columns=corrected_ensembles.columns,
    )
    return corrected_ensembles - error_hat_df


def run_workflow(
    river_id: int | str,
    station_id: str = DEFAULT_IDEAM_STATION_ID,
    ideam_variable: str = DEFAULT_IDEAM_VARIABLE,
    fews_url: str = DEFAULT_IDEAM_FEWS_URL,
    show_tables: bool = True,
    show_plots: bool = True,
) -> WorkflowResult:
    """Run the full bias-correction and initialization-adjustment workflow."""
    river_id = validate_river_id(river_id)
    latitude, longitude = get_reach_location(river_id)
    timezone_offset_hours = get_timezone_offset_hours(latitude, longitude)
    corrected_titles = [f"Reach ID: {river_id}"]

    location_figure = plot_reach_location(river_id, latitude, longitude)
    if show_plots:
        location_figure.show()

    daily_df, ensembles, stats_df, records = fetch_geoglows_data(river_id)
    simulated_return_periods = calculate_return_periods(daily_df, river_id)
    observed_discharge, df_obs, df_sen = fetch_observed_discharge(
        station_id=station_id,
        variable=ideam_variable,
        fews_url=fews_url,
        timezone_offset_hours=timezone_offset_hours,
    )

    raw_forecast_figure = plot_forecast_with_context(
        stats_df=stats_df,
        return_periods_df=simulated_return_periods,
        title="Forecast.",
        records=records,
        historical=daily_df,
        observed_discharge=observed_discharge,
        observed_realtime=df_obs,
    )
    if show_plots:
        raw_forecast_figure.show()

    corrected_historical = correct_historical(daily_df, observed_discharge)
    corrected_return_periods = calculate_return_periods(corrected_historical, river_id)
    corrected_ensembles = bias_correct_forecast_with_bounds(
        sim_hist=daily_df,
        forecast=ensembles,
        observed=observed_discharge,
    )
    corrected_stats = calculate_forecast_stats(corrected_ensembles)
    fixed_records = correct_forecast_records(records, daily_df, observed_discharge)

    corrected_forecast_figure = plot_forecast_with_context(
        stats_df=corrected_stats,
        return_periods_df=corrected_return_periods,
        title="Bias Corrected Forecast.",
        plot_titles=corrected_titles,
        records=fixed_records,
        historical=corrected_historical,
        observed_discharge=observed_discharge,
        observed_realtime=df_obs,
        historical_name="Bias Corrected Retrospective Simulation",
        value_precision=2,
    )
    if show_plots:
        corrected_forecast_figure.show()

    initialization_forecasts = download_bias_corrected_initialization_forecasts(
        river_id=river_id,
        observed_realtime=df_obs,
        simulated_history=daily_df,
        observed_discharge=observed_discharge,
    )
    corrected_ensembles_ar1 = apply_ar1_initial_condition_correction(
        corrected_ensembles=corrected_ensembles,
        forecasts=initialization_forecasts,
        observed_realtime=df_obs,
    )
    corrected_stats_ar1 = calculate_forecast_stats(corrected_ensembles_ar1)
    ar1_forecast_figure = plot_forecast_with_context(
        stats_df=corrected_stats_ar1,
        return_periods_df=corrected_return_periods,
        title="Transformed Water Level Forecast.",
        plot_titles=corrected_titles,
        historical=corrected_historical,
        observed_discharge=observed_discharge,
        observed_realtime=df_obs,
        historical_name="Bias Corrected Retrospective Simulation",
        value_precision=2,
    )
    if show_plots:
        ar1_forecast_figure.show()

    if show_tables:
        for df in (
            daily_df,
            simulated_return_periods,
            ensembles,
            stats_df,
            records,
            observed_discharge,
            df_obs,
            corrected_historical,
            corrected_return_periods,
            corrected_ensembles,
            corrected_stats,
            fixed_records,
            corrected_ensembles_ar1,
            corrected_stats_ar1,
        ):
            display_dataframe(df)

    return WorkflowResult(
        river_id=river_id,
        latitude=latitude,
        longitude=longitude,
        timezone_offset_hours=timezone_offset_hours,
        daily_df=daily_df,
        ensembles=ensembles,
        stats_df=stats_df,
        records=records,
        observed_discharge=observed_discharge,
        observed_realtime=df_obs,
        observed_sensors=df_sen,
        simulated_return_periods=simulated_return_periods,
        corrected_historical=corrected_historical,
        corrected_return_periods=corrected_return_periods,
        corrected_ensembles=corrected_ensembles,
        corrected_stats=corrected_stats,
        fixed_records=fixed_records,
        corrected_ensembles_ar1=corrected_ensembles_ar1,
        corrected_stats_ar1=corrected_stats_ar1,
        location_figure=location_figure,
        raw_forecast_figure=raw_forecast_figure,
        corrected_forecast_figure=corrected_forecast_figure,
        ar1_forecast_figure=ar1_forecast_figure,
    )


def main() -> WorkflowResult:
    """Prompt for a river ID and run the full workflow."""
    configure_notebook_output()
    river_id = input("Enter the river ID: ")
    return run_workflow(river_id)


if __name__ == "__main__":
    main()
