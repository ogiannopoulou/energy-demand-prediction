"""
Italian Electricity Data Ingestion
Uses ENTSO-E Transparency Platform API (real data) with synthetic fallback
"""
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ITALY_AREA = "10YIT-GRTN-----B"


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class EntsoeClient:
    """ENTSO-E Transparency Platform API client."""

    BASE_URL = "https://web-api.tp.entsoe.eu/api"

    # Document types
    DOC_ACTUAL_LOAD = "A65"      # Actual total system load
    DOC_FORECAST_LOAD = "A65"    # Day-ahead load forecast
    DOC_ACTUAL_GEN = "A75"       # Actual generation per type
    DOC_FORECAST_GEN = "A75"     # Wind/solar generation forecast

    # Process types
    PROC_ACTUAL = "A16"          # Realised
    PROC_FORECAST = "A01"        # Day-ahead

    # Generation type codes (PsrType)
    GEN_THERMAL = "B01"
    GEN_GAS = "B02"
    GEN_NUCLEAR = "B03"
    GEN_WIND = "B04"
    GEN_SOLAR = "B05"
    GEN_HYDRO = "B06"
    GEN_GEOTHERMAL = "B07"
    GEN_BIOMASS = "B08"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ENTSOE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ENTSO-E API key required. Set ENTSOE_API_KEY env var "
                "or get one at https://transparency.entsoe.eu/"
            )

    def _query(self, params: dict) -> str:
        """Make API request and return XML response."""
        params["securityToken"] = self.api_key
        resp = requests.get(self.BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        return resp.text

    def get_actual_load(self, start: str, end: str) -> pd.DataFrame:
        """
        Get actual total system load.

        Args:
            start: Start datetime (YYYYMMDDHHMM)
            end: End datetime (YYYYMMDDHHMM)
        """
        xml = self._query({
            "documentType": self.DOC_ACTUAL_LOAD,
            "processType": self.PROC_ACTUAL,
            "outBiddingZone_Domain": ITALY_AREA,
            "periodStart": start,
            "periodEnd": end,
        })
        return self._parse_load_xml(xml)

    def get_forecast_load(self, start: str, end: str) -> pd.DataFrame:
        """Get day-ahead load forecast."""
        xml = self._query({
            "documentType": self.DOC_FORECAST_LOAD,
            "processType": self.PROC_FORECAST,
            "outBiddingZone_Domain": ITALY_AREA,
            "periodStart": start,
            "periodEnd": end,
        })
        return self._parse_load_xml(xml)

    def get_actual_generation(self, start: str, end: str, psr_type: str = None) -> pd.DataFrame:
        """
        Get actual generation by type.

        Args:
            start: Start datetime (YYYYMMDDHHMM)
            end: End datetime (YYYYMMDDHHMM)
            psr_type: Optional generation type filter (e.g., B04 for wind)
        """
        params = {
            "documentType": self.DOC_ACTUAL_GEN,
            "processType": self.PROC_ACTUAL,
            "in_Domain": ITALY_AREA,
            "periodStart": start,
            "periodEnd": end,
        }
        if psr_type:
            params["psrType"] = psr_type

        xml = self._query(params)
        return self._parse_generation_xml(xml)

    def _parse_load_xml(self, xml: str) -> pd.DataFrame:
        """Parse ENTSO-E XML response for load data."""
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)

        # Detect namespace from root tag
        ns_uri = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
        ns = {"ns": ns_uri} if ns_uri else {}
        ns_prefix = "ns:" if ns else ""

        records = []

        for ts in root.findall(f".//{ns_prefix}TimeSeries", ns):
            for period in ts.findall(f".//{ns_prefix}Period", ns):
                resolution = period.find(f"{ns_prefix}resolution", ns)
                if resolution is not None:
                    res = resolution.text
                else:
                    res = "PT60M"

                start_str = period.find(f"{ns_prefix}timeInterval/{ns_prefix}start", ns)
                if start_str is None:
                    continue

                start_dt = pd.to_datetime(start_str.text)

                for point in period.findall(f".//{ns_prefix}Point", ns):
                    pos = int(point.find(f"{ns_prefix}position", ns).text)
                    val = point.find(f"{ns_prefix}quantity", ns)

                    if val is not None and val.text:
                        dt = start_dt + timedelta(minutes=(pos - 1) * self._resolution_minutes(res))
                        records.append({
                            "datetime": dt,
                            "value": float(val.text),
                        })

        if not records:
            return pd.DataFrame(columns=["demand_mw"])

        df = pd.DataFrame(records).set_index("datetime").sort_index()
        df.columns = ["demand_mw"]
        return df

    def _parse_generation_xml(self, xml: str) -> pd.DataFrame:
        """Parse ENTSO-E XML response for generation data."""
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml)

        # Detect namespace from root tag
        ns_uri = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
        ns = {"ns": ns_uri} if ns_uri else {}
        ns_prefix = "ns:" if ns else ""

        gen_type_map = {
            "B01": "thermal_mw",
            "B02": "gas_mw",
            "B03": "nuclear_mw",
            "B04": "wind_mw",
            "B05": "solar_mw",
            "B06": "hydro_mw",
            "B07": "geothermal_mw",
            "B08": "biomass_mw",
        }

        all_records = {}

        for ts in root.findall(f".//{ns_prefix}TimeSeries", ns):
            psr = ts.find(f".//{ns_prefix}psrType", ns)
            if psr is None:
                continue

            gen_type = psr.text
            col_name = gen_type_map.get(gen_type, f"gen_{gen_type}_mw")

            for period in ts.findall(f".//{ns_prefix}Period", ns):
                resolution = period.find(f"{ns_prefix}resolution", ns)
                res = resolution.text if resolution is not None else "PT60M"

                start_str = period.find(f"{ns_prefix}timeInterval/{ns_prefix}start", ns)
                if start_str is None:
                    continue

                start_dt = pd.to_datetime(start_str.text)

                for point in period.findall(f".//{ns_prefix}Point", ns):
                    pos = int(point.find(f"{ns_prefix}position", ns).text)
                    val = point.find(f"{ns_prefix}quantity", ns)

                    if val is not None and val.text:
                        dt = start_dt + timedelta(minutes=(pos - 1) * self._resolution_minutes(res))

                        if dt not in all_records:
                            all_records[dt] = {"datetime": dt}

                        all_records[dt][col_name] = float(val.text)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records.values()).set_index("datetime").sort_index()
        return df

    @staticmethod
    def _resolution_minutes(res: str) -> int:
        """Convert ENTSO-E resolution string to minutes."""
        mapping = {
            "PT1M": 1, "PT5M": 5, "PT15M": 15, "PT30M": 30,
            "PT60M": 60, "P1D": 1440, "P1W": 10080,
        }
        return mapping.get(res, 60)


def _format_entsoe_dt(dt: datetime) -> str:
    """Format datetime for ENTSO-E API (YYYYMMDDHHMM)."""
    return dt.strftime("%Y%m%d%H%M")


def download_entsoe_data(dataset: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download data from ENTSO-E with caching."""
    ensure_dirs()

    cache_path = RAW_DIR / f"{dataset}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"

    if cache_path.exists():
        logger.info("Loading cached data from %s", cache_path)
        return pd.read_parquet(cache_path)

    try:
        client = EntsoeClient()

        start_str = _format_entsoe_dt(start)
        end_str = _format_entsoe_dt(end)

        if dataset == "demand":
            df = client.get_actual_load(start_str, end_str)
        elif dataset == "demand_forecast":
            df = client.get_forecast_load(start_str, end_str)
        elif dataset == "generation":
            df = client.get_actual_generation(start_str, end_str)
        elif dataset == "wind":
            df = client.get_actual_generation(start_str, end_str, psr_type=EntsoeClient.GEN_WIND)
        elif dataset == "solar":
            df = client.get_actual_generation(start_str, end_str, psr_type=EntsoeClient.GEN_SOLAR)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if not df.empty:
            df.to_parquet(cache_path)
            logger.info("Saved %d rows to %s", len(df), cache_path)
            return df

        raise ValueError("API returned empty data")

    except Exception as e:
        logger.error("ENTSO-E API failed: %s", e)
        logger.warning("Generating synthetic data...")
        return generate_synthetic_data(dataset)


def generate_synthetic_data(dataset: str) -> pd.DataFrame:
    """Generate realistic synthetic Italian electricity data."""
    np.random.seed(42)

    dates = pd.date_range("2022-01-01", "2024-12-31", freq="h")
    n = len(dates)

    if dataset in ("demand", "demand_forecast"):
        hour = dates.hour
        dayofweek = dates.dayofweek
        month = dates.month

        base = 30000
        seasonal = 5000 * np.sin(2 * np.pi * (month - 1) / 12)
        daily = 8000 * np.sin(2 * np.pi * (hour - 6) / 24)
        weekly = 2000 * (dayofweek < 5).astype(float)
        noise = np.random.normal(0, 1500, n)

        demand = base + seasonal + daily + weekly + noise
        demand = np.maximum(demand, 15000)

        df = pd.DataFrame({
            "demand_mw": np.round(demand, 2)
        }, index=dates)

    elif dataset == "generation":
        hour = dates.hour
        month = dates.month

        thermal = 18000 + 4000 * np.sin(2 * np.pi * (month - 1) / 12) + \
                  3000 * np.sin(2 * np.pi * (hour - 12) / 24) + np.random.normal(0, 1000, n)

        solar = np.maximum(8000 * np.sin(2 * np.pi * (hour - 6) / 24) * (month > 3) * (month < 10), 0)
        solar += np.random.normal(0, 500, n)
        solar = np.maximum(solar, 0)

        wind = 3000 + 2000 * np.sin(2 * np.pi * np.arange(n) / (24 * 7))
        wind += np.random.normal(0, 800, n)
        wind = np.maximum(wind, 0)

        hydro = 4000 + 1000 * np.sin(2 * np.pi * (month - 1) / 12)
        hydro += np.random.normal(0, 300, n)

        df = pd.DataFrame({
            "thermal_mw": np.round(thermal, 2),
            "solar_mw": np.round(solar, 2),
            "wind_mw": np.round(wind, 2),
            "hydro_mw": np.round(hydro, 2),
        }, index=dates)

    elif dataset in ("wind", "solar"):
        hour = dates.hour
        month = dates.month

        if dataset == "wind":
            values = 3000 + 2000 * np.sin(2 * np.pi * np.arange(n) / (24 * 7))
            values += np.random.normal(0, 800, n)
            values = np.maximum(values, 0)
            col = "wind_mw"
        else:
            values = np.maximum(8000 * np.sin(2 * np.pi * (hour - 6) / 24) * (month > 3) * (month < 10), 0)
            values += np.random.normal(0, 500, n)
            values = np.maximum(values, 0)
            col = "solar_mw"

        df = pd.DataFrame({
            col: np.round(values, 2)
        }, index=dates)

    elif dataset == "frequency":
        base_freq = 50.0
        noise = np.random.normal(0, 0.02, n)
        spikes = np.random.choice(n, size=int(n * 0.001), replace=False)
        noise[spikes] += np.random.choice([-1, 1], size=len(spikes)) * np.random.uniform(0.1, 0.3, len(spikes))

        df = pd.DataFrame({
            "frequency_hz": np.round(base_freq + noise, 4)
        }, index=dates)

    else:
        raise ValueError(f"Unknown synthetic dataset: {dataset}")

    df.index.name = "datetime"

    cache_path = RAW_DIR / f"{dataset}_synthetic.csv"
    df.to_csv(cache_path)
    logger.info("Generated synthetic data saved to %s", cache_path)

    return df


def load_demand_data() -> pd.DataFrame:
    """Load Italian electricity demand data."""
    try:
        df = download_entsoe_data(
            "demand",
            datetime(2023, 1, 1),
            datetime(2024, 12, 31),
        )
    except Exception:
        df = generate_synthetic_data("demand")

    df = df[["demand_mw"]].dropna()
    return df


def load_generation_data() -> pd.DataFrame:
    """Load Italian electricity generation data by source."""
    try:
        df = download_entsoe_data(
            "generation",
            datetime(2023, 1, 1),
            datetime(2024, 12, 31),
        )
    except Exception:
        df = generate_synthetic_data("generation")

    cols = [c for c in ["thermal_mw", "solar_mw", "wind_mw", "hydro_mw"] if c in df.columns]
    df = df[cols].dropna()

    return df


def load_frequency_data() -> pd.DataFrame:
    """Load grid frequency data (synthetic — not available from ENTSO-E)."""
    return generate_synthetic_data("frequency")


def load_all_data() -> dict:
    """Load all available datasets."""
    return {
        "demand": load_demand_data(),
        "generation": load_generation_data(),
        "frequency": load_frequency_data(),
    }


def merge_demand_generation() -> pd.DataFrame:
    """Merge demand and generation data into a single DataFrame."""
    demand = load_demand_data()
    generation = load_generation_data()

    merged = demand.join(generation, how="inner")

    if "thermal_mw" in merged.columns:
        gen_cols = [c for c in merged.columns if c.endswith("_mw") and c != "demand_mw"]
        merged["total_generation_mw"] = merged[gen_cols].sum(axis=1)
        merged["renewable_mw"] = merged[[c for c in ["solar_mw", "wind_mw", "hydro_mw"] if c in merged.columns]].sum(axis=1)
        merged["renewable_share"] = merged["renewable_mw"] / merged["total_generation_mw"]

    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing data ingestion...")

    demand = load_demand_data()
    logger.info("Demand: %s", demand.shape)
    logger.info("Date range: %s to %s", demand.index.min(), demand.index.max())
    print(demand.head())

    generation = load_generation_data()
    logger.info("Generation: %s", generation.shape)
    print(generation.head())

    merged = merge_demand_generation()
    logger.info("Merged: %s", merged.shape)
    print(merged.head())
