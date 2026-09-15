"""Channel registry and observation-vector layout."""

OBS_CHANNELS = [
    "pm1_kw",
    "pm0_a_kw",
    "pm0_b_kw",
    "irr1_wm2",
    "irr2_wm2",
    "irr1_temp_c",
    "irr2_temp_c",
    "sat_gti_wm2",
    "fcst_pv_t1h_kw",
    "fcst_pv_t3h_kw",
    "fcst_pv_t6h_kw",
    "tariff_zar_kwh",
    "demand_exposure_kw",
    "soc",
    "charge_headroom_kw",
    "discharge_headroom_kw",
    "sin_hod",
    "cos_hod",
    "sin_dow",
    "cos_dow",
    "sin_doy",
    "cos_doy",
]

SENSOR_IDX = list(range(8))
METER_IDX = [0, 1, 2]
PV_METER_IDX = [0]
GRID_METER_IDX = [1, 2]
IRR_IDX = [3, 4]
TEMP_IDX = [5, 6]
SAT_IDX = [7]
FCST_IDX = [8, 9, 10]
TARIFF_IDX = [11]
EXPOSURE_IDX = [12]
SOC_IDX = [13]
CHG_HEAD_IDX = [14]
DIS_HEAD_IDX = [15]
TIME_IDX = list(range(16, 22))

DATA_IDX = list(range(12)) + TIME_IDX
STATE_IDX = [EXPOSURE_IDX[0], SOC_IDX[0], CHG_HEAD_IDX[0], DIS_HEAD_IDX[0]]
WEATHER_IDX = SAT_IDX + FCST_IDX

RAW_CHANNEL_BOUNDS = {
    "pm0_a": 300.0,
    "pm0_b": 75.0,
    "pm1": 300.0,
}

IRR_PHYS_MAX = 1500.0
SENTINEL = -999.0
PLATFORM_SENTINEL = 888.89
