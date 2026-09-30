import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NSE Swing Trading Scanner",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NSE Top 500 Swing Trading Scanner")
st.caption(
    "8-Day Midpoint + ADR + Liquidity + RSI + EMA20/EMA50 + "
    "20D Rolling VWAP + Stochastic Crossover"
)

# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("Scanner Settings")

MIDPOINT_DAYS = st.sidebar.number_input(
    "Midpoint Days", min_value=2, max_value=30, value=8
)

ADR_DAYS = st.sidebar.number_input(
    "ADR Days", min_value=5, max_value=60, value=20
)

VOLUME_DAYS = st.sidebar.number_input(
    "Volume Days", min_value=5, max_value=100, value=30
)

st.sidebar.subheader("Existing Filters")

MIN_RANGE_ADR = st.sidebar.number_input(
    "Minimum Range / ADR", min_value=0.0, value=0.0, step=0.10
)

MIN_ADR_PCT = st.sidebar.number_input(
    "Minimum ADR %", min_value=0.0, value=0.0, step=0.10
)

MIN_RANGE_POS = st.sidebar.number_input(
    "Minimum 8D Range Position %", min_value=0.0, max_value=100.0,
    value=0.0, step=5.0
)

MAX_CLOSE = st.sidebar.number_input(
    "Maximum Close", min_value=1.0, value=50000.0, step=100.0
)

MIN_MCAP = st.sidebar.number_input(
    "Minimum Market Cap ₹ Cr", min_value=0.0, value=10.0, step=10.0
)

MIN_RVOL = st.sidebar.number_input(
    "Minimum Relative Volume", min_value=0.0, value=0.0, step=0.10
)

MIN_ADTV_CR = st.sidebar.number_input(
    "Minimum ADTV ₹ Cr", min_value=0.0, value=0.0, step=1.0
)

st.sidebar.subheader("Technical Filters")

USE_RSI_FILTER = st.sidebar.checkbox(
    "Use RSI Filter", value=True
)

RSI_PERIOD = st.sidebar.number_input(
    "RSI Period", min_value=2, max_value=50, value=14
)

MIN_RSI = st.sidebar.number_input(
    "Minimum RSI", min_value=0.0, max_value=100.0, value=50.0
)

USE_EMA_FILTER = st.sidebar.checkbox(
    "Use EMA20/EMA50 Filter", value=True
)

EMA_FAST = st.sidebar.number_input(
    "Fast EMA", min_value=2, max_value=100, value=20
)

EMA_SLOW = st.sidebar.number_input(
    "Slow EMA", min_value=5, max_value=200, value=50
)

USE_VWAP_FILTER = st.sidebar.checkbox(
    "Use 20D VWAP Filter", value=True
)

VWAP_DAYS = st.sidebar.number_input(
    "Rolling VWAP Days", min_value=2, max_value=100, value=20
)

USE_STOCH_FILTER = st.sidebar.checkbox(
    "Use Stochastic Crossover Filter", value=True
)

STOCH_K_PERIOD = st.sidebar.number_input(
    "Stochastic K Period", min_value=2, max_value=50, value=14
)

STOCH_K_SMOOTH = st.sidebar.number_input(
    "Stochastic K Smoothing", min_value=1, max_value=20, value=3
)

STOCH_D_PERIOD = st.sidebar.number_input(
    "Stochastic D Period", min_value=1, max_value=20, value=3
)

USE_STOCH_OVERSOLD = st.sidebar.checkbox(
    "Require Stoch Cross Below Level", value=False
)

STOCH_OVERSOLD_LEVEL = st.sidebar.number_input(
    "Maximum Stoch %K for Cross",
    min_value=1.0,
    max_value=100.0,
    value=30.0
)

# ============================================================
# SIGNAL SETTINGS
# ============================================================

st.sidebar.subheader("Signal Settings")

STRONG_BUY_SCORE = st.sidebar.slider(
    "STRONG BUY Minimum Score", 1, 4, 4
)

BUY_SCORE = st.sidebar.slider(
    "BUY Minimum Score", 1, 4, 3
)

WATCH_SCORE = st.sidebar.slider(
    "WATCH Minimum Score", 0, 4, 1
)

# ============================================================
# LOAD MASTER
# ============================================================

@st.cache_data(ttl=3600)
def load_master():

    master = pd.read_csv(
        "ind_nifty500list_1.csv"
    )

    master["Symbol"] = (
        master["Symbol"]
        .astype(str)
        .str.strip()
    )

    return master


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# ============================================================
# STOCHASTIC
# ============================================================

def calculate_stochastic(
    high_series,
    low_series,
    close_series,
    k_period,
    k_smooth,
    d_period
):

    lowest_low = (
        low_series
        .rolling(k_period)
        .min()
    )

    highest_high = (
        high_series
        .rolling(k_period)
        .max()
    )

    price_range = (
        highest_high - lowest_low
    )

    raw_k = (
        (close_series - lowest_low)
        / price_range
    ) * 100

    stoch_k = (
        raw_k
        .rolling(k_smooth)
        .mean()
    )

    stoch_d = (
        stoch_k
        .rolling(d_period)
        .mean()
    )

    return stoch_k, stoch_d


# ============================================================
# RUN SCANNER
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def run_scanner(
    midpoint_days,
    adr_days,
    volume_days,
    min_range_adr,
    min_adr_pct,
    min_range_pos,
    max_close,
    min_mcap,
    min_rvol,
    min_adtv_cr,
    use_rsi,
    rsi_period,
    min_rsi,
    use_ema,
    ema_fast,
    ema_slow,
    use_vwap,
    vwap_days,
    use_stoch,
    stoch_k_period,
    stoch_k_smooth,
    stoch_d_period,
    use_stoch_oversold,
    stoch_oversold_level,
    strong_buy_score,
    buy_score,
    watch_score
):

    master = load_master()

    symbol_col = "Symbol"
    name_col = "Company"
    industry_col = "Industry"
    sector_col = "Sector"
    mcap_col = "MCapCr"

    tickers = (
        master["Symbol"] + ".NS"
    ).tolist()

    data = yf.download(
        tickers=tickers,
        period="15mo",
        interval="1d",
        group_by="column",
        auto_adjust=False,
        threads=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    volume = data["Volume"]

    results = []

    today = pd.Timestamp.now().normalize()

    for ticker in tickers:

        try:

            symbol = ticker.replace(".NS", "")

            stock_close = close[ticker].dropna()

            if len(stock_close) < 252:
                continue

            completed_close = (
                stock_close[
                    stock_close.index.normalize() < today
                ]
            )

            if len(completed_close) < 252:
                continue

            last_date = completed_close.index[-1]
            last_c = completed_close.iloc[-1]

            stock_high = (
                high[ticker]
                .dropna()
                .loc[:last_date]
            )

            stock_low = (
                low[ticker]
                .dropna()
                .loc[:last_date]
            )

            stock_volume = (
                volume[ticker]
                .dropna()
                .loc[:last_date]
            )

            if len(stock_high) < 252:
                continue

            last_h = stock_high.iloc[-1]
            last_l = stock_low.iloc[-1]

            # ------------------------------------------------
            # 8D RANGE
            # ------------------------------------------------

            stock_high_8 = stock_high.tail(
                midpoint_days
            ).max()

            stock_low_8 = stock_low.tail(
                midpoint_days
            ).min()

            midpoint = (
                stock_high_8 + stock_low_8
            ) / 2

            range_size = (
                stock_high_8 - stock_low_8
            )

            if range_size <= 0:
                continue

            range_position = (
                (last_c - stock_low_8)
                / range_size
            ) * 100

            midpoint_distance_pct = (
                (last_c - midpoint)
                / midpoint
            ) * 100

            # ------------------------------------------------
            # ADR
            # ------------------------------------------------

            stock_range = (
                stock_high - stock_low
            )

            stock_adr = (
                stock_range
                .tail(adr_days)
                .mean()
            )

            stock_adr_pct = (
                (
                    stock_range / completed_close
                ) * 100
            ).tail(adr_days).mean()

            if stock_adr <= 0:
                continue

            last_range = (
                last_h - last_l
            )

            last_range_pct = (
                last_range / last_c
            ) * 100

            range_vs_adr = (
                last_range / stock_adr
            )

            # ------------------------------------------------
            # 52 WEEK RANGE
            # ------------------------------------------------

            previous_high = (
                stock_high
                .iloc[:-1]
                .tail(252)
            )

            previous_low = (
                stock_low
                .iloc[:-1]
                .tail(252)
            )

            if (
                len(previous_high) < 252
                or len(previous_low) < 252
            ):
                continue

            week52_high = previous_high.max()
            week52_low = previous_low.min()

            week52_range = (
                week52_high - week52_low
            )

            if week52_range > 0:

                range_position_52w = (
                    (last_c - week52_low)
                    / week52_range
                ) * 100

            else:
                range_position_52w = np.nan

            # ------------------------------------------------
            # VOLUME
            # ------------------------------------------------

            completed_volume = stock_volume[
                stock_volume.index.normalize() < today
            ]

            if len(completed_volume) < volume_days + 1:
                continue

            current_volume = completed_volume.iloc[-1]

            previous_volumes = (
                completed_volume
                .iloc[:-1]
                .tail(volume_days)
            )

            volume_30_avg = previous_volumes.mean()

            if volume_30_avg <= 0:
                continue

            relative_volume = (
                current_volume
                / volume_30_avg
            )

            volume_change_pct = (
                relative_volume - 1
            ) * 100

            adtv_cr = (
                last_c * volume_30_avg
            ) / 10_000_000

            if adtv_cr < 2:
                liquidity = "VERY LOW"
            elif adtv_cr < 10:
                liquidity = "LOW"
            elif adtv_cr < 25:
                liquidity = "ACCEPTABLE"
            elif adtv_cr < 100:
                liquidity = "GOOD"
            else:
                liquidity = "VERY GOOD"

            # ------------------------------------------------
            # MASTER DATA
            # ------------------------------------------------

            info = master[
                master[symbol_col] == symbol
            ]

            if info.empty:
                continue

            company = info[name_col].iloc[0]
            sector = info[sector_col].iloc[0]
            industry = info[industry_col].iloc[0]
            mcap = info[mcap_col].iloc[0]

            if pd.isna(mcap):
                continue

            # ------------------------------------------------
            # RSI
            # ------------------------------------------------

            rsi_series = calculate_rsi(
                completed_close,
                rsi_period
            )

            rsi = rsi_series.iloc[-1]

            rsi_pass = (
                rsi >= min_rsi
            )

            # ------------------------------------------------
            # EMA
            # ------------------------------------------------

            ema20_series = (
                completed_close
                .ewm(
                    span=ema_fast,
                    adjust=False
                )
                .mean()
            )

            ema50_series = (
                completed_close
                .ewm(
                    span=ema_slow,
                    adjust=False
                )
                .mean()
            )

            ema20 = ema20_series.iloc[-1]
            ema50 = ema50_series.iloc[-1]

            close_above_ema20 = (
                last_c > ema20
            )

            ema20_above_ema50 = (
                ema20 > ema50
            )

            ema_pass = (
                close_above_ema20
                and ema20_above_ema50
            )

            # ------------------------------------------------
            # ROLLING VWAP
            # ------------------------------------------------

            pv = (
                completed_close
                * completed_volume
            )

            rolling_pv = (
                pv
                .rolling(vwap_days)
                .sum()
            )

            rolling_volume = (
                completed_volume
                .rolling(vwap_days)
                .sum()
            )

            vwap = (
                rolling_pv
                / rolling_volume
            ).iloc[-1]

            close_above_vwap = (
                last_c > vwap
            )

            # ------------------------------------------------
            # STOCHASTIC
            # ------------------------------------------------

            stoch_k_series, stoch_d_series = (
                calculate_stochastic(
                    stock_high,
                    stock_low,
                    completed_close,
                    stoch_k_period,
                    stoch_k_smooth,
                    stoch_d_period
                )
            )

            if (
                len(stoch_k_series.dropna()) < 2
                or len(stoch_d_series.dropna()) < 2
            ):
                continue

            stoch_k = stoch_k_series.iloc[-1]
            stoch_d = stoch_d_series.iloc[-1]

            previous_k = stoch_k_series.iloc[-2]
            previous_d = stoch_d_series.iloc[-2]

            bullish_cross = (
                previous_k <= previous_d
                and stoch_k > stoch_d
            )

            if use_stoch_oversold:

                stoch_pass = (
                    bullish_cross
                    and stoch_k <= stoch_oversold_level
                )

            else:

                stoch_pass = bullish_cross

            # ------------------------------------------------
            # TECHNICAL SCORE
            # ------------------------------------------------

            score = 0

            if rsi_pass:
                score += 1

            if ema_pass:
                score += 1

            if close_above_vwap:
                score += 1

            if stoch_pass:
                score += 1

            # ------------------------------------------------
            # SIGNAL CATEGORY
            # ------------------------------------------------
            #
            # STRONG BUY = 4/4
            # BUY        = 3/4
            # WATCH      = 1-2/4
            # NO SIGNAL  = 0/4
            #
            # User can adjust thresholds in sidebar.
            # ------------------------------------------------

            if score >= strong_buy_score:
                signal = "STRONG BUY"
            elif score >= buy_score:
                signal = "BUY"
            elif score >= watch_score:
                signal = "WATCH"
            else:
                signal = "NO SIGNAL"

            # ------------------------------------------------
            # EXISTING FILTERS
            # ------------------------------------------------

            existing_pass = (

                last_c <= max_close

                and mcap > min_mcap

                and range_position >= min_range_pos

                and range_vs_adr >= min_range_adr

                and stock_adr_pct >= min_adr_pct

                and relative_volume >= min_rvol

                and adtv_cr >= min_adtv_cr
            )

            # ------------------------------------------------
            # TECHNICAL FILTERS
            # ------------------------------------------------

            technical_pass = (

                (rsi_pass if use_rsi else True)

                and (ema_pass if use_ema else True)

                and (
                    close_above_vwap
                    if use_vwap
                    else True
                )

                and (
                    stoch_pass
                    if use_stoch
                    else True
                )
            )

            # Final scanner result
            if not existing_pass:
                continue

            if not technical_pass:
                continue

            results.append({

                "Ticker": symbol,
                "Company": company,
                "Sector": sector,
                "Industry": industry,
                "MCap Cr": mcap,

                "Last Date": last_date.strftime(
                    "%Y-%m-%d"
                ),

                "Last": last_c,

                "Signal": signal,
                "Technical Score": score,

                "RSI 14": rsi,

                "EMA 20": ema20,
                "EMA 50": ema50,

                "Close > EMA20":
                    "YES" if close_above_ema20 else "NO",

                "EMA20 > EMA50":
                    "YES" if ema20_above_ema50 else "NO",

                "VWAP 20D": vwap,

                "Close > VWAP":
                    "YES" if close_above_vwap else "NO",

                "Stoch %K": stoch_k,
                "Stoch %D": stoch_d,

                "Stoch Bull Cross":
                    "YES" if bullish_cross else "NO",

                "8D High": stock_high_8,
                "8D Low": stock_low_8,
                "8D Midpoint": midpoint,
                "8D Range Pos %": range_position,

                "52W High": week52_high,
                "52W Low": week52_low,
                "52W Range Pos %":
                    range_position_52w,

                "Above Mid %":
                    midpoint_distance_pct,

                "Last Range": last_range,
                "Last Range %": last_range_pct,

                "ADR 20": stock_adr,
                "ADR 20 %": stock_adr_pct,
                "Range / ADR": range_vs_adr,

                "Volume": current_volume,
                "30D Avg Volume": volume_30_avg,
                "Relative Volume": relative_volume,
                "Volume vs 30D %":
                    volume_change_pct,

                "ADTV Cr": adtv_cr,
                "Liquidity": liquidity
            })

        except Exception:
            continue

    return pd.DataFrame(results)


# ============================================================
# RUN BUTTON
# ============================================================

if "scan_clicked" not in st.session_state:
    st.session_state.scan_clicked = False

run_scan = st.sidebar.button(
    "🔍 RUN SCANNER",
    type="primary",
    use_container_width=True
)

if run_scan:
    st.cache_data.clear()
    st.session_state.scan_clicked = True

if not st.session_state.scan_clicked:

    st.info(
        "Configure the filters in the sidebar and click "
        "**RUN SCANNER**."
    )

    st.stop()


# ============================================================
# RUN
# ============================================================

with st.spinner(
    "Downloading NSE data and running scanner..."
):

    result = run_scanner(
        MIDPOINT_DAYS,
        ADR_DAYS,
        VOLUME_DAYS,
        MIN_RANGE_ADR,
        MIN_ADR_PCT,
        MIN_RANGE_POS,
        MAX_CLOSE,
        MIN_MCAP,
        MIN_RVOL,
        MIN_ADTV_CR,
        USE_RSI_FILTER,
        RSI_PERIOD,
        MIN_RSI,
        USE_EMA_FILTER,
        EMA_FAST,
        EMA_SLOW,
        USE_VWAP_FILTER,
        VWAP_DAYS,
        USE_STOCH_FILTER,
        STOCH_K_PERIOD,
        STOCH_K_SMOOTH,
        STOCH_D_PERIOD,
        USE_STOCH_OVERSOLD,
        STOCH_OVERSOLD_LEVEL,
        STRONG_BUY_SCORE,
        BUY_SCORE,
        WATCH_SCORE
    )


# ============================================================
# EMPTY RESULT
# ============================================================

if result.empty:

    st.warning(
        "No stocks met the selected filters."
    )

    st.stop()


# ============================================================
# TOP METRICS
# ============================================================

strong_buy_count = (
    result["Signal"] == "STRONG BUY"
).sum()

buy_count = (
    result["Signal"] == "BUY"
).sum()

watch_count = (
    result["Signal"] == "WATCH"
).sum()

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Qualifying Stocks",
    len(result)
)

c2.metric(
    "STRONG BUY",
    strong_buy_count
)

c3.metric(
    "BUY",
    buy_count
)

c4.metric(
    "WATCH",
    watch_count
)


# ============================================================
# SIGNAL FILTER
# ============================================================

st.subheader("Signal Filter")

signal_options = [
    "ALL",
    "STRONG BUY",
    "BUY",
    "WATCH"
]

selected_signal = st.selectbox(
    "Show",
    signal_options
)

display_df = result.copy()

if selected_signal != "ALL":

    display_df = display_df[
        display_df["Signal"]
        == selected_signal
    ].copy()


# ============================================================
# ADD SEARCH
# ============================================================

search = st.text_input(
    "🔎 Search ticker / company / sector / industry"
)

if search:

    search_lower = search.lower()

    mask = (
        display_df["Ticker"]
        .astype(str)
        .str.lower()
        .str.contains(search_lower, na=False)

        |

        display_df["Company"]
        .astype(str)
        .str.lower()
        .str.contains(search_lower, na=False)

        |

        display_df["Sector"]
        .astype(str)
        .str.lower()
        .str.contains(search_lower, na=False)

        |

        display_df["Industry"]
        .astype(str)
        .str.lower()
        .str.contains(search_lower, na=False)
    )

    display_df = display_df[mask]


# ============================================================
# SORT
# ============================================================

sort_column = st.selectbox(
    "Sort By",
    [
        "Signal",
        "Technical Score",
        "RSI 14",
        "8D Range Pos %",
        "52W Range Pos %",
        "Relative Volume",
        "ADTV Cr",
        "Above Mid %",
        "MCap Cr"
    ]
)

sort_desc = st.checkbox(
    "Descending",
    value=True
)

display_df = display_df.sort_values(
    sort_column,
    ascending=not sort_desc
)


# ============================================================
# FORMAT
# ============================================================

numeric_cols = [
    "MCap Cr",
    "Last",
    "Technical Score",
    "RSI 14",
    "EMA 20",
    "EMA 50",
    "VWAP 20D",
    "Stoch %K",
    "Stoch %D",
    "8D High",
    "8D Low",
    "8D Midpoint",
    "8D Range Pos %",
    "52W High",
    "52W Low",
    "52W Range Pos %",
    "Above Mid %",
    "Last Range",
    "Last Range %",
    "ADR 20",
    "ADR 20 %",
    "Range / ADR",
    "Volume",
    "30D Avg Volume",
    "Relative Volume",
    "Volume vs 30D %",
    "ADTV Cr"
]

for col in numeric_cols:

    if col in display_df.columns:

        display_df[col] = (
            pd.to_numeric(
                display_df[col],
                errors="coerce"
            ).round(2)
        )


# ============================================================
# COLUMN SELECTION
# ============================================================

default_columns = [

    "Ticker",
    "Company",
    "Sector",
    "Industry",
    "MCap Cr",
    "Last Date",
    "Last",

    "Signal",
    "Technical Score",

    "RSI 14",
    "EMA 20",
    "EMA 50",
    "VWAP 20D",

    "Stoch %K",
    "Stoch %D",
    "Stoch Bull Cross",

    "8D Range Pos %",
    "52W Range Pos %",
    "Above Mid %",

    "ADR 20 %",
    "Range / ADR",

    "Relative Volume",
    "ADTV Cr",
    "Liquidity"
]

available_defaults = [
    c for c in default_columns
    if c in display_df.columns
]

selected_columns = st.multiselect(
    "Columns to Display",
    options=list(display_df.columns),
    default=available_defaults
)

if selected_columns:

    table_df = display_df[
        selected_columns
    ].copy()

else:

    table_df = display_df.copy()


# ============================================================
# STYLING
# ============================================================

def highlight_signal(row):

    styles = [
        ""
    ] * len(row)

    if "Signal" in row.index:

        idx = row.index.get_loc(
            "Signal"
        )

        signal = row["Signal"]

        if signal == "STRONG BUY":
            styles[idx] = (
                "font-weight: bold;"
            )

        elif signal == "BUY":
            styles[idx] = (
                "font-weight: bold;"
            )

        elif signal == "WATCH":
            styles[idx] = (
                "font-weight: bold;"
            )

    return styles


styled_df = table_df.style.apply(
    highlight_signal,
    axis=1
)


# ============================================================
# TABLE
# ============================================================

st.subheader(
    f"Scanner Results — {len(table_df)} stocks"
)

st.dataframe(
    styled_df,
    use_container_width=True,
    height=650,
    hide_index=True
)


# ============================================================
# DOWNLOAD
# ============================================================

csv_data = display_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇️ Download Results CSV",
    data=csv_data,
    file_name="NSE_Swing_Trading_Scanner.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# SIGNAL LOGIC
# ============================================================

with st.expander(
    "📌 Signal Logic"
):

    st.markdown(
        """
### Technical Score

Each condition contributes **1 point**:

- **RSI:** RSI(14) >= selected minimum
- **EMA Trend:** Close > EMA20 > EMA50
- **VWAP:** Close > 20-day rolling VWAP
- **Stochastic:** Fresh bullish %K crossover above %D

### Default Signals

| Technical Score | Signal |
|---:|---|
| 4 / 4 | **STRONG BUY** |
| 3 / 4 | **BUY** |
| 1–2 / 4 | **WATCH** |
| 0 / 4 | NO SIGNAL |

The signal is a technical classification based on the configured rules; it is not a guarantee of future performance.
"""
    )
