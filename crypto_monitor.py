import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from textblob import TextBlob
import smtplib
from email.mime.text import MIMEText
import json # For JSONDecodeError handling
import time # For cooldown mechanisms
from translations import TRANSLATIONS


# Initialize session state for language if not already set.
if 'language' not in st.session_state:
    st.session_state.language = 'English'


def t(key):
    """
    Fetches a translation string for the given key based on the current session language.
    Falls back to English if the current language or key is not found.
    """
    lang = st.session_state.get('language', 'English')
    return TRANSLATIONS.get(lang, TRANSLATIONS['English']).get(key, key)


# --- Custom Metric Function ---
def custom_metric(label, value, delta, help_text=None):
    """
    Displays a styled metric card, similar to st.metric but with more custom styling.
    Shows a label, a main value, and a color-coded delta (change).
    """
    try:
        # Convert delta string (e.g., '+5%') to a float for styling.
        delta_value = float(str(delta).replace('%','').replace('+',''))
    except Exception:
        delta_value = 0 # Default to no change if parsing fails.

    # Determine color and arrow icon based on the delta's sign.
    if delta_value > 0:
        color, arrow, sign = '#4CAF50', '▲', '+' # Green for positive
    elif delta_value < 0:
        color, arrow, sign = '#F44336', '▼', ''  # Red for negative
    else: # No change
        color, arrow, sign = '#888', '', ''     # Gray for neutral

    # Construct HTML for the delta part, only if delta is meaningful.
    delta_html = (
        f'<div style="font-size: 1.2em; font-weight: bold; color: {color}; margin-top: 0.2em;">'
        f'{arrow} {sign}{delta}'
        f'</div>'
        if delta and str(delta).strip() not in ['', '</div>'] else ''
    )
    
    # Main HTML structure for the custom metric card.
    html = f'''
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1.2em 0; border-radius: 12px; background: var(--card-background, #23272f); margin-bottom: 1.2em; min-width: 180px;">
        <div style="font-size: 1.1em; color: var(--card-label-color, #b0b8c1);">{label}</div>
        <div style="font-size: 2em; font-weight: bold; color: var(--text-color, #fafafa);">{value}</div>
        {delta_html}
    </div>
    '''
    if help_text:
        # Wrap with a title attribute for a tooltip if help_text is provided.
        html = f'<div title="{help_text}">{html}</div>'
    st.markdown(html, unsafe_allow_html=True)


# --- Correlation Analysis ---
# Note: This section imports libraries specific to its functionality.
import plotly.express as px
from sklearn.linear_model import LinearRegression


def show_correlation_analysis(coin_list_dict, vs_currency):
    """
    Renders the 'Correlation Analysis' page.
    Displays a heatmap showing the correlation of daily returns between selected cryptocurrencies.
    """
    st.header(t("Correlation Analysis"))
    st.markdown(t("Analyze how the daily returns of selected cryptocurrencies move together."))
    
    coin_names = list(coin_list_dict.keys())
    # Initialize default selections for coins and days in session state if not present.
    if 'corr_selected_coins' not in st.session_state:
        st.session_state['corr_selected_coins'] = coin_names[:3] # Default to first 3 coins
    if 'corr_days' not in st.session_state:
        st.session_state['corr_days'] = 90 # Default to 90 days

    # User inputs for selecting coins and the number of days for analysis.
    selected_coins = st.multiselect(
        t("Select cryptocurrencies"),
        coin_names,
        default=st.session_state['corr_selected_coins'],
        key="corr_coins_multiselect"
    )
    days = st.slider(
        t("Number of days"), 30, 365, st.session_state['corr_days'], key="corr_days_slider"
    )
    # Update session state with current selections.
    st.session_state['corr_selected_coins'] = selected_coins
    st.session_state['corr_days'] = days

    cooldown_duration = 120  # 2 minutes for API cooldown
    overall_cooldown_active = False
    cooldown_messages = []

    # Check if any selected coin is currently in a cooldown period.
    for coin_name_iter in selected_coins:
        coin_id_iter = coin_list_dict.get(coin_name_iter)
        if coin_id_iter:
            cooldown_key_corr = f"corr_cooldown_{coin_id_iter}_{vs_currency}_{days}"
            now = time.time()
            cooldown_until = st.session_state.get(cooldown_key_corr, 0)
            if now < cooldown_until:
                overall_cooldown_active = True
                remaining_time = int(cooldown_until - now)
                cooldown_messages.append(f"{coin_name_iter}: {t('API rate limit previously reached. Please wait')} {remaining_time} {t('seconds.')}")

    if overall_cooldown_active and cooldown_messages:
        for msg_cd in cooldown_messages:
            st.info(msg_cd)

    refresh_button = st.button(t("Load/Refresh Data"), key="corr_refresh", disabled=overall_cooldown_active)
    # Use sorted coin names for a consistent cache key, regardless of selection order.
    cache_key = f"correlation_data_{'_'.join(sorted(selected_coins))}_{days}_{vs_currency}"

    price_data = {}
    error_msgs = []

    # Fetch data if refresh button is clicked and no cooldown is active.
    if refresh_button and not overall_cooldown_active:
        any_429_occurred = False
        for coin_name_fetch in selected_coins:
            coin_id_fetch = coin_list_dict.get(coin_name_fetch)
            if not coin_id_fetch:
                error_msgs.append(f"{coin_name_fetch}: {t('Could not find ID.')}")
                continue
            
            # Double-check individual cooldown before fetching.
            cooldown_key_individual = f"corr_cooldown_{coin_id_fetch}_{vs_currency}_{days}"
            if time.time() < st.session_state.get(cooldown_key_individual, 0):
                error_msgs.append(f"{coin_name_fetch}: {t('Still in cooldown, skipping.')}")
                continue

            with st.spinner(f"{t('Fetching data for')} {coin_name_fetch}..."):
                df_new, msg_new = get_coingecko_chart(coin_id_fetch, vs_currency=vs_currency, days=days)
            
            if not df_new.empty:
                price_data[coin_name_fetch] = df_new['price']
            elif msg_new and ("429" in msg_new or "Too Many Requests" in msg_new or "rate limit" in msg_new.lower()):
                error_msgs.append(f"{t('API rate limit reached for')} {coin_name_fetch}. {t('Please wait and try again.')}")
                st.session_state[cooldown_key_individual] = time.time() + cooldown_duration
                any_429_occurred = True
            elif msg_new and msg_new != "Success":
                error_msgs.append(f"{coin_name_fetch}: {msg_new}")
            else: 
                error_msgs.append(f"{coin_name_fetch}: {t('No data available for the selected period.')}")

        st.session_state[cache_key] = (price_data, error_msgs)
        st.session_state['corr_last_cache_key'] = cache_key
        if any_429_occurred:
            st.error(t("One or more API calls were rate-limited. Cooldowns have been applied. Please try again later."))
            st.rerun() # Rerun to update UI based on new cooldown state.

    # Load data from cache if available and no refresh was triggered.
    elif 'corr_last_cache_key' in st.session_state and st.session_state['corr_last_cache_key'] == cache_key and cache_key in st.session_state:
        price_data, error_msgs = st.session_state[cache_key]
        if error_msgs: # If cached data had errors, inform the user.
            st.info(t("Displaying cached data. Previous attempt had the following issues:"))
    else: # No refresh, no cache hit.
        if not overall_cooldown_active:
            st.info(t("Click 'Load/Refresh Data' to fetch or update the chart."))
        return # Exit if no data to process.

    # Display any accumulated error messages.
    if error_msgs:
        for m in error_msgs:
            st.warning(m)
        if any("API rate limit" in m for m in error_msgs):
             st.info(t("Some or all data could not be loaded due to API limits or errors."))
        elif not price_data: # No data at all due to errors.
            st.info(t("No data could be loaded for correlation analysis due to errors."))
            return

    # Check if enough data is available for correlation.
    if len(price_data) < 2:
        if not error_msgs: # Show only if no other specific errors were displayed.
            st.warning(t("Not enough data for correlation analysis. Please select at least two cryptocurrencies with available data."))
        return
        
    df_prices = pd.DataFrame(price_data)
    returns = df_prices.pct_change().dropna() # Calculate percentage change and drop NaNs.
    if returns.empty:
        st.warning(t("No return data available for the selected coins and period."))
        return
        
    corr = returns.corr() # Calculate correlation matrix.

    # Plotting settings based on theme.
    active_theme_mode = st.session_state.get('theme_mode', 'Dark')
    current_plotly_template = 'plotly_white' if active_theme_mode == 'Light' else 'plotly_dark'
    text_color_for_plotly = '#212529' if active_theme_mode == 'Light' else '#FAFAFA'

    # Create and display the correlation heatmap.
    fig = px.imshow(
        corr,
        text_auto=True, # Display correlation values on the heatmap.
        color_continuous_scale="RdBu", # Red-Blue color scale for correlation.
        zmin=-1, zmax=1, # Set min/max for color scale.
        title=t("Correlation Matrix of Daily Returns"),
        template=current_plotly_template
    )
    fig.update_layout(
        title_font_color=text_color_for_plotly,
        xaxis_title_font_color=text_color_for_plotly,
        yaxis_title_font_color=text_color_for_plotly,
        legend_font_color=text_color_for_plotly,
        font_color=text_color_for_plotly,
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
    fig.update_yaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
    st.plotly_chart(fig, use_container_width=True)


# --- Trend Analysis ---
def show_trend_analysis(coin_list_dict, vs_currency):
    """
    Renders the 'Trend Analysis' page.
    Displays price trend, moving averages (SMA, EMA), trendline, and volatility for a selected coin.
    """
    st.header(t("Trend Analysis"))
    coin_names = list(coin_list_dict.keys())

    # Default selections for coin and days.
    if 'trend_selected_coin' not in st.session_state:
        st.session_state['trend_selected_coin'] = coin_names[0] if coin_names else None
    if 'trend_days' not in st.session_state:
        st.session_state['trend_days'] = 90

    selected_coin = st.selectbox(
        t("Select cryptocurrency"),
        coin_names,
        index=coin_names.index(st.session_state['trend_selected_coin']) if st.session_state.get('trend_selected_coin') in coin_names else 0,
        key="trend_coin_selectbox"
    )
    days = st.slider(
        t("Number of days"), 30, 365, st.session_state['trend_days'], key="trend_days_slider"
    )
    # Update session state.
    st.session_state['trend_selected_coin'] = selected_coin
    st.session_state['trend_days'] = days

    coin_id = coin_list_dict.get(selected_coin)
    if not coin_id:
        st.error(f"Could not find ID for {selected_coin}")
        return

    # Cooldown logic for API rate limiting.
    cooldown_duration = 120  # 2 minutes
    cooldown_key_trend = f"trend_cooldown_{coin_id}_{vs_currency}_{days}"
    now = time.time()
    cooldown_until = st.session_state.get(cooldown_key_trend, 0)
    is_cooldown_active = now < cooldown_until

    if is_cooldown_active:
        remaining_time = int(cooldown_until - now)
        st.info(f"{t('API rate limit previously reached for this selection. Please wait')} {remaining_time} {t('seconds before trying again.')}")

    refresh_button = st.button(t("Load/Refresh Data"), key="trend_refresh", disabled=is_cooldown_active)
    cache_key = f"trend_data_{selected_coin}_{days}_{vs_currency}"

    df = pd.DataFrame()
    msg = ""

    # Fetch or load cached data.
    if refresh_button and not is_cooldown_active:
        with st.spinner(f"{t('Fetching data for')} {selected_coin}..."):
            df_new, msg_new = get_coingecko_chart(coin_id, vs_currency=vs_currency, days=days)
            st.session_state[cache_key] = (df_new, msg_new) # Cache new data/message.
            st.session_state['trend_last_cache_key'] = cache_key
            df, msg = df_new, msg_new
            
            if msg_new and ("429" in msg_new or "Too Many Requests" in msg_new or "rate limit" in msg_new.lower()):
                st.session_state[cooldown_key_trend] = time.time() + cooldown_duration
                st.error(f"{t('API rate limit reached (CoinGecko 429 error)')}: {msg_new}. {t('Please wait')} {cooldown_duration // 60} {t('minutes.')}")
                st.rerun() # Rerun to reflect cooldown state.
            elif msg_new and msg_new != "Success":
                st.warning(f"{t('API returned:')} {msg_new}")
            elif df_new.empty or 'price' not in df_new.columns:
                 st.warning(t("No data available for this coin for the selected period."))
                 
    elif 'trend_last_cache_key' in st.session_state and st.session_state['trend_last_cache_key'] == cache_key and cache_key in st.session_state:
        df, msg = st.session_state[cache_key]
        if msg and ("429" in msg or "Too Many Requests" in msg or "rate limit" in msg.lower()):
             st.info(t("Displaying cached data. Previous attempt resulted in a rate limit error."))
        elif msg and msg != "Success" and not any(no_data_msg in msg for no_data_msg in ["No data available", "No price data received", "No price data points"]):
            st.warning(f"{t('Displaying cached data. API previously returned:')} {msg}")
        elif df.empty or 'price' not in df.columns: # Cached data might be empty.
            st.warning(t("No data available for this coin for the selected period (cached)."))
    else: 
        if not is_cooldown_active:
            st.info(t("Click 'Load/Refresh Data' to fetch or update the chart."))
        return

    if df.empty or 'price' not in df.columns:
        return # Don't plot if no valid data.

    # Calculate technical indicators.
    df['SMA20'] = df['price'].rolling(window=20).mean()
    df['EMA20'] = df['price'].ewm(span=20, adjust=False).mean()
    
    # Linear trendline calculation.
    x_trend = np.arange(len(df)).reshape(-1, 1)
    y_trend = df['price'].values
    try:
        model = LinearRegression().fit(x_trend, y_trend)
        df['Trend'] = model.predict(x_trend)
    except Exception as e: # Handle potential errors in fitting model.
        df['Trend'] = np.nan
        st.info(f"{t('Trendline could not be calculated:')} {e}")
        
    df['Volatility'] = df['price'].pct_change().rolling(window=14).std() * 100 # 14-day rolling volatility.

    # Plotly theme settings.
    active_theme_mode = st.session_state.get('theme_mode', 'Dark')
    current_plotly_template = 'plotly_white' if active_theme_mode == 'Light' else 'plotly_dark'
    text_color_for_plotly = '#212529' if active_theme_mode == 'Light' else '#FAFAFA'

    # Main price trend chart.
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df.index, y=df['price'], name=t("Price")))
    if not df['SMA20'].isnull().all():
        fig_trend.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name=t("SMA20")))
    if not df['EMA20'].isnull().all():
        fig_trend.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name=t("EMA20")))
    if not df['Trend'].isnull().all():
        fig_trend.add_trace(go.Scatter(x=df.index, y=df['Trend'], name=t("Trendline"), line=dict(dash='dash')))
    
    fig_trend.update_layout(
        title_text=f"{selected_coin} {t('Price Trend')}",
        xaxis_title_text=t("Date"), yaxis_title_text=f"{t('Price')} ({vs_currency.upper()})",
        template=current_plotly_template, font_color=text_color_for_plotly,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    fig_trend.update_xaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
    fig_trend.update_yaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
    st.plotly_chart(fig_trend, use_container_width=True)

    # Volatility chart.
    if not df['Volatility'].isnull().all():
        fig_volatility = go.Figure()
        fig_volatility.add_trace(go.Scatter(x=df.index, y=df['Volatility'], name=t("14-day Volatility (%)")))
        fig_volatility.update_layout(
            title_text=f"{selected_coin} {t('Volatility')}",
            xaxis_title_text=t("Date"), yaxis_title_text=t("Volatility (%)"),
            template=current_plotly_template, font_color=text_color_for_plotly,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_volatility.update_xaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
        fig_volatility.update_yaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
        st.plotly_chart(fig_volatility, use_container_width=True)
    else:
        st.info(t("Not enough data to calculate volatility."))


# --- API Helper Functions ---
# Grouping all external API interaction logic here.

def _fetch_data(url, params=None, headers=None, timeout=10):
    """
    Centralized data fetching with common error handling for API requests.
    
    Args:
        url (str): API endpoint.
        params (dict, optional): Request parameters.
        headers (dict, optional): Request headers.
        timeout (int): Request timeout in seconds.
    Returns:
        tuple: (JSON response data or None, status message string)
    """
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        return response.json(), "Success"
    except requests.exceptions.ConnectionError as conn_err:
        return None, f"Network Error: Could not connect. Details: {conn_err}"
    except requests.exceptions.Timeout as timeout_err:
        return None, f"Request timed out. Details: {timeout_err}"
    except requests.exceptions.HTTPError as http_err:
        # Specific handling for HTTP errors, including status code
        status_code = http_err.response.status_code if http_err.response is not None else "N/A"
        return None, f"HTTP Error: {str(http_err)} (Status: {status_code})"
    except json.JSONDecodeError:
        # Handles cases where response isn't valid JSON
        return None, "JSON Error: Failed to decode API response."
    except Exception as e: # Catch-all for any other unexpected errors
        return None, f"An unexpected error occurred during fetch: {str(e)}"


@st.cache_data(ttl=300) # Cache market data for 5 minutes
def get_coingecko_markets(vs_currency='usd', limit=10):
    """
    Fetches top cryptocurrencies market data from CoinGecko.
    Uses the /coins/markets endpoint to get current price, market cap,
    volume, and percentage change for a specified number of coins
    against a given fiat/crypto currency.

    Args:
        vs_currency (str): Target currency for market data (e.g., 'usd', 'eur').
        limit (int): Number of top coins to retrieve.
    Returns:
        tuple: (pd.DataFrame with market data, status message string)
               Returns empty DataFrame on error.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = { # Parameters for CoinGecko /coins/markets endpoint
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": False
    }
    data, msg = _fetch_data(url, params=params, timeout=10)
    if msg != "Success":
        # Return the error message instead of calling st.error directly
        return pd.DataFrame(), msg
    if not isinstance(data, list):
        # Return a format error message
        return pd.DataFrame(), "API Error: Unexpected data format received for market data."
    df = pd.DataFrame(data)
    columns = ['id', 'symbol', 'name', 'current_price', 'market_cap',
              'market_cap_rank', 'total_volume', 'price_change_percentage_24h',
              'atl', 'ath']
    df = df.reindex(columns=columns) # Ensure consistent column order
    return df, "Success"


@st.cache_data(ttl=60) # Cache chart data for 1 minute; more volatile
def get_coingecko_chart(coin_id='bitcoin', vs_currency='usd', days=30):
    """
    Get historical price, volume, and market cap data from CoinGecko public API.
    Endpoint: /coins/{id}/market_chart
    Args:
        coin_id (str): CoinGecko ID (e.g., 'bitcoin').
        vs_currency (str): Target currency (e.g., 'usd').
        days (int): Number of days for historical data.
    Returns:
        tuple: (pd.DataFrame with price, volume, market_cap, status message string)
               Returns empty DataFrame on error or if no price data.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = { # Parameters for CoinGecko /market_chart endpoint
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily" if days > 2 else "hourly" # Daily for >2 days, else hourly
    }
    data, msg = _fetch_data(url, params=params, timeout=15)

    if msg != "Success":
        return pd.DataFrame(), msg # Pass error up

    if not data or 'prices' not in data or not data['prices']:
        return pd.DataFrame(), f"No price data points received for {coin_id}."

    # Process and merge price, volume, and market cap data
    prices_df = pd.DataFrame(data.get('prices', []), columns=['timestamp', 'price'])
    volumes_df = pd.DataFrame(data.get('total_volumes', []), columns=['timestamp', 'volume'])
    market_caps_df = pd.DataFrame(data.get('market_caps', []), columns=['timestamp', 'market_cap'])

    # Convert timestamps to datetime objects
    for df_part in [prices_df, volumes_df, market_caps_df]:
        if not df_part.empty:
            df_part['timestamp'] = pd.to_datetime(df_part['timestamp'], unit='ms')

    # Merge dataframes; start with prices as it's essential
    df = prices_df
    if not volumes_df.empty:
        df = pd.merge(df, volumes_df, on='timestamp', how='left')
    if not market_caps_df.empty:
        df = pd.merge(df, market_caps_df, on='timestamp', how='left')
    
    df.set_index('timestamp', inplace=True)
    return df, "Success"


@st.cache_data # Static data, cache indefinitely within session
def get_supported_vs_currencies():
    """
    Get the list of supported vs currencies from CoinGecko.
    Endpoint: /simple/supported_vs_currencies.
    
    Returns:
        tuple: (List of supported currency strings, status message string)
    """
    url = "https://api.coingecko.com/api/v3/simple/supported_vs_currencies"
    data, msg = _fetch_data(url, timeout=10)

    if msg != "Success":
        return [], msg # Pass error up
    
    return data if isinstance(data, list) else [], "Success" # Expecting a list


@st.cache_data # Static data, cache indefinitely
def get_coin_list():
    """
    Get the list of supported coins (id, symbol, name) from CoinGecko.
    Endpoint: /coins/list.
    
    Returns:
        tuple: (Dictionary mapping coin name to CoinGecko ID, status message string)
    """
    url = "https://api.coingecko.com/api/v3/coins/list"
    data, msg = _fetch_data(url, timeout=15)

    if msg != "Success":
        return {}, msg # Pass error up

    if isinstance(data, list):
        # Create a dictionary of name:id for easy lookup
        return {coin['name']: coin['id'] for coin in data if 'name' in coin and 'id' in coin}, "Success"
    else:
        return {}, "API Error: Unexpected data format for coin list."


@st.cache_data(ttl=300) # Cache global data for 5 minutes
def get_global_data():
    """
    Get cryptocurrency global data from CoinGecko.
    Endpoint: /global.
    
    Returns:
        tuple: (Dictionary with global crypto data, status message string)
    """
    url = "https://api.coingecko.com/api/v3/global"
    data, msg = _fetch_data(url, timeout=10)

    if msg != "Success":
        return {}, msg # Pass error up
    
    # The actual data is nested under a 'data' key in the response
    if data and 'data' in data and isinstance(data['data'], dict):
        return data['data'], "Success"
    else:
        return {}, "API Error: Global data not found or in unexpected format."


@st.cache_data(ttl=300) # Cache coin details for 5 minutes
def get_coin_details(coin_id):
    """
    Get detailed information for a specific coin from CoinGecko.
    Endpoint: /coins/{id}
    Args:
        coin_id (str): CoinGecko ID of the coin.
    Returns:
        tuple: (Dictionary with detailed coin information, status message string)
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = { # Parameters to minimize response size, fetching only essential details
        "localization": "false",
        "tickers": "false",
        "market_data": "false",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false"
    }
    data, msg = _fetch_data(url, params=params, timeout=15)

    if msg != "Success":
        return {}, msg # Pass error up
    
    return data if isinstance(data, dict) else {}, "Success"


@st.cache_data(ttl=600) # Cache exchanges list for 10 minutes
def get_exchanges_list():
    """
    Get cryptocurrency exchanges list from CoinGecko.
    Endpoint: /exchanges.
    
    Returns:
        tuple: (pd.DataFrame with exchange data, status message string)
    """
    url = "https://api.coingecko.com/api/v3/exchanges"
    params = {"per_page": 100} # Fetch top 100 exchanges
    data, msg = _fetch_data(url, params=params, timeout=15)

    if msg != "Success":
        return pd.DataFrame(), msg # Pass error up

    if isinstance(data, list):
        df = pd.DataFrame(data)
        # Select and reorder relevant columns
        columns = ['id', 'name', 'country', 'trust_score', 'trade_volume_24h_btc', 'url']
        df = df.reindex(columns=columns, fill_value=None) # Use fill_value for missing cols
        return df, "Success"
    else:
        return pd.DataFrame(), "API Error: Unexpected data format for exchanges list."


@st.cache_data(ttl=600) # Cache derivative exchanges for 10 minutes
def get_derivative_exchanges():
    """
    Get derivative exchanges list from CoinGecko.
    Endpoint: /derivatives/exchanges.
    
    Returns:
        tuple: (pd.DataFrame with derivative exchange data, status message string)
    """
    url = "https://api.coingecko.com/api/v3/derivatives/exchanges"
    params = {"per_page": 100} # Fetch top 100
    data, msg = _fetch_data(url, params=params, timeout=15)

    if msg != "Success":
        return pd.DataFrame(), msg # Pass error up

    if isinstance(data, list):
        df = pd.DataFrame(data)
        # Select and reorder relevant columns
        columns = ['id', 'name', 'open_interest_btc', 'number_of_perpetual_pairs', 'number_of_futures_pairs']
        df = df.reindex(columns=columns, fill_value=None)
        return df, "Success"
    else:
        return pd.DataFrame(), "API Error: Unexpected data format for derivative exchanges."


@st.cache_data(ttl=300) # Cache news for 5 minutes
def get_news_with_sentiment(coin_symbol=None):
    """
    Get and analyze crypto news sentiment from CryptoCompare.
    Optionally filters by a specific coin symbol if provided.
    Endpoint: /data/v2/news/
    Args:
        coin_symbol (str, optional): Coin symbol (e.g., BTC) for specific news.
                                     None for general crypto news.
    Returns:
        tuple: (pd.DataFrame with news and sentiment, status message string)
    """
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    params = {"limit": 50} 
    
    if coin_symbol and coin_symbol.upper() != "GENERAL":
        params['categories'] = coin_symbol.upper() # Filter by coin category if specified

    headers = None
    api_key_warning_key = "cryptocompare_api_key_warning_shown" # For one-time warning

    if "CRYPTOCOMPARE_API_KEY" in st.secrets:
        headers = {"authorization": f"Apikey {st.secrets['CRYPTOCOMPARE_API_KEY']}"}
    elif not st.session_state.get(api_key_warning_key, False):
        # Warn once if API key is missing, as it's crucial for CryptoCompare.
        st.warning("CRYPTOCOMPARE_API_KEY missing. News fetching might be limited or fail, especially for specific coins.")
        st.session_state[api_key_warning_key] = True

    data, msg = _fetch_data(url, params=params, headers=headers, timeout=15)

    if msg != "Success":
        return pd.DataFrame(), f"CryptoCompare news fetch error: {msg}"
        
    if not data or 'Data' not in data or not isinstance(data['Data'], list):
        return pd.DataFrame(), f"No news data or unexpected format from CryptoCompare for '{coin_symbol or 'General'}'. Status: {msg}"
    
    news_items = []
    for item in data.get('Data', []):
        # Basic validation for essential fields
        if all(k in item for k in ['title', 'body', 'url', 'published_on', 'source']):
            text_content = f"{item.get('title', '')} {item.get('body', '')}"
            try:
                sentiment_score = TextBlob(text_content).sentiment.polarity
            except Exception: # Broad exception for TextBlob issues
                sentiment_score = 0.0 # Neutral sentiment on analysis error
            
            news_items.append({
                'title': item['title'],
                'source': item.get('source', 'N/A'),
                'published_at': pd.to_datetime(item['published_on'], unit='s', errors='coerce'),
                'url': item['url'],
                'sentiment': sentiment_score
            })

    df_news = pd.DataFrame(news_items)
    df_news.dropna(subset=['published_at'], inplace=True) # Ensure valid dates
    if not df_news.empty:
        df_news = df_news.sort_values(by='published_at', ascending=False)
        return df_news, "Success"
    else:
        return pd.DataFrame(), "No news data processed"


@st.cache_data(ttl=60) # Cache current price for 1 minute
def get_current_coin_price(coin_ids, vs_currency):
    """
    Get the current price for one or more coins using CoinGecko's /simple/price endpoint.
    Endpoint: /simple/price.
    
    Args:
        coin_ids (list or str): List of CoinGecko IDs, or a single ID string.
        vs_currency (str): Target currency for prices.
    Returns:
        tuple: (Dictionary {coin_id: price}, status message string)
               Price is None if not found for a specific coin_id.
    """
    if isinstance(coin_ids, str): # Ensure coin_ids is a list for join
        coin_ids = [coin_ids]
        
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ",".join(coin_ids), # Comma-separated IDs
        "vs_currencies": vs_currency
    }
    data, msg = _fetch_data(url, params=params, timeout=10)

    if msg != "Success" or not isinstance(data, dict):
        return {cid: None for cid in coin_ids}, msg if msg != "Success" else "API Error: Price data in unexpected format."

    prices = {}
    all_prices_found = True
    for cid in coin_ids:
        if cid in data and vs_currency in data[cid]:
            prices[cid] = data[cid][vs_currency]
        else:
            prices[cid] = None # Price not found for this coin
            all_prices_found = False
            
    status_msg = "Success"
    if not all_prices_found:
        status_msg = f"Partial Success: Price data missing for some coins in {vs_currency}."
        
    return prices, status_msg


# --- Data Analysis Functions ---

def calculate_technical_indicators(df):
    """
    Calculates various technical indicators (SMA, Bollinger Bands, RSI, MACD)
    for a given price DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'price' column.
    Returns:
        pd.DataFrame: Original DataFrame augmented with indicator columns.
                      Returns original df if price data is insufficient or missing.
    """
    if df.empty or 'price' not in df.columns:
        # st.warning is handled by the caller if needed, this function focuses on calculation
        return df # Return original df if no price data
    
    try:
        # SMA and Bollinger Bands (require at least 20 data points)
        if len(df) >= 20:
            df['SMA20'] = df['price'].rolling(window=20).mean()
            df['StdDev20'] = df['price'].rolling(window=20).std() # Standard deviation for Bollinger Bands
            df['BollingerUpper'] = df['SMA20'] + (df['StdDev20'] * 2)
            df['BollingerLower'] = df['SMA20'] - (df['StdDev20'] * 2)
        else:
            # Not enough data, fill with NaN
            df['SMA20'] = np.nan
            df['BollingerUpper'] = np.nan
            df['BollingerLower'] = np.nan

        # RSI (Relative Strength Index - requires at least 14 data points)
        if len(df) >= 14:
            delta = df['price'].diff()
            gain = delta.where(delta > 0, 0.0).fillna(0.0) # Ensure 0.0 for fillna
            loss = (-delta.where(delta < 0, 0.0)).fillna(0.0) # Ensure 0.0 for fillna
            
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            
            rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss) # Avoid division by zero
            df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
            df['RSI'] = df['RSI'].replace([np.inf, -np.inf], 100.0).fillna(50.0) # Handle inf, fill NaN with 50 (neutral)
        else:
            df['RSI'] = np.nan

        # MACD (Moving Average Convergence Divergence)
        # Check for sufficient data for EMA calculations if necessary, though ewm handles shorter series.
        df['EMA12'] = df['price'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['SignalLine'] = df['MACD'].ewm(span=9, adjust=False).mean() # Signal line for MACD

        return df
    except Exception as e:
        # Log or handle error appropriately, return original df to prevent app crash
        # st.error is UI-specific, better to let caller handle UI messages.
        print(f"Error in calculate_technical_indicators: {e}") # Or use proper logging
        return df


def check_price_alerts(current_price, thresholds):
    """
    Check if the current price triggers high or low alert thresholds.

    Args:
        current_price (float | None): Current coin price. None if unavailable.
        thresholds (dict): {'high': float, 'low': float} price thresholds.

    Returns:
        list: List of triggered alert dicts (e.g., {'type': 'high', ...}).
              Empty if no alerts or invalid input.
    """
    alerts = []
    if not isinstance(current_price, (int, float)):
        return alerts # No valid price to check

    high_threshold = thresholds.get('high')
    low_threshold = thresholds.get('low')

    # Check high threshold
    if isinstance(high_threshold, (int, float)) and current_price > high_threshold:
        alerts.append({'type': 'high', 'price': current_price, 'threshold': high_threshold})
    
    # Check low threshold (elif ensures only one alert type if price crosses both, though unlikely with typical thresholds)
    elif isinstance(low_threshold, (int, float)) and current_price < low_threshold:
        alerts.append({'type': 'low', 'price': current_price, 'threshold': low_threshold})

    return alerts


# --- Notification Functions ---
# Handles sending alerts via email and SMS.

def send_alert_notification(email, alert_message):
    """
    Sends an email notification for a price alert using SMTP.
    Requires EMAIL_ADDRESS, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT in Streamlit secrets.
    """
    if not email: # Basic validation
        st.warning("Email for notification is missing.")
        return

    # Check for necessary SMTP secrets
    required_smtp_secrets = ["EMAIL_ADDRESS", "EMAIL_PASSWORD", "SMTP_SERVER", "SMTP_PORT"]
    if not all(secret in st.secrets for secret in required_smtp_secrets):
        missing_secrets = [s for s in required_smtp_secrets if s not in st.secrets]
        st.error(f"SMTP config incomplete. Missing secrets: {', '.join(missing_secrets)}. Check .streamlit/secrets.toml.")
        return

    try:
        # Retrieve secrets
        sender_email = st.secrets["EMAIL_ADDRESS"]
        sender_password = st.secrets["EMAIL_PASSWORD"]
        smtp_server = st.secrets["SMTP_SERVER"]
        smtp_port = int(st.secrets["SMTP_PORT"]) # Port must be an int

        # Construct email
        email_msg = MIMEText(alert_message)
        email_msg['Subject'] = "Crypto Price Alert"
        email_msg['From'] = sender_email
        email_msg['To'] = email

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls() # Upgrade to secure connection
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, email_msg.as_string())
        st.success(f"Email alert sent to {email}.")

    except smtplib.SMTPAuthenticationError:
        st.error("SMTP login failed. Check email credentials in secrets.")
    except (smtplib.SMTPConnectError, ConnectionRefusedError, smtplib.SMTPServerDisconnected) as conn_err:
        st.error(f"SMTP connection failed: {conn_err}. Check server/port and network.")
    except smtplib.SMTPException as smtp_err: # Other SMTP related errors
        st.error(f"SMTP error: {smtp_err}")
    except KeyError as key_err: # Should be caught by initial check, but as a safeguard
        st.error(f"A required SMTP secret ({key_err}) is missing.")
    except Exception as e: # Catch any other unexpected errors
        st.error(f"Failed to send email notification: {e}")


def send_sms_notification(phone_number, alert_message):
    """
    Sends an SMS notification for a price alert using Twilio.
    Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER in Streamlit secrets.
    """
    if not phone_number: # Basic validation
        st.warning("Phone number for SMS notification is missing.")
        return

    # Check for necessary Twilio secrets
    required_twilio_secrets = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"]
    if not all(secret in st.secrets for secret in required_twilio_secrets):
        missing_secrets = [s for s in required_twilio_secrets if s not in st.secrets]
        st.error(f"Twilio config incomplete. Missing secrets: {', '.join(missing_secrets)}. Check .streamlit/secrets.toml.")
        return

    try:
        from twilio.rest import Client # Import here to avoid error if Twilio not installed/used

        # Retrieve secrets
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        twilio_phone_number = st.secrets["TWILIO_PHONE_NUMBER"]

        # Initialize Twilio client and send message
        client = Client(account_sid, auth_token)
        client.messages.create(
            body=alert_message,
            from_=twilio_phone_number,
            to=phone_number
        )
        st.success(f"SMS alert sent to {phone_number}.")
    
    except ImportError:
        st.error("Twilio library not installed. SMS notifications require 'twilio'.")
    except Exception as e: # Catch other errors (e.g., Twilio API errors, invalid phone number)
        st.error(f"Failed to send SMS notification: {e}")


# --- Page Content Functions ---
# UI rendering for different app sections.
# Caching is handled by data fetching functions (get_*) to avoid issues with
# session_state dependencies in these UI-focused functions.

def human_format(num):
    """Converts a large number to a human-readable format (K, M, B, T)."""
    num = float(num) # Ensure float for calculations
    magnitude = 0
    while abs(num) >= 1000 and magnitude < 4:
        magnitude += 1
        num /= 1000.0
    return f'{num:.2f}{["", "K", "M", "B", "T"][magnitude]}'

def show_market_overview(vs_currency):
    st.markdown(
        """
        <style>
        .metric-card {
            background: var(--card-background);
            border-radius: 18px;
            box-shadow: 0 2px 16px rgba(0,0,0,0.10);
            padding: 1.7rem 1.1rem 1.3rem 1.1rem;
            margin-bottom: 2rem;
            text-align: center;
            color: var(--text-color);
            transition: box-shadow 0.2s, transform 0.2s;
            position: relative;
        }
        .metric-card:hover {
            box-shadow: 0 6px 24px rgba(0,0,0,0.18);
            transform: scale(1.03);
        }
        .metric-desc {
            font-size: 0.95rem;
            color: var(--card-label-color);
            margin-top: 0.3rem;
        }
        .section-divider {
            border-top: 2px solid var(--secondary-background-color);
            opacity: 0.5;
            margin: 2.5rem 0 2rem 0;
        }
        .main-bg {
            background: var(--main-bg-gradient);
            border-radius: 24px;
            padding: 2.5rem 2.5rem 2rem 2.5rem;
            margin-bottom: 2.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.header(t("Market Overview") + " 📊")

    # Theme-dependent settings for Plotly charts
    active_theme_mode = st.session_state.get('theme_mode', 'Dark')
    current_plotly_template = 'plotly_white' if active_theme_mode == 'Light' else 'plotly_dark'
    text_color_for_plotly = '#212529' if active_theme_mode == 'Light' else '#FAFAFA'

    # Introductory text for the market overview section.
    st.markdown(f"""
    **{t("Market Cap Explanation")}**

    **{t("How To Use Chart Header")}**
    {t("How To Use Chart Point 1")}
    {t("How To Use Chart Point 2")}
    {t("How To Use Chart Point 3")}
    """)
    with st.container():
        # Display global market metrics
        with st.spinner(t("Loading Global Data")):
            global_data, msg_global = get_global_data()
        
        if global_data: # Check if global_data is not empty and successfully fetched
            total_market_cap = global_data.get('total_market_cap', {}).get(vs_currency, 0)
            total_volume_24h = global_data.get('total_volume', {}).get(vs_currency, 0)
            market_cap_change_percentage_24h = global_data.get('market_cap_change_percentage_24h_usd', 0) # Assuming USD base for this %
            btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
            # These might not always be present, so default to 0 or handle appropriately
            total_volume_24h_change = global_data.get('total_volume_change_24h', {}).get(vs_currency, 0) or 0 
            btc_dominance_change = global_data.get('market_cap_percentage_24h_change', {}).get('btc', 0) or 0

            cols = st.columns(3)
            with cols[0]:
                custom_metric(t("Total Market Cap"), f"{human_format(total_market_cap)} {vs_currency.upper()}", f"{market_cap_change_percentage_24h:+.2f}%")
            with cols[1]:
                custom_metric(t("24h Volume"), f"{human_format(total_volume_24h)} {vs_currency.upper()}", f"{total_volume_24h_change:+.2f}%" if total_volume_24h_change else "")
            with cols[2]:
                custom_metric(t("BTC Dominance"), f"{btc_dominance:.2f}%", f"{btc_dominance_change:+.2f}%" if btc_dominance_change else "")
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        else:
            st.warning(f"{t('Error Fetching Global Data')} {msg_global}")

        # Top Cryptocurrencies by Market Cap section
        with st.container():
            st.subheader(f"🏆 {t('Top Cryptocurrencies by Market Cap')} ({vs_currency.upper()})")
            
            num_coins_to_display = st.number_input(
                t("Num Coins To Display Label"), 
                min_value=5, max_value=100, value=st.session_state.get('market_data_limit', 20), step=1,
                help=t("Num Coins To Display Help"),
                key="num_coins_market_overview" # Unique key for this widget
            )

            # Refresh data if button is clicked or if the number of coins to display changes
            # or if market_data is not in session_state (first load)
            if st.button(t("Load/Refresh Data"), key="refresh_market_data_button") or \
               'market_data' not in st.session_state or \
               st.session_state.get('market_data_limit') != num_coins_to_display:
                with st.spinner(f"{t('Fetching Top')} {num_coins_to_display} {t('Coins')}..."):
                    df_markets, msg_markets = get_coingecko_markets(vs_currency=vs_currency, limit=int(num_coins_to_display))
                    st.session_state['market_data'] = df_markets
                    st.session_state['market_data_msg'] = msg_markets
                    st.session_state['market_data_limit'] = num_coins_to_display # Store the limit used

            # Retrieve data from session state for display
            df_markets_display = st.session_state.get('market_data', pd.DataFrame())
            msg_markets_display = st.session_state.get('market_data_msg', "")

            if msg_markets_display and msg_markets_display != "Success":
                st.warning(f"{t('Error Retrieving Market Data')} {msg_markets_display}")

            if df_markets_display.empty:
                if not (msg_markets_display and msg_markets_display != "Success"): # Avoid double message if error already shown
                    st.info(t("No Market Data Info"))
                return # Stop if no data to display

            # Dropdown for selecting a specific coin from the top list for detailed view
            top_coin_options = [f"{row['name']} ({row['symbol'].upper()})" for _, row in df_markets_display.iterrows()]
            
            if 'selected_market_coin' not in st.session_state or st.session_state.selected_market_coin not in top_coin_options:
                 st.session_state.selected_market_coin = top_coin_options[0] if top_coin_options else None

            selected_coin_display_name = st.selectbox(
                t("Choose Crypto To Display Label"),
                top_coin_options,
                index=top_coin_options.index(st.session_state.selected_market_coin) if st.session_state.selected_market_coin in top_coin_options else 0,
                help=t("Choose Crypto To Display Help"),
                key="select_market_coin_dropdown"
            )
            st.session_state.selected_market_coin = selected_coin_display_name


            # Filter for the selected coin's details (if a coin is selected)
            df_selected_coin_details = pd.DataFrame()
            if selected_coin_display_name:
                selected_symbol_for_df = selected_coin_display_name.split('(')[-1].replace(')','').strip().lower()
                df_selected_coin_details = df_markets_display[df_markets_display['symbol'].str.lower() == selected_symbol_for_df]

            # Visualization: Market Cap Bar Chart for all top coins
            bar_colors = [("#4CAF50" if val >= 0 else "#F44336") for val in df_markets_display['price_change_percentage_24h'].fillna(0)]
            
            fig_market_cap_all = go.Figure(data=[
                go.Bar(
                    x=df_markets_display['symbol'].str.upper(), 
                    y=df_markets_display['market_cap'] / (1e9 if vs_currency != 'btc' else 1), # Billions or raw for BTC
                    name=t("Market Cap"), 
                    marker_color=bar_colors,
                    customdata=np.stack((df_markets_display['market_cap_rank'], df_markets_display['current_price']), axis=-1),
                    hovertemplate=(
                        '<b>%{y}</b><br>' + 
                        f"{t('Symbol Label')} %{{x}}<br>" +
                        f"{t('Rank Label')} %{{customdata[0]}}<br>" +
                        f"{t('Price Label')} %{{customdata[1]:,.2f}} {vs_currency.upper()}<extra></extra>"
                    )
                )
            ])
            fig_market_cap_all.update_layout(
                title_text=f"{t('Top')} {int(num_coins_to_display)} {t('Cryptos By Market Cap')}",
                xaxis_title_text=t("Symbol Axis Label"),
                yaxis_title_text=f"{t('Market Cap Axis Label')} ({t('Billion Unit') if vs_currency != 'btc' else ''}{vs_currency.upper()})",
                template=current_plotly_template, font_color=text_color_for_plotly,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig_market_cap_all.update_xaxes(tickfont=dict(color=text_color_for_plotly), gridcolor='rgba(128,128,128,0.2)' if active_theme_mode == 'Dark' else 'rgba(200,200,200,0.5)')
            fig_market_cap_all.update_yaxes(tickfont=dict(color=text_color_for_plotly), gridcolor='rgba(128,128,128,0.2)' if active_theme_mode == 'Dark' else 'rgba(200,200,200,0.5)')
            st.plotly_chart(fig_market_cap_all, use_container_width=True)

            # Visualization: 24h Price Change % Bar Chart
            fig_price_change_all = go.Figure(data=[
                go.Bar(
                    x=df_markets_display['symbol'].str.upper(), 
                    y=df_markets_display['price_change_percentage_24h'],
                    marker_color=bar_colors,
                    name=t("24h Change Percent Label"),
                    hovertemplate='<b>%{y:.2f}%</b><br>' + f"{t('Symbol Label')} %{{x}}<extra></extra>"
                )
            ])
            fig_price_change_all.update_layout(
                title_text=f"{t('24h Change Chart Title Prefix')} {int(num_coins_to_display)} {t('Coins')}",
                xaxis_title_text=t("Symbol Axis Label"), 
                yaxis_title_text=t("24h Change Percent Axis Label"),
                template=current_plotly_template, font_color=text_color_for_plotly,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig_price_change_all.update_xaxes(tickfont=dict(color=text_color_for_plotly), gridcolor='rgba(128,128,128,0.2)' if active_theme_mode == 'Dark' else 'rgba(200,200,200,0.5)')
            fig_price_change_all.update_yaxes(tickfont=dict(color=text_color_for_plotly), gridcolor='rgba(128,128,128,0.2)' if active_theme_mode == 'Dark' else 'rgba(200,200,200,0.5)')
            st.plotly_chart(fig_price_change_all, use_container_width=True)

            # Visualization: Market Cap Donut Chart for the single selected coin from dropdown
            if not df_selected_coin_details.empty:
                selected_cap = float(df_selected_coin_details['market_cap'].iloc[0])
                total_top_caps = float(df_markets_display['market_cap'].sum())
                others_cap = total_top_caps - selected_cap
                
                donut_labels = [selected_coin_display_name, t('Other Label')]
                donut_values = [selected_cap, others_cap]

                fig_donut_selected = go.Figure(data=[go.Pie(
                    labels=donut_labels, values=donut_values, hole=0.5,
                    marker_colors=["#FFD700", "#888"], # Gold for selected, gray for others
                    hoverinfo="label+percent+value", textinfo="label+percent"
                )])
                fig_donut_selected.update_layout(
                    title_text=f"{selected_coin_display_name} {t('Market Cap Share Chart Title Prefix')} {int(num_coins_to_display)}",
                    template=current_plotly_template, font_color=text_color_for_plotly,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=True
                )
                st.plotly_chart(fig_donut_selected, use_container_width=True)
                
                # Display detailed table for the selected coin
                st.dataframe(df_selected_coin_details.style.format({
                    'current_price': f'{{:,.2f}} {vs_currency.upper()}',
                    'market_cap': f'{{:,.0f}} {vs_currency.upper()}',
                    'total_volume': f'{{:,.0f}} {vs_currency.upper()}',
                    'atl': f'{{:,.2f}} {vs_currency.upper()}',
                    'ath': f'{{:,.2f}} {vs_currency.upper()}',
                    'price_change_percentage_24h': '{:.2f}%'
                }), use_container_width=True)


def show_technical_analysis(coin_names, coin_list_dict, vs_currency):
    """
    Displays the UI for the Technical Analysis section.
    Users can select a coin and period to view its price chart 
    along with technical indicators like SMA, RSI, and Bollinger Bands.
    Handles API rate limiting with a cooldown.
    """
    # Using custom CSS for a consistent card effect, similar to Market Overview.
    st.markdown(
        """
        <style>
        .metric-card {
            background: var(--card-background);
            border-radius: 18px;
            box-shadow: 0 2px 16px rgba(0,0,0,0.10);
            padding: 1.7rem 1.1rem 1.3rem 1.1rem;
            margin-bottom: 2rem;
            text-align: center;
            color: var(--text-color);
            transition: box-shadow 0.2s, transform 0.2s;
            position: relative;
        }
        .metric-card:hover {box-shadow: 0 6px 24px rgba(0,0,0,0.18);transform: scale(1.03);}
        .section-divider {
            border-top: 2px solid var(--secondary-background-color);
            opacity: 0.5;
            margin: 2.5rem 0 2rem 0;
        }
        .main-bg {
            background: var(--main-bg-gradient);
            border-radius: 24px;
            padding: 2.5rem 2.5rem 2rem 2.5rem;
            margin-bottom: 2.5rem;
        }
        .logo {width: 54px; height: 54px; border-radius: 12px; margin-bottom: 0.5rem;}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.header(t("Technical Analysis") + " 📈")

    # Configure Plotly chart appearance based on current app theme
    active_theme_mode = st.session_state.get('theme_mode', 'Dark')
    current_plotly_template = 'plotly_white' if active_theme_mode == 'Light' else 'plotly_dark'
    text_color_for_plotly = '#212529' if active_theme_mode == 'Light' else '#FAFAFA'

    # Brief explanation of indicators and usage tips
    st.markdown(t("""
    **About Technical Indicators:**
    - **SMA (Simple Moving Average):** Average price over a period; helps spot trends.
    - **RSI (Relative Strength Index):** Measures momentum; >70 is overbought, <30 is oversold.
    - **Bollinger Bands:** Indicate price volatility; extremes can suggest overbought/oversold.
    - **MACD:** Shows trend direction & momentum; crossovers can be buy/sell signals.
    
    **Quick Tips:**
    - Look for crossovers, divergences, and extreme values.
    - Combine multiple indicators for stronger signals.
    """))
    with st.container():
        st.markdown('<div class="main-bg">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            selected_coin_name = st.selectbox(
                t("Select Cryptocurrency"),
                coin_names,
                index=coin_names.index("Bitcoin") if "Bitcoin" in coin_names else 0,
                help=t("Choose a coin to analyze.")
            )
        with col2:
            days = st.slider(t("Analysis Period (days)"), 1, 730, 90, help=t("Select the number of days for analysis."))
        coin_id = coin_list_dict.get(selected_coin_name, selected_coin_name.lower().replace(" ", "-"))
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        # --- 429 Rate Limit Handling ---
        cooldown_key = f"ta_cooldown_{coin_id}_{vs_currency}"
        import time
        now = time.time()
        cooldown_until = st.session_state.get(cooldown_key, 0)
        cooldown_active = now < cooldown_until
        if cooldown_active:
            st.info(f"{t('API rate limit reached. Please wait')} {int(cooldown_until - now)} {t('seconds before trying again.')}")
        button_disabled = cooldown_active
        if st.button(t("Graph Price and Indicators"), disabled=button_disabled):
            with st.spinner(f"{t('Fetching and analyzing data for')} {selected_coin_name}..."):
                df, msg = get_coingecko_chart(coin_id, vs_currency=vs_currency, days=days)
                if msg and ("429" in msg or "Too Many Requests" in msg or "rate limit" in msg.lower()):
                    st.error(t("API rate limit reached (CoinGecko 429 error). Please wait 2 minutes before trying again."))
                    st.session_state[cooldown_key] = time.time() + 120  # 2 minutes cooldown
                    return
                if not df.empty:
                    df = calculate_technical_indicators(df)
                    st.markdown('<div class="metric-card">📊<br><b>' + t('Technical Indicators') + '</b><br><span style="font-size:1.1rem;">SMA20, RSI, Bollinger Bands, MACD, EMA</span><div class="metric-desc">' + t('Displayed for selected period if enough data.') + '</div></div>', unsafe_allow_html=True)
                    # --- Price, SMA, and Bollinger Bands chart ---
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['price'], name=t("Price"), line=dict(color='blue')))
                    if 'SMA20' in df.columns and not df['SMA20'].isnull().all():
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name=t("SMA20"), line=dict(color='orange', dash='dash')))
                    if 'BollingerUpper' in df.columns and not df['BollingerUpper'].isnull().all():
                        fig.add_trace(go.Scatter(x=df.index, y=df['BollingerUpper'], name=t("Upper Band"), line=dict(color='gray', dash='dot')))
                    if 'BollingerLower' in df.columns and not df['BollingerLower'].isnull().all():
                        fig.add_trace(go.Scatter(x=df.index, y=df['BollingerLower'], name=t("Lower Band"), line=dict(color='gray', dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
                    fig.update_layout(
                        title_text=f"{selected_coin_name} {t('Price, SMA, and Bollinger Bands')} ({vs_currency.upper()})", # Use title_text
                        xaxis_title_text=t("Date"), # Use xaxis_title_text
                        yaxis_title_text=f"{t('Price')} ({vs_currency.upper()})",
                        template=current_plotly_template, # Use dynamic template
                        font_color=text_color_for_plotly,
                        title_font_color=text_color_for_plotly,
                        xaxis_title_font_color=text_color_for_plotly,
                        yaxis_title_font_color=text_color_for_plotly,
                        legend_font_color=text_color_for_plotly,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    fig.update_xaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
                    fig.update_yaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
                    st.plotly_chart(fig, use_container_width=True)
                    # --- RSI chart ---
                    if 'RSI' in df.columns and not df['RSI'].isnull().all():
                        fig_rsi = go.Figure(data=[go.Scatter(x=df.index, y=df['RSI'], name=t("RSI"))])
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text=t("Overbought"), annotation_position="bottom right")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text=t("Oversold"), annotation_position="top right")
                        fig_rsi.update_layout(
                            title_text=t("Relative Strength Index (RSI)"),
                            yaxis_title_text=t("RSI"),
                            template=current_plotly_template, # Use dynamic template
                            font_color=text_color_for_plotly,
                            title_font_color=text_color_for_plotly,
                            xaxis_title_font_color=text_color_for_plotly,
                            yaxis_title_font_color=text_color_for_plotly,
                            legend_font_color=text_color_for_plotly,
                            yaxis=dict(range=[0, 100]),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        fig_rsi.update_xaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
                        fig_rsi.update_yaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    else:
                        st.info(t("Insufficient data to calculate RSI (requires at least 14 data points)."))
                else:
                    st.warning(f"{t('Could not fetch chart data for')} {selected_coin_name} {t('in')} {vs_currency.upper()}. {t('Status:')} {msg}")
        st.markdown('</div>', unsafe_allow_html=True)


def show_coin_details(coin_names, coin_list_dict, vs_currency):
    # --- Custom CSS and logo for consistency ---
    st.markdown(
        """
        <style>
        .metric-card {
            background: var(--card-background);
            border-radius: 18px;
            box-shadow: 0 2px 16px rgba(0,0,0,0.10);
            padding: 1.7rem 1.1rem 1.3rem 1.1rem;
            margin-bottom: 2rem;
            text-align: center;
            color: var(--text-color);
            transition: box-shadow 0.2s, transform 0.2s;
            position: relative;
        }
        .metric-card:hover {box-shadow: 0 6px 24px rgba(0,0,0,0.18);transform: scale(1.03);}
        .section-divider {
            border-top: 2px solid var(--secondary-background-color);
            opacity: 0.5;
            margin: 2.5rem 0 2rem 0;
        }
        .main-bg {
            background: var(--main-bg-gradient);
            border-radius: 24px;
            padding: 2.5rem 2.5rem 2rem 2.5rem;
            margin-bottom: 2.5rem;
        }
        .logo {width: 54px; height: 54px; border-radius: 12px; margin-bottom: 0.5rem;}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.header(t("Coin Details") + " :coin:")
    st.markdown(t("""
    **About Coin Details:**
    - See key stats, links, and a short description for each coin.
    - Use this info to research coins before investing.
    - Check the official website and blockchain explorers for more details.
    """))
    with st.container():
        st.markdown('<div class="main-bg">', unsafe_allow_html=True)
        # ...existing code for top_coin_names_for_selection and selectbox...
        top_coin_names_for_selection = coin_names # Default fallback
        with st.spinner(t("Loading top 20 coins for selection...")):
            df_top_20, msg_top_20 = get_coingecko_markets(vs_currency=vs_currency, limit=20)
            if not df_top_20.empty and 'name' in df_top_20.columns:
                fetched_names = df_top_20['name'].tolist()
                top_coin_names_for_selection = [name for name in fetched_names if name in coin_list_dict]
                if not top_coin_names_for_selection:
                    st.warning(t("Could not map top 20 fetched coins to known coin IDs. Using full list."))
                    top_coin_names_for_selection = coin_names
            elif msg_top_20 != "Success":
                st.warning(f"{t('Could not fetch top 20 coins for selection (Status:')} {msg_top_20}). {t('Using full list.')}")
        selected_coin_name = st.selectbox(
            t("Select Cryptocurrency (Top 20 by Market Cap)"),
            top_coin_names_for_selection,
            index=top_coin_names_for_selection.index("Bitcoin") if "Bitcoin" in top_coin_names_for_selection else 0,
            help=t("Type to search within the top 20 list.")
        )
        coin_id = coin_list_dict.get(selected_coin_name)
        if not coin_id:
            st.error(f"{t('Could not find ID for selected coin')} '{selected_coin_name}'. {t('Please try refreshing or selecting another coin.')}")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        if st.button(t("Get Details")):
            with st.spinner(f"{t('Fetching details for')} {selected_coin_name}..."):
                coin_details, msg = get_coin_details(coin_id)
                if coin_details:
                    col_img, col_info = st.columns([1, 3])
                    with col_img:
                        image_url = coin_details.get('image', {}).get('large')
                        if image_url:
                            st.image(image_url, width=100)
                    with col_info:
                        st.subheader(coin_details.get('name', 'N/A'))
                        st.write(f"{t('Symbol:')} {coin_details.get('symbol', 'N/A').upper()}")
                        st.write(f"{t('CoinGecko Rank:')} {coin_details.get('coingecko_rank', 'N/A')}")
                        categories = coin_details.get('categories', [])
                        if categories:
                            st.write(f"{t('Categories:')} {', '.join(filter(None, categories))}")
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    description = coin_details.get('description', {}).get('en', t('No description available.'))
                    if description and description.strip():
                        st.markdown('<div class="metric-card">📝<br><b>' + t('Description') + '</b><br><span style="font-size:1.05rem;">' + description[:500] + ('...' if len(description) > 500 else '') + '</span></div>', unsafe_allow_html=True)
                    else:
                        st.info(t("No description available for this coin."))
                    st.subheader(t("Links"))
                    links = coin_details.get('links', {})
                    for link_type, display_name in [('homepage', t('Website')), ('blockchain_site', t('Blockchain Explorers')), ('official_forum_url', t('Forums')), ('chat_url', t('Chat Channels'))]:
                        link_list = links.get(link_type, [])
                        if link_list and isinstance(link_list, list):
                            st.write(f"- {display_name}:")
                            count = 0
                            for link in link_list:
                                if link and isinstance(link, str):
                                    st.write(f"  - [{link}]({link})")
                                    count += 1
                                    if count >= 5 and link_type == 'blockchain_site':
                                        break
                                    if count >= 1 and link_type == 'homepage':
                                        break
                            if count == 0:
                                st.write("  - N/A")
                    st.write(f"{t('Genesis Date:')} {coin_details.get('genesis_date', 'N/A')}")
                    st.write(f"{t('Hashing Algorithm:')} {coin_details.get('hashing_algorithm', 'N/A')}")
                else:
                    st.warning(f"{t('Could not fetch details for')} {selected_coin_name}. {t('Status:')} {msg}")
        st.markdown('</div>', unsafe_allow_html=True)


def show_news_sentiment(coin_names, coin_list_dict): # Added coin_names and coin_list_dict
    st.header(t("News and Sentiment Analysis"))

    # Add a selectbox for choosing a cryptocurrency or 'General'
    news_coin_options = [t("General Crypto News")] + coin_names
    if 'news_selected_coin' not in st.session_state:
        st.session_state.news_selected_coin = t("General Crypto News")

    selected_coin_for_news = st.selectbox(
        t("Select Cryptocurrency for News"),
        news_coin_options,
        index=news_coin_options.index(st.session_state.news_selected_coin) if st.session_state.news_selected_coin in news_coin_options else 0,
        key="news_coin_selectbox"
    )
    st.session_state.news_selected_coin = selected_coin_for_news

    # Determine the symbol to pass to the API
    # If "General Crypto News" is selected, pass None or "GENERAL" to fetch all news
    # Otherwise, find the symbol for the selected coin name.
    coin_symbol_for_api = None
    if selected_coin_for_news != t("General Crypto News"):
        # Find the symbol from coin_list_dict based on the selected_coin_for_news name
        # This requires coin_list_dict to map name to ID, and we need a mapping from name to symbol.
        # For CryptoCompare, we often use the symbol directly.
        # We need to ensure coin_list_dict contains 'symbol' or we derive it.
        # For now, let's assume coin_list_dict maps name to ID, and we need to find the symbol.
        # This part might need adjustment based on how coin symbols are stored/retrieved.
        # A simple approach: try to find the symbol in parentheses if it's part of the name, or use the name itself.
        # This is a placeholder and might need a more robust way to get the symbol.
        # For CryptoCompare, the 'categories' parameter usually takes the coin symbol (e.g., BTC, ETH).
        # We need to get the symbol from the selected_coin_name.
        # Let's try to get the ID first, then get details to find the symbol if coin_list_dict doesn't have it directly.
        # This is becoming complex. For now, let's assume the API can take the name or we can derive a symbol.
        # A simpler approach for now: if the API takes symbols, we need a mapping from name to symbol.
        # The get_coingecko_markets returns 'symbol'. We can use that if we fetch it.
        # For now, let's assume the selected_coin_for_news (if not "General") can be used or mapped to a symbol.
        # The CryptoCompare API seems to use uppercase symbols for categories.
        
        # Attempt to get the symbol. This is a simplification.
        # A more robust solution would involve having a clear mapping from coin name to its symbol.
        # For now, we'll try to extract it or use a known mapping if available.
        # We'll use the coin_id to fetch details and then get the symbol if not directly available.
        # However, get_news_with_sentiment is designed to take a symbol.
        # Let's assume for now that the `coin_list_dict` can give us a symbol or we can derive it.
        # We will try to find the symbol from the coin_list_dict (assuming it might store 'symbol' or we adapt)
        # For simplicity, if the selected_coin_for_news is a known name, we try to get its ID,
        # then we'd ideally get its symbol. CryptoCompare uses symbols like BTC, ETH.
        
        # Let's try to get the symbol from coin_list_dict if it stores it, or derive it.
        # This is a common pattern: coin_list_dict maps 'Bitcoin' to 'bitcoin' (ID). We need 'BTC'.
        # We might need to enhance get_coin_list to also return symbols.
        # For now, let's pass the name and let get_news_with_sentiment handle it if it can,
        # or we assume the API takes the name. The current get_news_with_sentiment expects a symbol.
        
        # Find the ID, then try to find the symbol.
        # This is a placeholder for a more robust symbol retrieval.
        # We'll assume the coin name can be mapped to a symbol.
        # For CryptoCompare, it's usually the uppercase symbol.
        # We need a reliable way to get the symbol from the name.
        # Let's try to get it from coin_list_dict if it has 'symbol', or use the name as a fallback.
        # This part is tricky without knowing the exact structure of coin_list_dict for symbols.
        # We will assume for now that the API can handle the coin name or a derived symbol.
        # The `get_news_with_sentiment` function was updated to take `coin_symbol`.
        # We need to find the symbol for `selected_coin_for_news`.
        # A common way is to iterate through `coin_list_dict` if it contains symbols.
        # Or, if `coin_list_dict` maps name to ID, we might need another source for name to symbol.
        # Let's assume `coin_list_dict` is { 'Bitcoin': {'id': 'bitcoin', 'symbol': 'btc'}, ... }
        # Or that `get_coingecko_markets` can provide this.
        # For now, we'll try a simple approach:
        coin_id_lookup = coin_list_dict.get(selected_coin_for_news)
        if coin_id_lookup: # If it's a direct ID
            # This assumes coin_id_lookup might be the symbol or we need to fetch it.
            # The API expects a symbol like 'BTC'.
            # This is a placeholder. A robust solution needs a clear name-to-symbol mapping.
            # For now, we'll try to use the ID if it looks like a symbol, or fetch details.
            # This is simplified:
            if selected_coin_for_news == "Bitcoin": coin_symbol_for_api = "BTC"
            elif selected_coin_for_news == "Ethereum": coin_symbol_for_api = "ETH"
            # Add more common ones or a better lookup mechanism
            else: # Fallback, might not work well for all coins
                coin_symbol_for_api = selected_coin_for_news.split(" ")[0].upper() 
        else: # If not found, it might be a symbol already or needs better handling
             coin_symbol_for_api = selected_coin_for_news.upper()


    active_theme_mode = st.session_state.get('theme_mode', 'Dark')
    current_plotly_template = 'plotly_white' if active_theme_mode == 'Light' else 'plotly_dark'
    text_color_for_plotly = '#212529' if active_theme_mode == 'Light' else '#FAFAFA'

    st.markdown(t("""
    **What is Sentiment Analysis?**
    This section analyzes the tone of recent crypto news headlines. Positive sentiment may indicate bullish trends, while negative sentiment can signal caution.
    
    **How to use:**
    - Watch for shifts in average sentiment.
    - Read the latest news to understand what’s driving the market mood.
    """))

    # Add a refresh button
    if st.button(t("Refresh News"), key="news_refresh_button"):
        # Clear cache for the specific coin news or general news
        if coin_symbol_for_api:
            get_news_with_sentiment.clear(coin_symbol=coin_symbol_for_api)
        else:
            get_news_with_sentiment.clear(coin_symbol=None) # Or however general news is cached

    # Fetch news based on selection
    with st.spinner(f"{t('Fetching news for')} {selected_coin_for_news}..."):
        df_news, msg = get_news_with_sentiment(coin_symbol=coin_symbol_for_api)

    if msg and msg != "Success" and "No news data received" not in msg and "No news data processed" not in msg :
        st.warning(f"{t('Could not fetch news data for')} {selected_coin_for_news}. {t('Status:')} {msg}")
        # Display cached data if available from a previous successful run for this coin
        cache_key_news = f"news_data_{selected_coin_for_news}"
        if cache_key_news in st.session_state:
            df_news, _ = st.session_state[cache_key_news]
            if not df_news.empty:
                st.info(t("Displaying previously cached news data due to current fetch error."))
            else:
                return # Nothing to show
        else:
            return # Nothing to show

    if not df_news.empty:
        st.session_state[f"news_data_{selected_coin_for_news}"] = (df_news, msg) # Cache successful fetch
        st.subheader(f"{t('News Sentiment Metrics for')} {selected_coin_for_news}")
        avg_sentiment = df_news['sentiment'].mean()
        # Define thresholds clearly
        positive_threshold = 0.05
        negative_threshold = -0.05
        positive_news = (df_news['sentiment'] > positive_threshold).sum()
        negative_news = (df_news['sentiment'] < negative_threshold).sum()
        neutral_news = ((df_news['sentiment'] >= negative_threshold) & (df_news['sentiment'] <= positive_threshold)).sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1: custom_metric(t("Average Sentiment"), f"{avg_sentiment:.3f}", "")
        with col2: custom_metric(t("Positive News"), positive_news, "")
        with col3: custom_metric(t("Negative News"), negative_news, "")
        with col4: custom_metric(t("Neutral News"), neutral_news, "")

        st.subheader(t("News Sentiment Distribution"))

        # --- Manual Binning and Bar Chart Creation ---
        num_bins = 30
        bin_labels = [f'{i:.2f} to {j:.2f}' for i, j in zip(np.linspace(-1, 1, num_bins + 1)[:-1], np.linspace(-1, 1, num_bins + 1)[1:])]
        df_news['sentiment_bin'] = pd.cut(df_news['sentiment'], bins=np.linspace(-1, 1, num_bins + 1), labels=bin_labels, right=True, include_lowest=True)

        # Aggregate data for the bar chart
        sentiment_dist = df_news.groupby('sentiment_bin', observed=False).agg(
            count=('title', 'size'),
            titles=('title', lambda x: '<br>- '.join(x.head(5)))
        ).reset_index()
        # ...existing code continues...

        fig_sentiment_dist = go.Figure(data=[
            go.Bar(
                x=sentiment_dist['sentiment_bin'],
                y=sentiment_dist['count'],
                text=sentiment_dist['titles'],
                hovertemplate='<b>%{x}</b><br>' + t('Count:') + ' %{y}<br>' + t('Titles:') + '<br>%{text}',
                marker_color='blue'
            )
        ])
        fig_sentiment_dist.update_layout(
            title_text=t("Distribution of News Sentiment"),
            xaxis_title_text=t("Sentiment Range"),
            yaxis_title_text=t("Number of Articles"),
            template=current_plotly_template, # Use dynamic template
            font_color=text_color_for_plotly,
            title_font_color=text_color_for_plotly,
            xaxis_title_font_color=text_color_for_plotly,
            yaxis_title_font_color=text_color_for_plotly,
            legend_font_color=text_color_for_plotly,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_sentiment_dist.update_xaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
        fig_sentiment_dist.update_yaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
        st.plotly_chart(fig_sentiment_dist, use_container_width=True)

        st.subheader(t("Latest News"))
        for _, row in df_news.iterrows():
            sentiment_color = "green" if row['sentiment'] > positive_threshold else "red" if row['sentiment'] < negative_threshold else "gray"
            st.markdown(f"**[{row['title']}]({row['url']})**")
            st.write(f"{t('Source:')} {row['source']} | {t('Published:')} {row['published_at'].strftime('%Y-%m-%d %H:%M:%S')} | {t('Sentiment:')} {row['sentiment']:.3f}")
            st.markdown(f"<hr style='border-color: {sentiment_color};'>", unsafe_allow_html=True)
    else:
        st.warning(f"{t('Could not fetch news data. Status:')} {msg}")


def show_exchanges_derivatives():
    st.header(t("Exchanges and Derivatives"))
    st.markdown(t("""
    **What are Exchanges and Derivatives?**
    - **Exchanges:** Platforms where you can buy, sell, and trade cryptocurrencies.
    - **Derivatives:** Financial contracts that derive their value from an underlying asset, like futures and options.
    
    **How to use:**
    - Compare exchanges by trust score, volume, and other metrics.
    - Explore derivative exchanges for advanced trading options.
    """))
    with st.spinner(t("Fetching exchanges data...")):
        df_exchanges, msg_exchanges = get_exchanges_list()
        df_derivatives, msg_derivatives = get_derivative_exchanges()

    if not df_exchanges.empty:
        st.subheader(t("Top Cryptocurrency Exchanges"))
        st.dataframe(df_exchanges.style.format({
            'trade_volume_24h_btc': '{:,.2f} BTC',
            'trust_score': '{:.0f}'
        }), use_container_width=True)
    else:
        st.warning(f"{t('Could not fetch exchanges data. Status:')} {msg_exchanges}")

    if not df_derivatives.empty:
        st.subheader(t("Top Derivative Exchanges"))
        st.dataframe(df_derivatives.style.format({
            'open_interest_btc': '{:,.2f} BTC',
            'number_of_perpetual_pairs': '{:.0f}',
            'number_of_futures_pairs': '{:.0f}'
        }), use_container_width=True)
    else:
        st.warning(f"{t('Could not fetch derivative exchanges data. Status:')} {msg_derivatives}")


def show_price_alerts(coin_names, coin_list_dict, vs_currency):
    st.header(t("Price Alerts") + " 🔔")
    st.markdown(t("""
    **What are Price Alerts?**
    Set alerts to get notified when a cryptocurrency's price crosses a certain threshold. This helps you stay informed and make timely decisions.
    
    **How to use:**
    - Select a coin and set high and low price thresholds.
    - Enter your email and/or phone number to receive notifications.
    - Click "Set Alert" to save your alert.
    """))
    with st.container():
        st.markdown('<div class="main-bg">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            selected_coin_name = st.selectbox(
                t("Select Cryptocurrency"),
                coin_names,
                index=coin_names.index("Bitcoin") if "Bitcoin" in coin_names else 0,
                help=t("Choose a coin to set alerts for.")
            )
        with col2:
            vs_currency = st.selectbox(
                t("Select Currency"),
                ["usd", "eur", "gbp", "btc"],
                index=0,
                help=t("Choose the currency for price alerts.")
            )
        coin_id = coin_list_dict.get(selected_coin_name, selected_coin_name.lower().replace(" ", "-"))
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        email = st.text_input(t("Email for Alerts (optional)"), help=t("Enter your email to receive price alerts."))
        phone_number = st.text_input(t("Phone Number for SMS Alerts (optional)"), help=t("Enter your phone number to receive SMS alerts."))
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # --- Get current price for setting defaults and checking alerts ---
        st.subheader(t("Current Price & Alert Check"))
        current_price = None
        msg_current_price = t("Not fetched yet") # Default message

        cooldown_duration_alert = 120  # 2 minutes
        cooldown_key_alert = f"price_alert_cooldown_{coin_id}_{vs_currency}"
        now = time.time()
        cooldown_until_alert = st.session_state.get(cooldown_key_alert, 0)
        is_cooldown_active_alert = now < cooldown_until_alert

        if is_cooldown_active_alert:
            remaining_time_alert = int(cooldown_until_alert - now)
            st.info(f"{t('API rate limit previously reached for current price. Please wait')} {remaining_time_alert} {t('seconds.')}")
            # Try to use last known price from session state if in cooldown
            current_price = st.session_state.get(f'last_known_price_{coin_id}_{vs_currency}', 0)
            if current_price != 0:
                 st.metric(t("Last Known Price"), f"{current_price:,.8f} {vs_currency.upper()}")
                 msg_current_price = t("Displaying last known price due to cooldown.")
            else:
                msg_current_price = t("Current price fetch is on cooldown, and no last known price available.")
        else:
            with st.spinner(f"{t('Fetching current price for')} {selected_coin_name}..."):
                prices, msg_fetch = get_current_coin_price(coin_id, vs_currency)
                current_price = prices.get(coin_id)
                msg_current_price = msg_fetch # Update message with fetch result

            if msg_current_price and ("429" in msg_current_price or "Too Many Requests" in msg_current_price or "rate limit" in msg_current_price.lower()):
                st.session_state[cooldown_key_alert] = time.time() + cooldown_duration_alert
                st.error(f"{t('API rate limit reached for current price')}: {msg_current_price}. {t('Please wait')} {cooldown_duration_alert // 60} {t('minutes.')}")
                # Use last known price if available, otherwise set to 0 for defaults
                current_price = st.session_state.get(f'last_known_price_{coin_id}_{vs_currency}', 0)
            elif current_price is not None and msg_current_price == "Success":
                st.metric(t("Current Price"), f"{current_price:,.8f} {vs_currency.upper()}")
                st.session_state[f'last_known_price_{coin_id}_{vs_currency}'] = current_price # Store successfully fetched price
            elif current_price is None: # Fetch failed for non-429 reason or partial success without this coin
                st.warning(f"{t('Could not retrieve current price data for')} {selected_coin_name}. {t('Status:')} {msg_current_price}")
                current_price = st.session_state.get(f'last_known_price_{coin_id}_{vs_currency}', 0) # Fallback to last known
                if current_price == 0: # If no last known price either
                    st.info(t("Defaulting alert thresholds as current price is unavailable."))


        # Ensure current_price is a float for calculations, default to 0 if it's None after all checks
        display_price_for_defaults = current_price if isinstance(current_price, (int, float)) else 0
        
        # Set alert thresholds (using potentially fetched/last_known/fallback current_price)
        col1, col2 = st.columns(2)
        with col1:
            default_high = display_price_for_defaults * 1.1 if display_price_for_defaults > 0 else 100000.0
            high_threshold = st.number_input(f"{t('Alert ABOVE')} ({vs_currency.upper()})", value=float(default_high), format="%.8f", step=float(max(0.00000001, default_high * 0.01 if default_high > 0 else 1000.0)), min_value=0.0)
        with col2:
            default_low = display_price_for_defaults * 0.9 if display_price_for_defaults > 0 else 80000.0
            default_low = max(0.0, default_low) # Ensure low threshold is not negative
            low_threshold = st.number_input(f"{t('Alert BELOW')} ({vs_currency.upper()})", value=float(default_low), format="%.8f", step=float(max(0.00000001, default_low * 0.01 if default_low > 0 else 1000.0)), min_value=0.0)

        # Check alerts only if current_price was successfully obtained (not None and not in cooldown for initial fetch)
        # msg_current_price can be "Success" or "Partial Success..."
        if isinstance(current_price, (int, float)) and (msg_current_price == "Success" or "Partial Success" in msg_current_price):
            alerts = check_price_alerts(current_price, {'high': high_threshold, 'low': low_threshold})
            if alerts:
                st.subheader(t("Active Alerts Triggered"))
                for alert in alerts:
                    alert_message = f"{selected_coin_name.upper()} {t('price')} ({alert['price']:,.8f} {vs_currency.upper()}) {t('has gone')} {'above' if alert['type'] == 'high' else 'below'} {t('your threshold of')} {alert['threshold']:,.8f} {vs_currency.upper()}."
                    st.warning(f"{t('TRIGGERED:')} {alert_message}")
                    # Attempt to send notification only if email is provided
                    if email:
                        send_alert_notification(email, alert_message)
                    # Attempt to send SMS notification only if phone number is provided
                    if phone_number:
                        send_sms_notification(phone_number, alert_message)
            else:
                 st.info(t("No price alerts triggered based on the current price and thresholds."))
        elif msg_current_price != t("Not fetched yet"):
             # Message already displayed by the warning above if fetch failed
             st.info(t("Price alerts could not be checked as current price fetch failed."))


# Add a portfolio tracker feature
def show_portfolio_tracker():
    st.header(t("Portfolio Tracker"))
    st.markdown(t("""
    **What is a Portfolio?**
    Your portfolio tracks the value of your crypto holdings in real time. Diversification can help manage risk.
    
    **How to use:**
    - Add your coins and amounts.
    - Monitor your total value and individual coin performance.
    - Use this tool to track gains, losses, and allocation.
    """))
    # Initialize session state for portfolio
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []

    # Add a new holding
    with st.form("add_holding_form"):
        st.subheader(t("Add a New Holding"))
        coin = st.text_input(t("Cryptocurrency (e.g., Bitcoin, Ethereum)"))
        amount = st.number_input(t("Amount Held"), min_value=0.0, step=0.01)
        submit = st.form_submit_button(t("Add Holding"))

        if submit and coin and amount > 0:
            st.session_state["portfolio"].append({"coin": coin, "amount": amount})
            st.success(f"{t('Added')} {amount} {t('of')} {coin} {t('to your portfolio.')}")

    # Display portfolio
    if st.session_state["portfolio"]:
        st.subheader(t("Your Portfolio"))
        portfolio_df = pd.DataFrame(st.session_state["portfolio"])

        # Fetch current prices for portfolio coins
        coins = [entry["coin"] for entry in st.session_state["portfolio"]]
        prices, _ = get_current_coin_price(coins, vs_currency="usd")

        # Calculate portfolio value
        portfolio_df["Current Price (USD)"] = portfolio_df["coin"].map(prices)
        portfolio_df["Value (USD)"] = portfolio_df["amount"] * portfolio_df["Current Price (USD)"]

        st.dataframe(portfolio_df)

        total_value = portfolio_df["Value (USD)"].sum()
        st.metric(t("Total Portfolio Value (USD)"), f"${total_value:,.2f}")
    else:
        st.info(t("Your portfolio is empty. Add holdings to start tracking."))


# Enhance data visualization with zoomable time-series charts
def plot_zoomable_time_series(df, title, y_column, y_label):
    """
    Plot a zoomable time-series chart using Plotly.

    Args:
        df (pd.DataFrame): DataFrame containing the time-series data.
        title (str): Title of the chart.
        y_column (str): Column name for the y-axis data.
        y_label (str): Label for the y-axis.

    Returns:
        None
    """
    if df.empty or y_column not in df.columns:
        st.warning(f"{t('No data available for')} {title}.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[y_column], mode='lines', name=y_label))

    active_theme_mode = st.session_state.get('theme_mode', 'Dark') # Get current theme
    current_plotly_template = 'plotly_white' if active_theme_mode == 'Light' else 'plotly_dark'
    text_color_for_plotly = '#212529' if active_theme_mode == 'Light' else '#FAFAFA'

    fig.update_layout(
        title_text=title,
        xaxis_title_text=t("Date"),
        yaxis_title_text=y_label,
        template=current_plotly_template, # Use dynamic template
        font_color=text_color_for_plotly,
        title_font_color=text_color_for_plotly,
        xaxis_title_font_color=text_color_for_plotly,
        yaxis_title_font_color=text_color_for_plotly,
        legend_font_color=text_color_for_plotly,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=True
    )
    fig.update_xaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
    fig.update_yaxes(tickfont=dict(color=text_color_for_plotly), gridcolor=text_color_for_plotly if active_theme_mode == 'Light' else 'rgba(128,128,128,0.2)')
    st.plotly_chart(fig, use_container_width=True)

# Add social media sentiment analysis using Twitter and Reddit APIs
def fetch_social_media_sentiment():
    """Fetch and analyze sentiment from Twitter and Reddit."""
    st.header(t("Social Media Sentiment Analysis"))

    st.subheader(t("Twitter Sentiment"))
    twitter_query = st.text_input(t("Enter a keyword or hashtag for Twitter analysis"), "#Bitcoin")
    if st.button(t("Analyze Twitter Sentiment")):
        with st.spinner(t("Fetching and analyzing tweets...")):
            # Placeholder for Twitter API integration
            st.info(t("Twitter API integration is required. Replace this placeholder with actual API calls."))

    st.subheader(t("Reddit Sentiment"))
    reddit_query = st.text_input(t("Enter a subreddit or keyword for Reddit analysis"), "cryptocurrency")
    if st.button(t("Analyze Reddit Sentiment")):
        with st.spinner(t("Fetching and analyzing Reddit posts...")):
            # Placeholder for Reddit API integration
            st.info(t("Reddit API integration is required. Replace this placeholder with actual API calls."))

# --- Streamlit App Layout (Main Function) ---

def update_language():
    """Update the app's language setting based on the canonical key from the selector."""
    # 'language_selector' now holds the canonical key (e.g., "English", "Spanish")
    selected_language_key = st.session_state.get('language_selector')
    if selected_language_key and st.session_state.get('language') != selected_language_key:
        st.session_state.language = selected_language_key
        # Streamlit automatically reruns the script when a widget with an on_change callback is changed.
        # Explicitly calling st.rerun() here is unnecessary and causes the "no-op" warning.
        pass # No explicit rerun needed.

def route_page(page, selected_translations, coin_names, coin_list_dict, vs_currency):
    """Route to the appropriate page function based on the selected page"""
    if page == selected_translations["Market Overview"]:
        show_market_overview(vs_currency)
    elif page == selected_translations["Technical Analysis"]:
        show_technical_analysis(coin_names, coin_list_dict, vs_currency)
    elif page == selected_translations["Trend Analysis"]:
        show_trend_analysis(coin_list_dict, vs_currency)
    elif page == selected_translations["Coin Details"]:
        show_coin_details(coin_names, coin_list_dict, vs_currency)
    elif page == selected_translations["News & Sentiment"]:
        show_news_sentiment(coin_names, coin_list_dict) # Pass coin_names and coin_list_dict
    elif page == selected_translations["Exchanges and Derivatives"]:
        show_exchanges_derivatives()
    elif page == selected_translations["Price Alerts"]:
        show_price_alerts(coin_names, coin_list_dict, vs_currency)
    elif page == selected_translations["Help / About"]:
        show_help_section()
    elif page == "Portfolio Tracker":
        show_portfolio_tracker()

def main():
    global coin_names  # Make coin_names available to other functions
    
    st.set_page_config(page_title="Crypto Monitor", page_icon="📈", layout="wide")
    st.title(t("Cryptocurrency Monitor and Analysis"))

    # --- Initial Data Loading ---
    api_error = False
    all_supported_currencies, msg_currency = get_supported_vs_currencies()
    if not all_supported_currencies or "429" in str(msg_currency):
        api_error = True
        all_supported_currencies = ['usd'] # Fallback
        st.warning("⚠️ " + t("The app is experiencing API rate limits. Please wait a few minutes and try again."))

    # Define a list of common/top fiat currency codes
    # This list aims to cover major and frequently used fiat currencies.
    # CoinGecko's supported list will be the ultimate filter.
    known_fiat_currencies = [
        'usd', 'eur', 'jpy', 'gbp', 'aud', 'cad', 'chf', 'cny', 'hkd', 'nzd',
        'sek', 'krw', 'sgd', 'nok', 'mxn', 'inr', 'rub', 'zar', 'try', 'brl',
        'twd', 'dkk', 'pln', 'thb', 'idr', 'huf', 'czk', 'ils', 'clp', 'php',
        'aed', 'cop', 'sar', 'myr', 'ron', 'vnd', 'ngn', 'ars', 'uah', 'iqd',
        'pyg', 'pen', 'egp', 'pkr', 'bdt', 'vef', 'qar', 'kwd', 'omr', 'bhd'
        # Add more as needed, up to a reasonable number like 50-60
    ]

    # Filter CoinGecko's supported currencies to only include our known fiat currencies
    # And ensure they are lowercase for consistent matching
    supported_fiat_currencies = [
        currency.lower() for currency in all_supported_currencies
        if currency.lower() in known_fiat_currencies
    ]

    if not supported_fiat_currencies: # Fallback if intersection is empty
        supported_fiat_currencies = ['usd']
        st.warning("Could not determine a list of supported fiat currencies, defaulting to USD.")


    # Initialize coin list and names
    coin_list_dict, msg_coin_list = get_coin_list()
    if not coin_list_dict or "429" in str(msg_coin_list):
        api_error = True
        coin_list_dict = {"Bitcoin": "bitcoin", "Ethereum": "ethereum"}
    coin_names = sorted(list(coin_list_dict.keys())) if coin_list_dict else ["Bitcoin", "Ethereum"]

    # Rest of main() function remains the same...

    # --- Sidebar Navigation ---
    # Add support for multiple languages and currencies

    # Language selection
    language_keys = ["English", "Spanish", "French", "German", "Chinese"]
    # Get the language key from session state to set the default index
    current_language_key_from_state = st.session_state.get('language', 'English')
    current_language_index = language_keys.index(current_language_key_from_state) if current_language_key_from_state in language_keys else 0

    # Define a formatting function that translates the language key based on the *current* app language state
    def format_language_option(lang_key):
        # Use the session state language for formatting the display options
        active_lang = st.session_state.get('language', 'English')
        active_lang_translations = TRANSLATIONS.get(active_lang, TRANSLATIONS['English'])
        return active_lang_translations.get(lang_key, lang_key)

    # The 'language' variable will hold the canonical key selected *in this run*
    language = st.sidebar.selectbox(
        t("Select Language"), # Label uses the current session state language via t()
        options=language_keys,
        index=current_language_index, # Default index based on session state
        key='language_selector', # Widget state key
        format_func=format_language_option, # Display formatting uses session state language
        on_change=update_language # Callback updates session state for the *next* run
    )

    # Use the 'language' variable (reflecting the current selection) for this run's translations
    selected_translations = TRANSLATIONS.get(language, TRANSLATIONS["English"])

    # Currency selection (already implemented but enhanced for localization)
    if supported_fiat_currencies:
        # Ensure 'usd' is an option if available, otherwise use the first in the list as default
        default_fiat = 'usd' if 'usd' in supported_fiat_currencies else supported_fiat_currencies[0]
        default_index = supported_fiat_currencies.index(default_fiat)
        vs_currency = st.sidebar.selectbox(
            t("Select Fiat Currency"),
            options=sorted(list(set(s.upper() for s in supported_fiat_currencies))), # Show unique, sorted, uppercase
            index=default_index
        ).lower() # Convert back to lower for API calls
    else:
        vs_currency = 'usd' # Ultimate fallback
        st.sidebar.warning(t("Could not fetch supported fiat currencies. Using USD."))

    # Apply translations based on selected language
    # The 'language' variable is derived from st.session_state.language_selector (via the selectbox),
    # which is updated by the on_change callback 'update_language'.
    # TRANSLATIONS is the global dictionary containing all translations.
    selected_translations = TRANSLATIONS.get(language, TRANSLATIONS["English"])

    # Update all page headers and subheaders dynamically based on the selected language
    page = st.sidebar.selectbox(
        selected_translations["Choose Analysis"],
        [
            selected_translations["Market Overview"],
            selected_translations["Technical Analysis"],
            selected_translations["Trend Analysis"],
            selected_translations["Coin Details"],
            selected_translations["News & Sentiment"],
            selected_translations["Exchanges and Derivatives"],
            selected_translations["Price Alerts"],
            selected_translations["Help / About"]
        ]
    )

    # Initialize canonical theme mode in session state if not already set
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'Dark' 

    # Theme selector radio button
    # Its value will be available in st.session_state.theme_radio_widget_value due to the key
    selected_theme_from_widget = st.sidebar.radio(
        selected_translations["Select Theme Mode"], # Use translated label
        options=["Dark", "Light"], # Direct values for options
        index=["Dark", "Light"].index(st.session_state.theme_mode), # Set index from canonical state
        key="theme_radio_widget_value" # Key for this specific radio button widget
    )

    # If the user's selection via the radio button differs from our stored canonical theme,
    # update the canonical theme and force a rerun to apply changes.
    if st.session_state.theme_mode != selected_theme_from_widget:
        st.session_state.theme_mode = selected_theme_from_widget
        st.rerun()

    # Apply theme CSS based on the canonical theme_mode from session state
    active_theme_mode = st.session_state.theme_mode
    if active_theme_mode == "Dark":
        st.markdown(
            """<style>
            :root {
                --primary-color: #FF4B4B;
                --background-color: #0E1117;
                --secondary-background-color: #262730;
                --text-color: #FAFAFA; /* White text for dark bg */
                --card-background: linear-gradient(135deg, #23272f 60%, #2d3140 100%);
                --main-bg-gradient: linear-gradient(120deg, #181c24 60%, #23272f 100%);
                --card-label-color: #b0b8c1; /* Light gray for card labels on dark bg */
                --plotly-template: 'plotly_dark';
            }
            body, .stApp {
                background-color: var(--background-color) !important;
                color: var(--text-color) !important;
            }
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
            .stApp p, .stApp div, .stApp span, .stApp li, .stApp label, 
            .stApp .stMarkdownContainer, .stApp .stTextInput > div > div > input,
            .stApp .stSelectbox > div > div, .stApp .stRadio > label > div {
                color: var(--text-color) !important;
            }
            /* Ensure Plotly chart backgrounds are transparent */
            .js-plotly-plot .plotly .main-svg, .js-plotly-plot .plotly .main-svg .bg {
                fill: transparent !important;
                background-color: transparent !important;
            }
            </style>""",
            unsafe_allow_html=True
        )
    else: # Light mode
        st.markdown(
            """<style>
            :root {
                --primary-color: #FF4B4B;
                --background-color: #FFFFFF; /* White background */
                --secondary-background-color: #F0F0F0; /* Light gray for secondary elements */
                --text-color: #212529; /* Dark text for light bg (Bootstrap's default body color) */
                --card-background: linear-gradient(135deg, #f8f9fa 60%, #e9ecef 100%); /* Lighter card bg */
                --main-bg-gradient: linear-gradient(120deg, #ffffff 60%, #f8f9fa 100%); /* Lighter main bg */
                --card-label-color: #495057; /* Darker gray for card labels on light bg */
                --plotly-template: 'plotly_white';
            }
            body, .stApp {
                background-color: var(--background-color) !important;
                color: var(--text-color) !important;
            }
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
            .stApp p, .stApp div, .stApp span, .stApp li, .stApp label,
            .stApp .stMarkdownContainer, .stApp .stTextInput > div > div > input,
            .stApp .stSelectbox > div > div, .stApp .stRadio > label > div {
                color: var(--text-color) !important;
            }
            /* Ensure Plotly chart backgrounds are transparent */
            .js-plotly-plot .plotly .main-svg, .js-plotly-plot .plotly .main-svg .bg {
                fill: transparent !important;
                background-color: transparent !important;
            }
            </style>""",
            unsafe_allow_html=True
        )

    # --- Page Content Routing ---
    if api_error:
        st.error(t("The app is currently rate-limited. Please wait a few minutes."))
    route_page(page, selected_translations, coin_names, coin_list_dict, vs_currency)

    # Add Help/About section to the sidebar navigation
    if page == "Help / About":
        show_help_section()


# Add a Help/About section to guide users
def show_help_section():
    """Displays the Help/About section."""
    st.header(t("Help / About"))

    st.subheader(t("About the App"))
    st.write(
        t("Crypto Monitor provides tools for tracking, analyzing, and staying updated on cryptocurrency markets. "
        "Features include market overviews, technical analysis, price alerts, news sentiment, and portfolio tracking.\n\n**Made by HAMZA TEBRI**")
    )

    st.subheader(t("How to Use"))
    st.markdown(
        "- **" + t("Market Overview") + "**: " + t("View top cryptocurrencies and global market data.") + "\n"
        "- **" + t("Technical Analysis") + "**: " + t("Analyze price trends and indicators.") + "\n"
        "- **" + t("Portfolio Tracker") + "**: " + t("Track your holdings and performance.") + "\n"
        "- **" + t("Price Alerts") + "**: " + t("Set thresholds to receive notifications.") + "\n"
        "- **" + t("News & Sentiment") + "**: " + t("Monitor the latest news and sentiment.") + "\n"
        "- **" + t("Exchanges & Derivatives") + "**: " + t("Explore trading platforms and derivatives.")
    )

    st.subheader(t("Contact"))
    st.write(
        t("For support or feedback, contact: [support@cryptomonitor.com](mailto:support@cryptomonitor.com).")
    )


if __name__ == "__main__":
    main()
