Crypto Monitor: Your Comprehensive Cryptocurrency Analysis & Alert Platform

📊 Project Overview
Crypto Monitor is an interactive Streamlit web application designed to provide users with essential tools for monitoring, analyzing, and staying informed about the dynamic cryptocurrency market. From real-time market data and advanced technical analysis to news sentiment and personalized price alerts, this platform aims to be a go-to resource for investors, traders, and crypto enthusiasts.

💡 Business Need Addressed
The cryptocurrency market is characterized by extreme volatility, rapid developments, and a vast amount of distributed information. This often leads to challenges for individuals in:

Information Overload: Difficulty in aggregating and making sense of disparate data points.
Timely Decision-Making: Missing critical price movements or sentiment shifts due to delayed access to information.
Risk Management: Lack of comprehensive analytical tools to assess market health and asset trends.
Portfolio Oversight: Manual tracking of diverse crypto holdings can be tedious and prone to errors.
Crypto Monitor provides a centralized, user-friendly solution to these problems, enabling users to efficiently monitor, analyze, and react to market dynamics with greater confidence.

✨ Features
Crypto Monitor offers a robust set of functionalities, organized into intuitive sections:

Market Overview:

Displays global cryptocurrency market capitalization, 24-hour trading volume, and Bitcoin dominance.
Lists top cryptocurrencies by market capitalization with real-time prices, volume, and 24-hour percentage changes.
Visualizes market cap and daily price changes through interactive bar charts.
Features a donut chart showing a selected coin's market cap share.
Technical Analysis:

Plots historical price data for selected cryptocurrencies over customizable periods.
Integrates popular technical indicators: Simple Moving Average (SMA), Bollinger Bands, Relative Strength Index (RSI), and MACD.
Includes a linear trendline and volatility chart to aid in trend identification and risk assessment.
Trend Analysis:

A focused view on price trends, moving averages (SMA, EMA), and overall market direction for a selected coin.
Complements Technical Analysis by highlighting the long-term price trajectory and underlying momentum.
Coin Details:

Provides in-depth information for any selected cryptocurrency, including its description, official links (website, blockchain explorers, forums), categories, genesis date, and hashing algorithm.
Offers essential background for researching potential investments.
News & Sentiment Analysis:

Fetches recent cryptocurrency news, filtered by "General Crypto News" or specific coin symbols.
Performs sentiment analysis on news headlines and body text using NLP (TextBlob).
Displays average sentiment, counts of positive/negative/neutral articles, and a distribution chart of sentiment scores.
Presents a scrollable list of the latest news articles with direct links and sentiment indicators.
Exchanges & Derivatives:

Lists top cryptocurrency spot exchanges by trust score and 24-hour trading volume.
Lists top derivative exchanges by open interest and number of perpetual/futures pairs.
Provides transparency into the trading infrastructure.
Price Alerts:

Allows users to set custom "Alert ABOVE" and "Alert BELOW" price thresholds for any cryptocurrency.
Provides optional email and SMS notifications when thresholds are breached, leveraging SMTP and Twilio.
Displays the current price for easy alert setting.
Portfolio Tracker:

Enables users to add and track their cryptocurrency holdings (coin name and amount).
Calculates and displays the real-time current value of individual holdings and the total portfolio value in USD.
Multi-language Support:

Supports English, Spanish, French, German, and Chinese for enhanced accessibility.
Dynamic Theme Switching:

Allows users to toggle between Light and Dark modes to suit their preference.
🛠️ Technical Requirements & Implementation Details
External APIs Used (at least 2 required):

CoinGecko API: The primary data source for market data, historical charts, coin details, supported currencies, and global market statistics.
CryptoCompare API: Used exclusively for fetching cryptocurrency news and providing content for sentiment analysis.
Twilio Messaging API: Integrated for sending SMS price alerts (requires API key setup).
SMTP (via smtplib): Used for sending email price alerts (requires SMTP server configuration).
Distinct API Endpoints (at least 6 required):

/coins/markets (CoinGecko: Top coins market data)
/coins/{id}/market_chart (CoinGecko: Historical price/volume data)
/coins/{id} (CoinGecko: Detailed coin information)
/global (CoinGecko: Global market statistics)
/simple/price (CoinGecko: Current price for specific coins)
/simple/supported_vs_currencies (CoinGecko: List of supported fiat currencies)
/coins/list (CoinGecko: List of all supported coins by ID)
/exchanges (CoinGecko: Top spot exchanges)
/derivatives/exchanges (CoinGecko: Top derivative exchanges)
/data/v2/news/ (CryptoCompare: Cryptocurrency news feed)
Twilio's Send Message API (for SMS)
Pure HTTP Requests: All API interactions are implemented using the requests library for direct HTTP GET calls, avoiding client SDKs.

Streamlit Secret Management: All sensitive API keys and credentials (e.g., CRYPTOCOMPARE_API_KEY, Twilio ACCOUNT_SID, AUTH_TOKEN, PHONE_NUMBER, email ADDRESS, PASSWORD, SMTP_SERVER, SMTP_PORT) are securely stored and accessed via Streamlit's st.secrets.

User Interaction: The application heavily relies on Streamlit widgets such as st.selectbox, st.multiselect, st.slider, st.text_input, st.number_input, st.button, and st.radio for dynamic user input and control.

Data Analysis:

Correlation Analysis: Calculates and visualizes the correlation matrix of daily returns between selected cryptocurrencies using Pandas and Plotly.
Technical Indicators: Computes SMA, Bollinger Bands, RSI, and MACD using Pandas and NumPy for price trend and momentum analysis.
Sentiment Analysis: Applies TextBlob for natural language processing to gauge the polarity of news articles.
Volatility Calculation: Measures rolling volatility from historical price data.
Data Visualization: Utilizes Plotly Graph Objects and Plotly Express to create interactive and aesthetically pleasing charts, including:

Bar charts for market cap and 24h price change.
Donut chart for market cap share.
Line charts for price trends, moving averages, Bollinger Bands, RSI, MACD, and volatility.
Heatmap for correlation matrix.
Caching (st.cache_data): Implemented strategically to optimize API calls and improve application responsiveness by caching frequently accessed data for specific time-to-live (TTL) durations (e.g., 5 minutes for market data, 1 minute for chart data, 10 minutes for exchanges, static data indefinitely).

Rate Limit Handling: Basic cooldown mechanisms are implemented for CoinGecko API calls (e.g., 2 minutes) to prevent hitting rate limits too frequently, displaying informative messages to the user.

🚀 How to Run Locally
To run this application on your local machine, follow these steps:

Clone the Repository:

Bash
git clone https://github.com/hamzatebri/Crypto_monitor.git
cd Crypto_monitor
Create a Virtual Environment (Recommended):

Bash

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install Dependencies:

Bash

pip install -r requirements.txt
(If requirements.txt is not provided, you'll need to create one. Here's what it should roughly contain based on the code:)

streamlit
pandas
numpy
requests
plotly
scikit-learn
textblob
# For email alerts (smtplib is built-in Python standard library, no pip install needed)
# For SMS alerts:
twilio
Set Up API Keys (Crucial for full functionality):
Create a .streamlit folder in the root directory of your project (if it doesn't exist) and inside it, create a file named secrets.toml.

Populate secrets.toml with your API keys and credentials:

Ini, TOML

# CryptoCompare API Key (Required for News & Sentiment)
CRYPTOCOMPARE_API_KEY = "YOUR_CRYPTOCOMPARE_API_KEY"

# SMTP Email Configuration (Optional, for email alerts)
EMAIL_ADDRESS = "your_email@example.com"
EMAIL_PASSWORD = "your_email_app_password_or_regular_password" # Use app password if available
SMTP_SERVER = "smtp.yourprovider.com" # e.g., "smtp.gmail.com"
SMTP_PORT = 587 # or 465 for SSL

# Twilio SMS Configuration (Optional, for SMS alerts)
TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN = "your_auth_token"
TWILIO_PHONE_NUMBER = "+1234567890" # Your Twilio phone number
Important: Replace placeholder values with your actual keys and credentials.
Refer to the documentation for CoinGecko (no API key typically needed for public endpoints used here), CryptoCompare, and Twilio on how to obtain their API keys.
For email, ensure you use an "App Password" if your email provider (like Gmail) uses 2-Factor Authentication, as using your regular password might not work or is less secure.
Run the Streamlit App:

Bash

streamlit run crypto_monitor.py
(Assuming your main app file is crypto_monitor.py)

The app will open in your default web browser.

🌐 Deployment
This application is designed for deployment on Streamlit Cloud. Ensure your requirements.txt and .streamlit/secrets.toml files are correctly set up in your GitHub repository for successful deployment.

🚧 Current Limitations & Future Improvements
While feature-rich, the current version of Crypto Monitor has some limitations:

API Rate Limits: Relying on free-tier public APIs means occasional rate limit hits, leading to temporary "cooldown" periods. This can interrupt the real-time data flow for users.
Manual API Key Setup: Configuring API keys in secrets.toml can be a hurdle for less technical users.
Basic Social Media Sentiment: The "Social Media Sentiment Analysis" section is currently a placeholder, awaiting full integration with social media APIs (e.g., Twitter, Reddit) and more advanced NLP models.
Limited Portfolio History: The portfolio tracker currently only shows real-time value. Historical tracking, transaction logging, and detailed PnL (Profit & Loss) analysis are not yet implemented.
Future Enhancements planned include:

Robust API Gateway/Proxy: Implementing a custom backend to intelligently manage API calls, reduce rate limit issues, and provide a smoother user experience.
Advanced Social Media Sentiment: Full integration with social media APIs for richer, real-time sentiment analysis.
Persistent User Data: Implementing user authentication and a database to save portfolios, alerts, and preferences across sessions.
Expanded Alert Options: Adding more alert types (e.g., volume spikes, indicator crossovers) and additional notification channels.
Basic Predictive Analytics: Exploring the integration of simple machine learning models for short-term price forecasting (with disclaimers).
Enhanced UI/UX: Further refinement of the user interface for even greater intuitiveness and aesthetic appeal.
🤝 Contribution & Support
This project is developed by a team of 5 students as a final project.
For any questions, feedback, or support, please contact: support@cryptomonitor.com

Made with ❤️ by Hamza Tebri
