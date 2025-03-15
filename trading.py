from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, OrderRequest
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API keys
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

trading_client = TradingClient(API_KEY, SECRET_KEY)

# Get our account information.
account = trading_client.get_account()

# Check if our account is restricted from trading.
if account.trading_blocked:
    print('Account is currently restricted from trading.')

# Check how much money we can use to open new positions.
print('${} is available as buying power.'.format(account.buying_power))


# Create a market order to buy 1 share of a stock
symbol = "AAPL"  # Replace with the desired stock symbol
buy_order = trading_client.submit_order(
    OrderRequest(
        symbol=symbol,
        qty=1,  # Number of shares to buy
        side='buy',
        type='market',  # Market order
        time_in_force='gtc'  # Good 'til canceled
    )
)

print(f"Buy order for 1 share of {symbol} has been submitted.")