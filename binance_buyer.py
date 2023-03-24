from binance.client import Client
import os 
from dotenv import load_dotenv
import time


load_dotenv()

API_KEY =  os.getenv("API_KEY") 
API_SECRET =  os.getenv("API_SECRET") 

client = Client(API_KEY, API_SECRET )
limit = 28600
balance = client.get_asset_balance(asset='BTC')
print ( "Actual_price")
print("Available BTC balance:", balance['free'])
quantity = 0.005  # Order quantity in BTC


sell=True
while sell:
    # Put your code here
    print("Checking if price is in range")
    price = float(client.get_symbol_ticker(symbol='BTCUSDT')["price"])
    print(price)
    if price >= limit:
        client.order_market_sell(symbol="BTCUSDT", quantity=quantity)
        sell=False
    # Sleep for 3 minutes
    time.sleep(180)

