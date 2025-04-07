from binance.client import Client
import os 
from dotenv import load_dotenv
from datetime import datetime



load_dotenv()

API_KEY =  os.getenv("API_KEY") 
API_SECRET =  os.getenv("API_SECRET") 

client = Client(API_KEY, API_SECRET )


def get_timestamp(date_str):
        date_obj = datetime.strptime(date_str, "%d %b %Y")
        timestamp_int = int(date_obj.timestamp())
        return timestamp_int




  
def get_binance_data ( coin, start , end , interval=Client.KLINE_INTERVAL_1HOUR):

        from binance.client import Client
        client = Client(API_KEY, API_SECRET)

        data = client.get_historical_klines( symbol = coin , interval=interval ,start_str =start, end_str =end)
        
    
        return data

def get_tickers():
        client = Client(API_KEY, API_SECRET)

            
        return  client.get_all_tickers()
