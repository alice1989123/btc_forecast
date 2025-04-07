label_width = 12
input_width= 200 #label_width*4
input_shape = (input_width, 1)

variables_used = ['close' ,] # 'open', 'high', 'low', 'volume', 'quote_asset_volume','num_trades','taker_base_vol','taker_quote_vol' ]
coins = ['BTCUSDT' ,   'XTZUSDT', 'XRPUSDT', 'XLMUSDT', 'VETUSDT', 'UNIUSDT', 'TRXUSDT', 'THETAUSDT', 'NEOUSDT', 'MKRUSDT', 'LTCUSDT', 'LINKUSDT', 'FILUSDT', 'ETHUSDT', 'ETCUSDT', 'EOSUSDT', 'DOTUSDT', 'DOGEUSDT', 'BCHUSDT', 'ATOMUSDT', 'ADAUSDT', 'AAVEUSDT']

LOG_DIR = "logs"
LOG_FILE_NAME = "my_application.log"