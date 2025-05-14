import os
import time
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert
from binance.client import Client

# Load secrets
load_dotenv("keys/.env")

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")
client = Client(API_KEY, API_SECRET)

# DB config
DB_USER = os.getenv("DBUSER")
DB_PASSWORD = os.getenv("DBPASSWORD")
DB_HOST = os.getenv("DBHOST")
DB_PORT = os.getenv("DBPORT")
DB_NAME = os.getenv("DBNAME")

# Constants
TABLE_NAME = "binance_klines"
SLEEP_SECONDS = 1


def get_db_engine():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)


def get_latest_open_time(symbol: str, timeframe: str, engine) -> datetime | None:
    query = text(f"""
        SELECT MAX(open_time)
        FROM {TABLE_NAME}
        WHERE symbol = :symbol AND timeframe = :timeframe
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"symbol": symbol, "timeframe": timeframe})
        return result.scalar()


def get_all_binance_klines(symbol: str, start_str: str, end_str: str, interval: str):
    start_ts = int(datetime.strptime(start_str, "%d %b %Y").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%d %b %Y").timestamp() * 1000)
    limit = 1000
    klines = []

    while start_ts < end_ts:
        batch = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            endTime=end_ts,
            limit=limit
        )
        if not batch:
            break
        klines.extend(batch)
        start_ts = batch[-1][6] + 1
        time.sleep(SLEEP_SECONDS)

    return klines


def clean_klines(raw_klines, symbol: str, timeframe: str) -> pd.DataFrame:
    df = pd.DataFrame(raw_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float,
        "quote_asset_volume": float,
        "taker_buy_base_volume": float,
        "taker_buy_quote_volume": float,
        "number_of_trades": int
    })
    return df[[
        "symbol", "timeframe", "open_time", "close_time", "open", "high", "low", "close",
        "volume", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume"
    ]]


def insert_to_postgres(df: pd.DataFrame, engine):
    metadata = MetaData(bind=engine)
    metadata.reflect()
    table = metadata.tables[TABLE_NAME]

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        records = df.to_dict(orient="records")
        stmt = pg_insert(table).values(records)
        stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "timeframe", "open_time"])
        session.execute(stmt)
        session.commit()
        print(f"✅ Inserted {len(records)} rows.")
    except Exception as e:
        session.rollback()
        print(f"❌ Insert failed: {e}")
    finally:
        session.close()


def run_etl(symbol: str, timeframe: str, end: str = "1 Jan 2026"):
    engine = get_db_engine()
    latest_time = get_latest_open_time(symbol, timeframe, engine)
    start_str = latest_time.strftime("%d %b %Y") if latest_time else "1 Jan 2017"

    print(f"🚀 Fetching {symbol} klines ({timeframe}) from {start_str} to {end}")
    raw = get_all_binance_klines(symbol, start_str, end, interval=timeframe)
    if not raw:
        print("⚠️ No new data fetched.")
        return
    df = clean_klines(raw, symbol, timeframe)
    insert_to_postgres(df, engine)


if __name__ == "__main__":
    # Run ETL for 1h klines
    run_etl("BTCUSDT", "1h")
