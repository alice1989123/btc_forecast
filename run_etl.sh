#!/bin/bash
cd /home/alice/btc_forecast
/home/alice/btc_forecast/env/bin/python3 etl.py >> etl.log 2>&1
