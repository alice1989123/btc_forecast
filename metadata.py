import json
import config.config as config
import glob

coins = config.coins
# Read and load the JSON file
def read_metadata(coin:str):
    # Change the path to the directory where the text files are stored
    path = f"./models/*{coin}*.txt"
    print ( [file for file in glob.glob(path)])
    # Loop through all the files with the .txtx extension
    for file_path in glob.glob(path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    # Now you can access the data as a Python dictionary
        #print(data)
#read_meatadata("BTCUSDT")