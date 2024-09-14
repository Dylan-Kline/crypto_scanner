from data.DataPreprocessing.Extract_Crypto_Data import collect_crypto_exchange_data

def main():
    trading_sym = 'BTC-USD'
    data = collect_crypto_exchange_data(trading_sym)
    print(data)
    
main()


