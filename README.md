# crypto_scanner

# SETUP

Clone the repo
git clone https://github.com/Dylan-Kline/crypto_scanner.git

Navigate to the project directory
cd <project_directory_name>

Before installing or running the program you will need to install ta-lib (technical indicators library):

    For windows: https://medium.com/pythons-gurus/how-to-properly-install-ta-lib-on-windows-11-for-python-a-step-by-step-guide-13ebb684f4a6

    For Mac: https://pypi.org/project/TA-Lib/
        - you will need to have homebrew installed on your mac first

Next run the following command to get the necessary libraries for the project:
    pip install -r requirements.txt

Running the parts of the program:
    Data:
        - first set fetch to True and process to True in <main.py> and then run <main.py>
        - you should now have .csv files inside of the data folder

    Backtests:
        - for backtests look inside of <backtest_test.py> for an example of how to use the backtest module.

    Running real-time prediction loop:
        <testing_predict.py> shows an example of how to run it.



