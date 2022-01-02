import time
import hmac
import requests
import json
import time
SECONDS_IN_HOUR = 60*60
SECONDS_IN_DAY = SECONDS_IN_HOUR*24
SECONDS_IN_MONTH = SECONDS_IN_DAY*30
SECONDS_IN_YEAR = SECONDS_IN_DAY*365

ENDPOINT="https://ftx.us/api"

"""
Params:
- market_name (eg. 'BTC/USD')
- window_length: length in seconds of period for data to be returned between start_time and end_time
- start_time: epoch time
- end_time: epoch time
"""
def get_market_historical(market_name, window_length, start_time, end_time):
    addendum = "/markets/{market_name}/candles?resolution={resolution}&start_time={start_time}&end_time={end_time}"
    params = {'market_name': market_name, 'resolution': window_length, 'start_time': start_time, 'end_time': end_time}
    addendum = addendum.format(**params)

    url = ENDPOINT + addendum
    response = requests.get(url)
    if response.status_code == 200:
        dic = json.loads(response.text)
        return dic['result']
    else:
        return None


def add_info(datum):
    datum['range'] = datum['high'] - datum['low']
    datum['change'] = (datum['close'] - datum['open']) / datum['open']
    return datum

# start_time = int(time.time() - 3*SECONDS_IN_YEAR)
# end_time = int(time.time())
#
# results = get_market_historical('BTC/USD', SECONDS_IN_DAY, start_time, end_time)
# results = [add_info(datum) for datum in results]
#
#
# results[0]
