from secrets import API_TOKEN, API_TOKEN_SANDBOX
from iexfinance.stocks import get_historical_data
import pandas as pd

## SANDBOX TESTING
import os
os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'

class PriceData:

    token = API_TOKEN_SANDBOX # replace with API_TOKEN to use real world data 

    def __init__(self, period, intv, depth) -> None:
        
        self.start = period[0]
        self.end = period[-1]
        self.intv = intv
        self.depth = depth

    def to_csv(self, ticker):

        start = self.start
        end = self.end

        priceDataframe = get_historical_data(ticker, start, end, close_only=True, token=PriceData.token)
        priceDataframe.to_csv(f"datasets/{ticker}_history_{start}_{end}.csv")

    def parse(self, ticker):
        start = self.start
        end = self.end
        filename = f"datasets/{ticker}_history_{start}_{end}.csv"
        
        df = pd.read_csv(filename)
        df = df.rename(columns=({'Unnamed: 0': 'dates'}))

        df_dates = df['dates'].values
        totalDays = len(df) 
        dateList = []
        i = 0
        intv = self.intv
        while(i < totalDays//intv):
            nextDate = df_dates[i*intv]
            dateList += [nextDate]
            i += 1

        df_dates_consecutive = [
            dateList[i:i+2] for i in range(len(dateList)-1)
        ]
        depth = self.depth
        if depth >= len(df_dates_consecutive):
            df_dates_sequence = df_dates_consecutive
        else:
            df_dates_sequence = [
                df_dates_consecutive[i:i + depth] for i in range(len(df_dates_consecutive) - depth + 1)
            ]
       
        featureCols = [
                f"feature_{i}" for i in range(self.depth - 1)
            ]
        features_df = pd.DataFrame(columns = featureCols)
        labelsCol = [
                'buy/sell'
            ]
        labels_df = pd.DataFrame(columns = labelsCol)

        for featureDateSeq in df_dates_sequence:
            feature = []
            l = len(featureDateSeq) - 1
            for i in range(l):
                start = featureDateSeq[i][0]
                startPrice = df[df['dates'] == start]['close'].values[0]

                end = featureDateSeq[i][-1]
                endPrice = df[df['dates'] == end]['close'].values[0]

                rate = endPrice/startPrice - 1
                feature += [rate]

            lastStartDate = featureDateSeq[-1][0]
            lastStartPrice = df[df['dates'] == lastStartDate]['close'].values[0]

            lastEndDate = featureDateSeq[-1][-1]
            lastEndPrice = df[df['dates'] == lastEndDate]['close'].values[0]

            rate = lastEndPrice/lastStartPrice - 1
            label = rate//abs(rate)

            features_df = features_df.append(pd.Series(
                feature, index = featureCols
            ), ignore_index = True)

            labels_df = labels_df.append(pd.Series(
                [label], index = labelsCol
            ), ignore_index = True)
        
        features_df.to_csv(f"datasets/{ticker}_training_{self.start}_{self.end}.csv", index=False)
        labels_df.to_csv(f"datasets/{ticker}_labels_{self.start}_{self.end}.csv", index=False)

    def weightings(self, ticker):

        filename = f"datasets/{ticker}_labels_{self.start}_{self.end}.csv"
        df = pd.read_csv(filename)

        buy = df[df['buy/sell'] == 1].count().values[0]
        sell = df[df['buy/sell'] == -1].count().values[0]

        buySignalRate = buy/(buy + sell)
        sellSignalRate = sell/(buy + sell)

        print(f"Between {self.start} and {self.end}, the security {ticker} with respect to {self.intv} day periods had:\nBuy signal rate {buySignalRate}\nSell signal rate {sellSignalRate}")

