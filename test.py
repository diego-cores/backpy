
from time import time
import backpy

class FristStrategy(backpy.StrategyClass):
    def next(self) -> None:
        tendecia = self.idc_ema(400,1).iloc[0]

        if self.close > tendecia:
            if self.close < self.idc_ema(100,1).iloc[0] and self.prev("Close", 5).iloc[0:3].std() < self.prev("Close",3).std(): 
                self.act_open('l', self.close-self.close*0.3, self.close+self.close*1, 100)
        elif self.close < tendecia:
            if self.close > self.idc_ema(100,1).iloc[0] and self.prev("Close", 5).iloc[0:3].std() < self.prev("Close",3).std():
                self.act_open('s', self.close+self.close*0.1, self.close-self.close*1, 100)

start = time()
backpy.load_yfinance_data(tickers="BTC-USD", start="2010-02-01", end="2024-03-01", interval="1d", statistics=False, progress=True)

#instance_indicators = backpy.StrategyClass()

backpy.run(FristStrategy,False)
backpy.stats_trades(True)
backpy.plot(True)

print(time()-start)
