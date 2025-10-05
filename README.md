![BackPy logo](images/logo.png)
![Version](https://img.shields.io/badge/version-0.9.72b4-blue) ![Status](https://img.shields.io/badge/status-beta-orange)

# BackPy

**BackPy** is a Python library for backtesting strategies in financial markets.
You can provide your own historical data or use the built-in integration with the `yfinance` or `binance-connector` modules.

With **BackPy-binance-connector** you can connect your strategy to the real market using Binance.
Official repository: [BackPy-binance-connector](https://github.com/diego-cores/BackPy-binance-connector "BackPy-binance-connector").

---

## ❓ Why BackPy?

BackPy integrates in one place:

- **Backtesting**
- **Data loading** from multiple sources (yfinance, Binance, your own files)
- **Interactive charts**
- **Advanced statistics**

In tests performed locally on a personal computer (PC with AMD Ryzen 7 5800X, 32 GB of RAM, and Python 3.12), BackPy showed exceptional performance even with large volumes of data.

Estimated time comparison for processing 50,000 candles:

| Method           | Estimated time         |
| ---------------- | ---------------------- |
| Manual           | ~7.4 hours             |
| **BackPy v0.9.72** | **~-.- seconds** |
| Another module   | ~-.- seconds           |

📌 These times are illustrative comparisons, performed locally. Results may vary depending on hardware, configuration, and data volume.

💡 **Conclusion:** BackPy not only centralizes your workflow, but also accelerates your development and analysis.

---

## ⚠️ Important Notices

Please make sure to read the following before using this software:

- [Risk Notice](Risk_notice.txt)
- [License](LICENSE)

By using this software, you acknowledge that you have read and agree to the terms outlined in these documents.

---

## 📦 How to install backpy with pip

1. Download the latest version from GitHub

- Go to this project GitHub page.
- Download the ZIP file of the latest version of the project.

2. Unzip the ZIP file

- Unzip the ZIP file you downloaded.
- This will give you a folder containing the project files.

3. Open the terminal

- Open the terminal in your operating system.
- Navigate to the folder you just unzipped. You can use the cd command to change directories.

4. Install the module

- Once you are in the project folder in terminal, run the following command: `pip install .`.
- This command will install the Python module using the setup.py file located in the project folder.

5. Verify installation

- After the installation process finishes without errors, you can verify if the module has been installed correctly by running some code that imports the newly installed module.

6. Clean downloaded files

- After you have verified that the module is working correctly, you can delete the downloaded ZIP file and unzipped folder if you wish.

---

## 🚀 Code example

BackPy allows you to design strategies quickly and easily:

```python

import backpyf

backpyf.load_binance_data_spot(
    symbol='BTCUSDT',
    start_time='2023-01-01',
    end_time='2024-01-01',
    interval='1h'
)

class macdStrategy(backpyf.StrategyClass):
    def next(self):
        if len(self.date) < 30 or len(self.prev_positions()) > 0:
            return

        macd = self.idc_macd()[-1]
        sma = self.idc_sma(42)[-1]

        if (
            self.close[-1] > sma
            and macd['histogram'] > 0
        ):
            self.act_taker(True, amount=self.get_init_funds())

            self.ord_put('takeProfit', self.close[-1]*1.06)
            self.ord_put('stopLoss', self.close[-1]*0.98)

backpyf.run_config(
    initial_funds=10000,
    commission=(0.04, 0.08),
    spread=0.01,
    slippage=0.01
)
backpyf.run(macdStrategy)

backpyf.plot_strategy(style='darkmode', block=False)
backpyf.plot(log=True)
```

Don't forget to view your results:

![statistics graph image](images/graph.png "BackPy graph")
