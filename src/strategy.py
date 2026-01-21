import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import warnings

warnings.filterwarnings('ignore')

class SeasonalETF:
    def __init__(self, start_date='2015-01-01', end_date=None, benchmark_ticker='QQQ'):
        self.backtest_start = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else dt.datetime.now()
        # 10 Year backtesting seasonality
        self.data_start = self.backtest_start - pd.DateOffset(years=10)
        
        self.benchmark_ticker = benchmark_ticker
        self.tickers = []
        self.data = pd.DataFrame()
        self.persistence_scores = pd.DataFrame()
        
    def get_universe(self):
        """
        Fetches NASDAQ 100 universe.
        UPDATED: Added headers to prevent HTTP 403 Forbidden error.
        """
        print(f"--- Fetching NASDAQ 100 Universe ---")
        try:
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            
            # THE FIX: spoof a browser so Wikipedia doesn't block us
            storage_options = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            
            tables = pd.read_html(url, storage_options=storage_options)
            
            # Robust check: Find the table that actually has the tickers
            df = None
            for table in tables:
                if 'Ticker' in table.columns:
                    df = table
                    break
            
            if df is None:
                raise ValueError("Could not find table with 'Ticker' column on Wikipedia.")
                
            self.tickers = df['Ticker'].tolist()
            
            # Clean tickers (e.g. BRK.B -> BRK-B)
            self.tickers = [ticker.replace('.', '-') for ticker in self.tickers]
            
            # Add Benchmark if missing
            if self.benchmark_ticker not in self.tickers:
                self.tickers.append(self.benchmark_ticker)
                
            print(f"Successfully loaded {len(self.tickers)} tickers.")
            
        except Exception as e:
            print(f"Error fetching universe: {e}. Using fallback.")
            # Fallback list for testing if wifi/site is down
            self.tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'QQQ']

    def ingest_data(self):
        """Downloads historical adjusted close prices."""
        print(f"\n--- Ingesting Data (Start: {self.data_start.date()}) ---")
        
        raw_data = yf.download(
            self.tickers, 
            start=self.data_start, 
            end=self.end_date, 
            interval='1d',
            group_by='ticker',
            auto_adjust=True,
            threads=True,
            progress=False
        )
        
        df_close = pd.DataFrame()
        for ticker in self.tickers:
            if ticker in raw_data.columns.levels[0]:
                df_close[ticker] = raw_data[ticker]['Close']
                
        df_close.dropna(axis=1, how='all', inplace=True)
        self.data = df_close.ffill()
        
        # Separate benchmark from tradable universe
        if self.benchmark_ticker in self.data.columns:
            self.benchmark_data = self.data[[self.benchmark_ticker]].copy()
            self.data = self.data.drop(columns=[self.benchmark_ticker])
            
        print(f"Data Loaded. Tradable Universe: {len(self.data.columns)} tickers.")

    def calculate_seasonality(self):
        """Calculates the Persistence Score (Win Rate * Median Return) for each month."""
        print("\n--- Calculating Seasonal Metrics ---")
        
        # Resample Monthly Returns
        monthly_prices = self.data.resample('BM').last()
        monthly_returns = monthly_prices.pct_change()
        
        self.persistence_scores = pd.DataFrame(index=monthly_returns.index, columns=monthly_returns.columns)
        
        # Start loop after first 12 months
        for date in monthly_returns.index[12:]:
            current_month = date.month
            
            # Historical Window Logic
            historical_data = monthly_returns.loc[:date - pd.DateOffset(days=1)]
            seasonal_history = historical_data[historical_data.index.month == current_month]
            seasonal_history = seasonal_history.tail(10) # Last 10 years
            
            if len(seasonal_history) < 3:
                continue
            
            win_rate = (seasonal_history > 0).sum() / len(seasonal_history)
            median_return = seasonal_history.median()
            score = win_rate * median_return
            
            self.persistence_scores.loc[date] = score

        self.persistence_scores = self.persistence_scores.fillna(-np.inf)
        print("Seasonal Rankings Calculated.")


    def run_backtest(self, initial_capital=10000):

        """
        Simulates trading the Top 10 'Seasonal' stocks every month.
        """
        print("\n--- Running Backtest Simulation ---")
        
        # 1. Align Data
        monthly_prices = self.data.resample('BM').last()
        monthly_returns = monthly_prices.pct_change()
        
        # 2. Create the Portfolio container
        self.portfolio_history = pd.DataFrame(index=monthly_returns.index)
        self.portfolio_history['Total Value'] = initial_capital
        self.portfolio_history['Monthly Return'] = 0.0
        
        # 3. The Strategy Loop
        # start at index 12 to match seasonalikty calculations
        current_capital = initial_capital
        
        # Iterate through
        for i in range(12, len(monthly_returns) - 1):
            date = monthly_returns.index[i]
            next_date = monthly_returns.index[i+1]
            
            # if not at start date then ignore
            if date < self.backtest_start:
                continue
            
            # Step A: Get the Rankings for THIS month
            if date not in self.persistence_scores.index:
                continue
                
            scores = self.persistence_scores.loc[date]
            
            # Step B: Select Top 10 Stocks
            top_10_tickers = scores.nlargest(10).index.tolist()
            
            # Step C: Calculate Return for the NEXT month
            future_returns = monthly_returns.loc[next_date, top_10_tickers]
            portfolio_return = future_returns.mean()
            
            if np.isnan(portfolio_return):
                portfolio_return = 0.0
            
            # Step D: Update Portfolio Value
            current_capital = current_capital * (1 + portfolio_return)
            
            self.portfolio_history.loc[next_date, 'Total Value'] = current_capital
            self.portfolio_history.loc[next_date, 'Monthly Return'] = portfolio_return
            
        
        # Performance Summary
        total_return = (current_capital - initial_capital) / initial_capital
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital:   ${current_capital:,.2f}")
        print(f"Total Return:    {total_return*100:.2f}%")
        
        return self.portfolio_history
    
    def plot_results(self):
        """
        Plots the Strategy vs. Benchmark (QQQ) starting from the Backtest Date.
        """
        import matplotlib.pyplot as plt
        
        print("\n--- Generating Performance Plot ---")
        
        # 1. Get Benchmark Data (QQQ)
        benchmark_prices = self.benchmark_data[self.benchmark_ticker].resample('BM').last()
        benchmark_returns = benchmark_prices.pct_change()
        
        # 2. Slice Data to Backtest Start Date (2010)
        # plot from 2020 onwards
        strategy_data = self.portfolio_history.copy()
        strategy_data = strategy_data[strategy_data.index >= self.backtest_start]
        
        # Align benchmark to this new sliced timeframe
        aligned_benchmark = benchmark_returns.loc[strategy_data.index]
        
        # 3. Calculate Cumulative Returns (Growth of $1)
        # Strategy
        strategy_cumulative = strategy_data['Total Value'] / strategy_data['Total Value'].iloc[0]
        
        # Benchmark (Start at 1.0 and compound)
        benchmark_cumulative = (1 + aligned_benchmark).cumprod()
        benchmark_cumulative = benchmark_cumulative / benchmark_cumulative.iloc[0] 
        
        # 4. Plotting
        plt.figure(figsize=(12, 6))
        
        # Plot Strategy
        plt.plot(strategy_cumulative.index, strategy_cumulative, label='Seasonal Momentum', color='#00ff00', linewidth=2)
        
        # Plot Benchmark
        plt.plot(benchmark_cumulative.index, benchmark_cumulative, label=f'Benchmark ({self.benchmark_ticker})', color='gray', linestyle='--', alpha=0.7)
        
        plt.title(f'Seasonal Strategy vs {self.benchmark_ticker}: Cumulative Return', fontsize=14)
        plt.ylabel('Growth of $1 (Normalized)', fontsize=12)
        plt.xlabel('Year', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log') 
        
        plt.savefig('backtest_results.png')
        print("Plot saved as 'backtest_results.png'. Check it now!")