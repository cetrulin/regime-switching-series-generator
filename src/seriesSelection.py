import os
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

config = {
    'path': 'C:\\Users\\suare\\data\\raw\\quantquote\\order_530886\\',
    'years_to_explore': ['2015', '2016', '2017', '2018', '2019', '2020'],
    'symbol': 'ews',  # 'spy',
    'prefix': 'table_',
    'extension': '.csv',
    'output_path': 'C:\\Users\\suare\\data\\tmp\\',
    'output_selection_path': 'C:\\Users\\suare\\data\\tmp\\selection',
    'desired_length': 1000,
    'desired_abs_mean_tresh': 0.00000000001
    }

out_path = os.sep.join([config['output_path'],
                        config['symbol'] + ('_'.join(config['years_to_explore'])) + config['extension']])

# Read tmp file if it exists
if os.path.exists(out_path):
    print('Loading existing file...')
    concated_df = pd.read_csv(out_path, index_col=0)
else:
    dfs = []
    for file in os.listdir(config['path']):
        if file[:4] in config['years_to_explore']:
            print(f'Reading {file}')
            path = os.sep.join([config['path'], file])
            df = pd.read_csv(os.sep.join([path, config['prefix'] + config['symbol'] + config['extension']]),
                             names=['date', 'time', 'open', 'high', 'low', 'close', 'volume',
                                    'suspicious', 'Dividends', 'Extrapolation'])
            # print(df.head())
            dfs.append(df)

    print(f'Concatenating subsets...')
    # Full DF
    concated_df = pd.concat(dfs)
    dfs = None
    concated_df.drop(columns=['suspicious', 'Dividends', 'Extrapolation'], axis=1, inplace=True)
    concated_df['datetime'] = concated_df.date.astype(str) + 'T' + concated_df.time.astype(str)
    concated_df.set_index(['datetime'], drop=True, inplace=True)
    concated_df.sort_index(ascending=True, inplace=True)
    concated_df.sort_values(['date', 'time'], ascending=True)
    concated_df.drop_duplicates(['date', 'time'], inplace=True)

    print(f'Saving subsets into a concatenated file ...')
    Path(config['output_path']).mkdir(parents=True, exist_ok=True)
    concated_df.to_csv(out_path)


# Parse dataframe

# Generate close price returns and moving average to select the period with a mean close to 0
concated_df['close_returns'] = concated_df['close'] / concated_df['close'].shift(1) - 1
sma_col = f"SMA_{config['desired_length']}"
concated_df[sma_col] = \
    concated_df['close_returns'].rolling(window=config['desired_length']).mean()
concated_df['SMA_start_date'] = concated_df['date'].shift(config['desired_length'])
concated_df['SMA_start_ms'] = concated_df['time'].shift(config['desired_length'])

# Drop first rows as the moving average is NaN
len_with_nans = len(concated_df)
concated_df.dropna(inplace=True)
concated_df['SMA_start_date'] = concated_df['SMA_start_date'].astype(int)
concated_df['SMA_start_ms'] = concated_df['SMA_start_ms'].astype(int)
concated_df['datetime'] = concated_df.index.astype(str)
assert (len_with_nans - len(concated_df)) == config['desired_length'], 'There are non expected NaNs'

# Filter by desired mean
selected_df = concated_df[concated_df[sma_col].abs() <= config['desired_abs_mean_tresh']]

# Export selection
print(f"{len(selected_df)} sets selected.\nExporting selection...")

for index, row in selected_df.iterrows():  # row['SMA_start_ms']) (not needed by now)
    print(f"Current selection from {row['SMA_start_date']} to { row['date']} with mean {sma_col}")
    Path(config['output_selection_path']).mkdir(parents=True, exist_ok=True)
    export_path = os.sep.join([config['output_selection_path'], config['symbol'] + row['datetime']])
    current_selection_df = concated_df[(concated_df['SMA_start_date'].between(row['SMA_start_date'], row['date']))]
    concated_df.sort_index(ascending=True, inplace=True)
    current_selection_df.to_csv(export_path + config['extension'])

    # Plot set
    current_selection_df.close.plot()
    plt.xticks(rotation=45)
    plt.savefig(export_path+'_close_price.png')
    plt.show()
    current_selection_df.close_returns.plot()
    plt.xticks(rotation=45)
    plt.savefig(export_path+'_returns.png')
    plt.show()
    current_selection_df.close_returns.hist(bins=100, alpha=0.5)
    plt.xticks(rotation=45)
    plt.savefig(export_path+'_returns_histogram.png')
    plt.show()
