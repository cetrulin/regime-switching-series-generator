import os
import pandas as pd

config = {
    'path': 'C:\\Users\\suare\\data\\raw\\quantquote\\order_530886\\',
    'years_to_explore': ['2015', '2016', '2017'],
    'symbol': 'ews',
    'prefix': 'table_',
    'extension': '.csv',
    'output_path': 'C:\\Users\\suare\\data\\tmp\\'
    }

out_path = os.sep.join([config['output_path'],
                        config['symbol'] + ('_'.join(config['years_to_explore'])) + config['extension']])

# Read tmp file if it exists
if os.path.exists(out_path):
    df = pd.read_csv(out_path, names=['date', 'milliseconds', 'open', 'high', 'low', 'close', 'volume'])
else:
    dfs = []
    for file in os.listdir(config['path']):
        if file[:4] in config['years_to_explore']:
            print(f'Reading {file}')
            path = os.sep.join([config['path'], file])
            df = pd.read_csv(os.sep.join([path, config['prefix'] + config['symbol'] + config['extension']]),
                             names=['date', 'milliseconds', 'open', 'high', 'low', 'close', 'volume',
                                    'suspicious', 'Dividends', 'Extrapolation'])
            # print(df.head())
            dfs.append(df)

    print(f'Concatenating subsets...')
    # Full DF
    concated_df = pd.concat(dfs)
    dfs = None
    concated_df.drop(columns=['suspicious', 'Dividends', 'Extrapolation'], axis=1, inplace=True)
    concated_df.sort_values(['date', 'milliseconds'], ascending=True)
    concated_df.drop_duplicates(['date', 'milliseconds'], inplace=True)

    print(f'Saving subsets into a concatenated file ...')
    concated_df.to_csv(out_path)

# TODO: do something with dataframe
print(df.head())


