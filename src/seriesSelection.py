import os
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

config = {
    'path': 'C:\\Users\\suare\\data\\raw\\quantquote\\order_530886\\',
    # 'years_to_explore': ['2015', '2016', '2017', '2018', '2019', '2020'],
    # ############### For Tests
    # 'category': {unespecified: ['ewp']}  # 'gld', 'spy','xle', 'emb','dia', 'qqq', 'ewp'
    # 'years_to_explore': ['2007', '2008', '2009', '2010', '2011', '2012', '2013',
    #                      '2014', '2015', '2016', '2017', '2018', '2019', '2020'],
    # ############### For Synthetic_2
    # 'years_to_explore': ['2017', '2018', '2019', '2020'],
    # 'category': {unespecified:
    #              ['aaxj','acwi','agg','agq','bal','biv','bkf','bnd','brf','bsv','bwx','dba','dbb','dbc','dbe','dbo',
    #             'ddm','dem','dgaz','dgp','dgs','dia','dig','djp','dog','drn','drv','dto','dug','dust','dvy','dxd',
    #             'dzz','ech','edc','edz','eem','eev','efa','emb','epi','epp','epu','erx','ery','eum','euo','ewa','ewc',
    #             'ewd','ewg','ewh','ewj','ewl','ewm','ewp','ews','ewt','ewu','eww','ewy','ewz','eza','fas','faz','fcg',
    #             'fdn','fxc','fxd','fxe','fxf','fxi','fxo','fxp','fxz','gcc','gdx','gld','gll','gltr','gsg','gsp','hyg',
    #             'iau','ibb','icf','idx','ief','ieo','iev','iez','ige','ijh','ijj','ijk','ijr','ijs','ijt','ilf','itb',
    #             'ive','ivv','ivw','iwb','iwc','iwd','iwf','iwm','iwn','iwo','iwp','iwr','iws','iwv','ixc','iye','iyf',
    #             'iyj','iym','iyr','iyt','iyw','iyz','jjc','jnk','kbe','kie','kol','kre','lqd','mdy','midu','moo','mub',
    #             'mvv','mzz','nlr','nugt','oef','oih','oil','pall','pbw','pcy','pff','pgf','pgx','pho','pie','pin',
    #             'pph','pplt','psp','psq','qid','qld','qqq','qtec','rja','rji','rsp','rsx','rth','rwm','rwr','rwx',
    #             'sco','scz','sdow','sds','sdy','sh','shm','shy','skf','slv','smh','smn','spxl','spxs','spxu','spy',
    #             'sqqq','srs','sso','svxy','tan','tbt','tecl','tecs','tip','tlt','tmf','tmv','tna','tqqq','tur','twm',
    #             'tza','uco','udow','uga','ugaz','ugl','ung','upro','ure','usd','usl','uso','uup','uvxy','uwm','uyg',
    #             'uym','vb','vbk','vbr','vde','vea','veu','vfh','vgk','vgt','vig','vis','vnm','vnq','vo','vot','vti',
    #             'vtv','vug','vv','vwo','vxx','vxz','vym','xes','xhb','xlb','xle','xlf','xli','xlk','xlp','xlu','xlv',
    #             'xly','xme','xop','xrt','ycs','zsl']},
    # 'desired_abs_mean_tresh': 0.00000000005,
    # 'output_path': 'C:\\Users\\suare\\data\\tmp\\',
    # 'output_selection_path': 'C:\\Users\\suare\\data\\tmp\\selection',
    # ############### For Synthetic_3
    'years_to_explore': ['2020'],
    'categories': {'1_equities': ['spy', 'ivv', 'vti', 'voo', 'qqq'],
                   '2_fixed_bonds': ['agg', 'bnd', 'lqd', 'vcit', 'bndx', 'vcsh'],
                   '2_fixed_pref': ['pff', 'pgx'],
                   '3_inter_bonds': ['igov', 'bwx', 'hyem', 'bwz', 'ibnd', 'cemb',
                                     # second batch
                                     'femb', 'picb', 'ihy', 'ishg', 'hyxu', 'cbon', 'emcb', 'sovb', 'kccb',
                                     # https://etfdb.com/themes/broad-international-bond-etfs/
                                     'bndx', 'iagg', 'igov', 'bwx', 'wip', 'bwz', 'ibnd', 'picb', 'ishg', 'hyxu', 'sovb'],
                   '4_energy': ['xle', 'amlp', 'vde', 'icln', 'tan', 'xop'],
                   '4_commodities': ['pdbc', 'dbc', 'gsg', 'djp', 'bci', 'comt', 'ftgc'],
                   '4_metals':  ['gld', 'iau', 'slv'],
                   '4_oilgas': ['uso', 'dbo', 'ung', 'bno'],
                   '5_alternatives': ['qai', 'mna', 'ftls', 'clix'],
                   '5_real_state': ['vnq', 'vnqi', 'iyr', 'schh', 'reet', 'xlre']},
    'prefix': 'table_',
    'extension': '.csv',
    'output_path': 'C:\\Users\\suare\\data\\tmp3\\',
    'output_selection_path': 'C:\\Users\\suare\\data\\tmp3\\selection',
    'desired_length': 1000,
    'desired_abs_mean_tresh': 0.01,
    'allowed_outliers_pct': 0.01,
    'specific_period': True,
    'period': '202006'   # , '202006']
    }


def read_file(filename: str) -> pd.DataFrame:
    path = os.sep.join([config['path'], filename])
    filepath = os.sep.join([path, config['prefix'] + config['symbol'] + config['extension']])
    if os.path.isfile(filepath):
        print(f'Reading {filename}')
        df = pd.read_csv(filepath, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume',
                                          'suspicious', 'Dividends', 'Extrapolation'])
        # print(df.head())
        return df


# for symbol in config['symbols']:
for category in config['categories'].keys():
    print(f'--\nSTART WITH CATEGORY: {category}')
    for symbol in config['categories'][category]:
        # print(f'--\nSTART WITH SYMBOL: {symbol}')
        config['symbol'] = symbol
        if config['specific_period']:
            out_path = os.sep.join([config['output_path'],
                                    category + '_' + config['symbol'] + config['period'] + config['extension']])
        else:
            out_path = os.sep.join([config['output_path'],
                                    category + '_' + config['symbol'] +
                                    ('_'.join(config['years_to_explore'])) + config['extension']])

        # Read tmp file if it exists
        if os.path.exists(out_path):
            print('Loading existing file...')
            concated_df = pd.read_csv(out_path, index_col=0)
        else:
            dfs = []
            for file in os.listdir(config['path']):
                if config['specific_period']:
                    if file[:6] in config['period']:
                        file_df = read_file(file)
                        if file_df is not None:
                            dfs.append(file_df)
                else:
                    if file[:4] in config['years_to_explore']:
                        file_df = read_file(file)
                        if len(file_df) > 0:
                            dfs.append(file_df)

            print(f'Concatenating subsets...')
            # Full DF
            if len(dfs) > 0:
                # print(dfs)
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
            else:
                continue

        # TODO? level 30min/hourly/day to reduce noise (maybe daily??)
        # if config['specific_periods']:
            # for freq in ['1min', '5min', '15min', '30min']: # TODO if necessary...

        # Parse dataframe
        if config['specific_period']:
            concated_df = concated_df.head(config['desired_length'] + 1)

        # Generate close price returns and moving average to select the period with a mean close to 0
        log_ret = False  # otherwise percentual
        if log_ret:
            concated_df['close_returns'] = np.log(concated_df['close'] / concated_df['close'].shift(1))
            # concated_df['close_returns'] = np.log(1 + concated_df['close'].pct_change())
        else:
            concated_df['close_returns'] = concated_df['close'] / concated_df['close'].shift(1) - 1
            # concated_df['close_returns'] = concated_df['close'].pct_change(1)  # same result
        concated_df.dropna(inplace=True)

        sma_col = f"SMA_{config['desired_length']}"
        concated_df[sma_col] = \
            concated_df['close_returns'].rolling(window=config['desired_length']).mean()
        concated_df['SMA_start_date'] = concated_df['date'].shift(config['desired_length'])
        concated_df['SMA_start_ms'] = concated_df['time'].shift(config['desired_length'])

        # Drop first rows as the moving average is NaN
        len_with_nans = len(concated_df)
        mean_return = concated_df['close_returns'].mean()
        concated_df['datetime'] = concated_df.index.astype(str)

        if not config['specific_period']:
            concated_df.dropna(inplace=True)
            concated_df['SMA_start_date'] = concated_df['SMA_start_date'].astype(int)
            concated_df['SMA_start_ms'] = concated_df['SMA_start_ms'].astype(int)
            assert (len_with_nans - len(concated_df)) == config['desired_length'], 'There are non expected NaNs'
            mean_return = False
            # Filter by desired mean
            selected_df = concated_df[concated_df[sma_col].abs() <= config['desired_abs_mean_tresh']]
        else:
            selected_df = pd.DataFrame() if np.abs(mean_return) > config['desired_abs_mean_tresh'] else concated_df

        # Export selection
        print(f"{len(selected_df)} sets were selected.")
        for index, row in selected_df.iterrows() if not config['specific_period'] \
                else selected_df.tail(1).iterrows():  # row['SMA_start_ms']) (not needed by now)
            print(f"Current selection from {row['SMA_start_date']} to { row['date']} with mean {sma_col}")
            Path(config['output_selection_path']).mkdir(parents=True, exist_ok=True)
            export_path = os.sep.join([config['output_selection_path'],
                                       category + '_' + config['symbol'] + row['datetime']])
            if not config['specific_period']:
                current_selection_df = concated_df[(concated_df['SMA_start_date'].between(row['SMA_start_date'],
                                                                                          row['date']))]
            else:
                current_selection_df = selected_df
            current_selection_df.sort_index(ascending=True, inplace=True)
            count = len(current_selection_df[current_selection_df['close_returns'] >
                                             (current_selection_df['close_returns'].std() * 3)])

            # These two are only relevant if we use specific periods.
            assert config['specific_period'], \
                'Comment out the next two lines of code and remove references in export names'
            trend = 'down' if mean_return < 0 else 'up'

            if (count/len(current_selection_df)) < config['allowed_outliers_pct']:
                # if the amount of outliers is less than 0.5% of the dataset we go ahead.
                print('Exporting selection...')
                current_selection_df.to_csv(export_path + config['extension'])

                # Plot set
                current_selection_df.close.plot()
                plt.xticks(rotation=45)
                plt.savefig(export_path+f'_close_price_{trend}_{mean_return}.png')
                plt.show()
                current_selection_df.close_returns.plot()
                plt.xticks(rotation=45)
                plt.savefig(export_path+f'_returns_{trend}_{mean_return}.png')
                plt.show()
                current_selection_df.close_returns.hist(bins=100, alpha=0.5)
                plt.xticks(rotation=45)
                plt.savefig(export_path+f'_returns_histogram_{trend}_{mean_return}.png')
                plt.show()
                plt.boxplot(current_selection_df.close_returns, labels=["Close price returns"])
                plt.xticks(rotation=45)
                plt.savefig(export_path+f'_boxplot_returns_{trend}_{mean_return}.png')
                plt.show()
            else:
                # print(len(current_selection_df))
                # print(current_selection_df.close_returns.head())
                # print(current_selection_df['close_returns'].std())
                print(f"{count} rows surpass the normal distribution. "
                      f"This is a {np.round((count/len(current_selection_df)*100), 2)}% of the set.")
