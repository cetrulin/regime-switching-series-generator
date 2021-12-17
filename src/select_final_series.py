import pandas as pd
import os
from matplotlib import pyplot as plt

path = "C:\\Users\\suare\\PycharmProjects\\RegimeSwitchingSeriesGenerator\\output\\synthetic3\\"

differences = list()
for file in os.listdir(path):
    if 'differences' not in file:
        df = pd.read_csv(path+file, usecols=['ts'])
        df = df.tail(1200000)
        diff = df.ts.max() - df.ts.min()
        print(f'In file {file} the diff between prices is {diff}')
        differences.append({'file': str(file), 'diff': diff})
differences_df = pd.DataFrame(differences)
differences_df.to_csv(path+'differences_tail.csv')

thresh = (differences_df['diff'].mean() - differences_df['diff'].std())

# differences_df = pd.read_csv(path + 'differences.csv', )

for index, row in differences_df[differences_df['diff'] < thresh].iterrows():
    diff, file = row
    # print(file)
    # print(diff)
    # print('HELLO')
    print(f'Plotting file {file} with a price diff of {diff}')
    df = pd.read_csv(path+file)
    df.ts.plot(figsize=(15, 6), title=diff)
    plt.savefig(path+file.replace('csv', 'png'))
    plt.show()
    plt.close()

    # selected: timeseries_created_1607572321
