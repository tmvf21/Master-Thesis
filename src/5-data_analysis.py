import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

font = {'size': 14}
plt.rc('font', **font)

data_path = './data/data_'
save_path = './Img/'

def heatmaps(df: pd.DataFrame, selected_space: str) -> None:
    plt.figure(figsize=(10, 6))
    for day, group in df.groupby(['Year', 'Month', 'Day']):
        plt.plot(group['Hour'], group['Wh'] * 1e-3, color="tab:blue", alpha=0.1)
    plt.xlabel("Hour", fontsize=16)
    plt.ylabel("Electricity Consumption [kWh]", fontsize=16)
    plt.xticks(ticks=[i for i in range(0, 24)], labels=[str(i) for i in range(0,24)], fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path + selected_space + '_daily.png',  format='png', dpi=400)
    plt.close()

    plt.figure(figsize=(10, 6))
    for week, group in df.groupby(['Year', 'Week_Num']):
        group['Hour_of_Week'] = (group['Week_Day'] - 1) * 24 + group['Hour']
        plt.plot(group['Hour_of_Week'], group['Wh'] * 1e-3, color="tab:blue", alpha=0.1)
    plt.xlabel("Week Day", fontsize=16)
    plt.ylabel("Electricity Consumption [kWh]", fontsize=16)
    plt.xticks(ticks=[i * 24 for i in range(7)], labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path + selected_space + '_weekly.png',  format='png', dpi=400)
    plt.close()

    # plt.figure(figsize=(15, 8))
    # for month, group in df.groupby(['Year', 'Month']):
    #     group['Hour_of_Month'] = (group['Day'] - 1) * 24 + group['Hour']
    #     plt.plot(group['Hour_of_Month'], group['Wh'] * 1e-3, color="tab:blue", alpha=0.05)
    # plt.xlabel("Day of the Month")
    # plt.ylabel("Electricity Consumption [kWh]")
    # plt.xticks(ticks=[i * 24 for i in range(1, 32)], labels=[str(i) for i in range(1, 32)])
    # plt.tight_layout()
    # plt.savefig(save_path + selected_space + '_monthly.pdf', dpi=400)
    # plt.close()

    # plt.figure(figsize=(15, 8))
    # for year, group in df.groupby(['Year']):
    #     group['Day of Year'] = group['Month'] * 30 + group['Day']
    #     group['Hour of Year'] = (group['Day of Year'] - 1) * 24 + group['Hour']
    #     plt.plot(group['Month'], group['Wh'] * 1e-3, color="tab:blue", alpha=0.5)
    # plt.xlabel("Month")
    # plt.ylabel("Electricity Consumption [kWh]")
    # plt.xticks(ticks=[i for i in range(1, 13)], labels=[str(i) for i in range(1, 13)])
    # plt.tight_layout()
    # plt.savefig(save_path + selected_space + '_yearly.pdf', dpi=400)
    # plt.close()

def rolling_avg(df: pd.DataFrame, selected_space: str) -> None:
    # Calculate rolling averages
    df['Daily_MA'] = df['Wh'].rolling(window=24, min_periods=1).mean()  # 24-hour rolling average
    df['Weekly_MA'] = df['Wh'].rolling(window=24*7, min_periods=1).mean()  # 7-day rolling average

    # Plot the original data and the moving averages
    plt.figure(figsize=(15, 8))
    plt.plot(df['Wh'], label="Hourly Consumption", color="lightgrey")
    plt.plot(df['Daily_MA'], label="24-hour Moving Average", color="blue", linewidth=2)
    plt.plot(df['Weekly_MA'], label="7-day Moving Average", color="orange", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Electricity Consumption (Wh)")
    plt.title("Electricity Consumption with Moving Averages")
    plt.legend()
    plt.show()

if __name__ == '__main__':

    spaces = ['Alameda', 'Torre_Norte', 'LSDC1']
    # selected_space = spaces[0]
    for selected_space in spaces:
        df = pd.read_csv(data_path + selected_space + '.csv', index_col='Date', parse_dates=True)
        # print(df.columns)

        # rolling_avg(df, selected_space)

        heatmaps(df, selected_space)




