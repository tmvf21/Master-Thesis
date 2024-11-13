import os
import numpy as np
import pandas as pd

base_path = './IST_Power_Load/Alameda'
save_data_path = './data/'

def analyze_path(path: str):
    path_parts = path.split('/')

    if len(path_parts) < 4:
        return None, None, None
    elif len(path_parts) == 4:
        building = path_parts[3]
        floor = None
        room = None
    elif len(path_parts) > 4:
        building = path_parts[3]
        floor = path_parts[len(path_parts) - 2]
        room = path_parts[len(path_parts) - 1].replace('_power.csv', '')

    return building, floor, room

def process_data(df_data: pd.DataFrame, selected_space: str, selected_space_type: str) -> pd.DataFrame:
    if (selected_space or selected_space_type) is None:
        print('Please provide a space and space type...')
        exit()

    df_data['Lesson_Type'] = df_data['Lesson_Type'].replace({'P': 'PB'})

    def lesson_count(df: pd.DataFrame) -> pd.DataFrame:
        lesson_counts = pd.DataFrame()

        lesson_counts = df.pivot_table(index=['Year', 'Month', 'Day', 'Week_Num', 'Week_Day', 'Working_Day', 'Hour'], 
                                columns='Lesson_Type', 
                                aggfunc='size', 
                                fill_value=0).reset_index()
        
        if 'T' not in lesson_counts.columns:
            lesson_counts['T'] = 0
        if 'TP' not in lesson_counts.columns:
            lesson_counts['TP'] = 0
        if 'PB' not in lesson_counts.columns:
            lesson_counts['PB'] = 0
        if 'L' not in lesson_counts.columns:
            lesson_counts['L'] = 0
        if 'S' not in lesson_counts.columns:
            lesson_counts['S'] = 0
        if 'OT' not in lesson_counts.columns:
            lesson_counts['OT'] = 0
        if 'TC' not in lesson_counts.columns:
            lesson_counts['TC'] = 0
        if 'E' not in lesson_counts.columns:
            lesson_counts['E'] = 0

        lesson_counts['Others'] = lesson_counts['S'] + lesson_counts['OT'] + lesson_counts['TC'] + lesson_counts['E']
        lesson_counts.drop(columns=['S', 'OT', 'TC', 'E'], inplace=True)
        
        lesson_counts.columns.name = None
        lesson_counts = lesson_counts.rename(columns={
            'T': 'Count_T',
            'TP': 'Count_TP',
            'PB': 'Count_PB',
            'L': 'Count_L',
            'Others': 'Count_O',
        })
    
        return lesson_counts
    
    if selected_space_type == 'Campus':
        df_aux = df_data
        lesson_counts = lesson_count(df_aux)
        grouped_df = df_aux.groupby(['Year', 'Month', 'Day', 'Week_Num', 'Week_Day', 'Working_Day', 'Hour']).agg({
            'Enrolled_Students': 'sum',
            'Room_in_use': 'sum',
            'Total_Power': 'sum'
        }).reset_index()
    elif selected_space_type == 'Building':
        df_aux = df_data[df_data['Building'] == selected_space]
        lesson_counts = lesson_count(df_aux)
        grouped_df = df_aux.groupby(['Year', 'Month', 'Day', 'Week_Num', 'Week_Day', 'Working_Day', 'Hour']).agg({
            'Enrolled_Students': 'sum',
            'Room_in_use': 'sum',
            'Total_Power': 'sum'
        }).reset_index()
    elif selected_space_type == 'Room':
        df_aux = df_data[df_data['Room'] == selected_space]
        lesson_counts = lesson_count(df_aux)
        grouped_df = df_aux.groupby(['Year', 'Month', 'Day', 'Week_Num', 'Week_Day', 'Working_Day', 'Hour']).agg({
            'Enrolled_Students': 'sum',
            'Room_in_use': 'sum',
            'Total_Power': 'sum'
        }).reset_index()

    grouped_df = grouped_df.rename(columns={'Room_in_use': 'Rooms_in_use', 'Total_Power': 'Wh'})

    df = pd.merge(grouped_df, lesson_counts, on=['Year', 'Month', 'Day', 'Week_Num', 'Week_Day', 'Working_Day', 'Hour'], how='left')

    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])

    df.set_index('Date', inplace=True)
    
    df = df[['Year', 'Month', 'Day', 'Week_Num', 'Week_Day', 'Working_Day', 'Hour', 
             'Enrolled_Students', 'Rooms_in_use', 
             'Count_T', 'Count_TP', 'Count_PB', 'Count_L', 'Count_O',
             'Wh']].sort_values(by=['Year', 'Month', 'Day', 'Hour']).fillna(0)
    
    if df['Count_T'].sum() == 0:
        df.drop(columns=['Count_T'], inplace=True)
    if df['Count_TP'].sum() == 0:
        df.drop(columns=['Count_TP'], inplace=True)
    if df['Count_PB'].sum() == 0:
        df.drop(columns=['Count_PB'], inplace=True)
    if df['Count_L'].sum() == 0:
        df.drop(columns=['Count_L'], inplace=True)
    if df['Count_O'].sum() == 0:
        df.drop(columns=['Count_O'], inplace=True)
        
    return df

if __name__ == '__main__':
    selected_space = ['Alameda', 'Torre_Norte', 'LSDC1'] # Use the name of the file or folder
    selected_space_type = ['Campus', 'Building', 'Room']

    df = pd.DataFrame()
    df_info_csv = pd.DataFrame()

    for root, dirs, files in os.walk(base_path):
        path = root.replace('\\', '/')
        building, floor, room = analyze_path(path)
        
        for file in files:
            if file.startswith('_' + building) or not file.endswith(".csv"):
                file_path = os.path.join(root, file).replace('\\', '/')
                print(f"Processing {file_path} ...")
                
                df_info_csv = pd.read_csv(file_path)
                
                break
                
        for file in files:
            if file.endswith("_power.csv"):
                file_path = os.path.join(root, file).replace('\\', '/')
                print(f"Processing {file_path} ...")                
                
                building, floor, room = analyze_path(file_path)
                # print(f"Building: {building}, Floor: {floor}, Room: {room}")

                df_info_aux = df_info_csv[df_info_csv['Space_Name'] == room.replace('_', ' ')]
                                
                file_path = os.path.join(root, file).replace('\\', '/')
                df_csv = pd.read_csv(file_path)

                df_csv['Enrolled_Students'] = df_csv['Enrolled_Students'].fillna(0).astype(int)
                df_csv['Lesson_Type'] = df_csv['Lesson_Type'].fillna('None')

                df_aux = df_csv.groupby(['Year', 'Month', 'Day', 'Week_Num', 'Week_Day', 'Working_Day', 'Hour', 'Lesson_Type', 'Enrolled_Students', 'Room_in_use']).agg({'Total_Power': 'sum'}).reset_index()
                df_aux['Building'] = building
                df_aux['Floor'] = floor
                df_aux['Room'] = room
                
                df_aux['Room_id'] = df_info_aux['Space_id'].values[0]

                df_aux = df_aux[['Building', 'Floor', 'Room', 'Room_id', 'Year', 'Month', 'Day', 'Week_Num', 'Week_Day', 'Working_Day', 'Hour', 'Lesson_Type', 'Enrolled_Students', 'Room_in_use', 'Total_Power']]

                # os.makedirs(save_data_path + 'All_Spaces', exist_ok=True)
                # df_aux.to_csv(f'{save_data_path}/All_Spaces/{file}', index=False)

                df = pd.concat([df, df_aux])

    # print('\nSaving raw data...')
    # df.to_csv(save_data_path + 'data_raw.csv', index=False)

    print('\nProcessing data...')
    if len(selected_space) != len(selected_space_type):
        print('Please provide the same number of spaces and space types...')
        exit()
    for space in selected_space:
        processed_data = process_data(df, selected_space=space, selected_space_type=selected_space_type[selected_space.index(space)])
        os.makedirs('./data', exist_ok=True)
        processed_data.to_csv(save_data_path + 'data_' + space + '.csv', index=True, index_label='Date')

    print('\nDone!')