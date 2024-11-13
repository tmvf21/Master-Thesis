import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unidecode import unidecode
from workalendar.europe import Portugal

# Selected buildings, floors and rooms. Empty list to process all.
selected_buildings = [] # e.g. ['Torre Norte', 'Pavilhão Central]
selected_floors = []
selected_rooms = []

# Start and end date to extrapolate data.
start_date = datetime.strptime("01/09/2021", "%d/%m/%Y")
end_date = datetime.strptime("01/07/2024", "%d/%m/%Y")

# Base path of the csv files.
base_path = './IST_Schedule/Alameda'

def sanitize_name(name):
    name = unidecode(name)
    name = re.sub(r'[^\w\s-]', '', name)
    return name

def generate_time_sample(start_date: datetime, end_date: datetime):
    time_sample = []
    current_date = start_date
    
    while current_date <= end_date:
        for hour in range(24):
            cal = Portugal()

            # TODO: add carnival(mon and tue) and easter as non-working days
            working_day = int(cal.is_working_day(current_date))
            week_number = current_date.isocalendar().week
            week_day = current_date.isocalendar().weekday

            time_sample.append([
                current_date.year,
                current_date.month,
                current_date.day,
                week_number,
                week_day,
                working_day,
                hour
            ])

        current_date += timedelta(days=1)
    
    time_sample = np.array(time_sample)
    
    return time_sample

def analyze_path(path: str):
    path_parts = path.split('/')
    # print(path_parts)

    if len(path_parts) < 4:
        return None, None
    elif len(path_parts) == 4:
        building = path_parts[3]
        floor = None
    elif len(path_parts) > 4:
        building = path_parts[3]
        floor = path_parts[len(path_parts) - 1]

    return building, floor

def analyse_room(df: pd.DataFrame):
    Lesson_Type_ratios = df['Lesson_Type'].value_counts(normalize=True)
    # dominant_lesson_type = Lesson_Type_ratios.idxmax()
    unique_combinations = df[['Lesson_Type', 'Shift', 'Max_Occupation', 'Lab_Group_Elements', 'Course_id']].drop_duplicates()
    Max_Occupation_mean = int (np.ceil(unique_combinations['Max_Occupation'].mean()))
    NaN_ratio = unique_combinations['Lab_Group_Elements'].isnull().sum() / len(unique_combinations)
    
    # print(unique_combinations)
    # print(Lesson_Type_ratios)
    # print(f'Max_Occupation_mean: {Max_Occupation_mean}')
    # print(f'NaN_ratio: {NaN_ratio}')

    if 'L' not in Lesson_Type_ratios:
        # print('Exit: L Lesson_Type not found')
        return 0, 0
    else:
        if Lesson_Type_ratios['L'] < 0.1:
            # print('Exit: L Lesson_Type ratio < 0.1')
            return 0, 0

    if Max_Occupation_mean > 40:
        # print('Exit: Max_Occupation_mean > 40')
        return 0, 0
    
    if NaN_ratio > 0.9:
        # print('Exit: NaN_ratio > 0.9')
        return 0, 0
    
    unique_combinations_aux = unique_combinations.dropna(subset=['Lab_Group_Elements'])
    
    if unique_combinations_aux.empty:
        # print('Exit: All Lab_Group_Elements are NaN')
        return 0, 0
    
    # print('Exit: Normal')
      
    Group_Elements_mean = int(np.round(unique_combinations_aux['Lab_Group_Elements'].mean()))
    unique_combinations_aux = unique_combinations_aux[unique_combinations_aux['Max_Occupation'] != 0]
    
    if unique_combinations_aux.empty:
        # print('Exit: All Max_Occupation are NaN')
        return 0, 0

    N_Groups_list = []
    for _, row in unique_combinations_aux.iterrows():
        Max_Occupation = row['Max_Occupation']
        N_Elements = row['Lab_Group_Elements']

        N_Groups_list.append(np.ceil(Max_Occupation / N_Elements))
    
    Max_N_Groups = int(np.ceil(np.mean(N_Groups_list)))
    
    return Max_N_Groups, Group_Elements_mean

def get_course_quc_attendance(course: pd.DataFrame, event):
    lesson_type = event['Lesson_Type']
    
    if lesson_type == 'T': # theoretical
        if course['QUC_T_attendance'].isnull().any():
            return 0.4
        else:
            return course['QUC_T_attendance']
    
    elif lesson_type == 'TP': # theoretical-practical
        if course['QUC_TP_attendance'].isnull().any():
            return 0.5
        else:
            return course['QUC_TP_attendance']
    
    elif lesson_type == 'PB'or lesson_type == 'P': # problem-based
        if course['QUC_PB_attendance'].isnull().any():
            return 0.8
        else:
            return course['QUC_PB_attendance']
    
    elif lesson_type == 'L': # laboratory
        if course['QUC_Lab_attendance'].isnull().any():
            return 0.9
        else:
            return course['QUC_Lab_attendance']
    
    elif lesson_type == 'S': # seminar
        return 0.9
    
    elif lesson_type == 'OT': # orientation tutorial
        return 0.9
    
    elif lesson_type == 'TC': # field work
        return 0.9
    
    elif lesson_type == 'E': # internship
        return 0.9
    
    else:
        print(f'Error getting QUC attendance (lesson_type = {lesson_type})!')
        return -1

def get_course_laptop_usage(course: pd.DataFrame, event):
    lesson_type = event['Lesson_Type']

    if lesson_type == 'T': # theoretical
        if course['T_Laptops'].isnull().any():
            return 0.05
        else:
            return course['T_Laptops']
    
    elif lesson_type == 'TP': # theoretical-practical
        if course['TP_Laptops'].isnull().any():
            return 0.06
        else:
            return course['TP_Laptops']
    
    elif lesson_type == 'PB'or lesson_type == 'P': # problem-based
        if course['PB_Laptops'].isnull().any():
            return 0.2
        else:
            return course['PB_Laptops']
    
    elif lesson_type == 'L': # laboratory
        if course['Lab_Laptops'].isnull().any():
            return 0.8
        else:
            return course['Lab_Laptops']
    
    elif lesson_type == 'S': # seminar
        return 0.1
    
    elif lesson_type == 'OT': # orientation tutorial
        return 0.1
    
    elif lesson_type == 'TC': # field work
        return 0.1
    
    elif lesson_type == 'E': # internship
        return 0.1
    
    else:
        print(f'Error getting laptop usage (lesson_type = {lesson_type})!')
        return -1

def get_course_desktop_usage(course: pd.DataFrame, event):
    lesson_type = event['Lesson_Type']

    if lesson_type == 'T': # theoretical
        if course['T_Desktops'].isnull().any():
            return 0
        else:
            return course['T_Desktops']

    elif lesson_type == 'TP': # theoretical-practical
        if course['TP_Desktops'].isnull().any():
            return 0
        else:
            return course['TP_Desktops']
    
    elif lesson_type == 'PB'or lesson_type == 'P': # problem-based
        if course['PB_Desktops'].isnull().any():
            return 0.3
        else:
            return course['PB_Desktops']
    
    elif lesson_type == 'L': # laboratory
        if course['Lab_Desktops'].isnull().any():
            return 0.8
        else:
            return course['Lab_Desktops']
    
    elif lesson_type == 'S': # seminar
        return 0
    
    elif lesson_type == 'OT': # orientation tutorial
        return 0
    
    elif lesson_type == 'TC': # field work
        return 0
    
    elif lesson_type == 'E': # internship
        return 0
    
    else:
        print(f'Error getting desktop usage (lesson_type = {lesson_type})!')
        return -1

def estimate_students_in_room(students_enrolled, attendance_ratio):
    # add some gaussian noise to the attendance_ratio
    attendance_ratio = attendance_ratio + np.random.normal(0, 0.1 * attendance_ratio)
    attendance_ratio = np.clip(attendance_ratio, 0, 1)

    students_in_lesson = int (np.round(students_enrolled * attendance_ratio))
    return students_in_lesson

def estimate_laptops_in_use(students_in_lesson, laptop_ON_ratio):
    laptop_SB_ratio = np.random.uniform(0, 0.05)
    laptop_SB_ratio = np.clip(laptop_SB_ratio, 0, 1)
    
    # add some gaussian noise to the laptop_ON_ratio
    laptop_ON_ratio = laptop_ON_ratio + np.random.normal(0, 0.1 * laptop_ON_ratio)
    laptop_ON_ratio = np.clip(laptop_ON_ratio, 0, 1)
    
    N_Laptops_ON = int(np.ceil(students_in_lesson * laptop_ON_ratio))
    if N_Laptops_ON >= students_in_lesson:
        N_Laptops_ON = students_in_lesson

    # TODO: check this
    N_Laptops_SB = int(np.ceil(students_in_lesson * laptop_SB_ratio))
    if N_Laptops_ON + N_Laptops_SB > students_in_lesson:
        N_Laptops_SB = 0

    return N_Laptops_ON, N_Laptops_SB

def estimate_desktops_in_use(event, students_in_lesson, desktop_ON_ratio, n_desktops, room_group_elements_mean):
    if n_desktops == 0:
        N_Desktops_ON = 0
        N_Desktops_SB = 0
        return N_Desktops_ON, N_Desktops_SB
    
    # add some gaussian noise to the desktop_ON_ratio
    desktop_ON_ratio = desktop_ON_ratio + np.random.normal(0, 0.05 * desktop_ON_ratio)
    desktop_ON_ratio = np.clip(desktop_ON_ratio, 0, 1)
    
    students_enrolled = event['Enrolled_Students']
    group_elements = event['Lab_Group_Elements']

    if np.isnan(group_elements):
        # print(f'Group elements is nan (shift {event["Shift"]})!')
        group_elements = room_group_elements_mean

    n_groups_in_lesson = int (np.ceil(students_enrolled / group_elements))
    
    N_Desktops_ON = int (np.ceil(n_groups_in_lesson * desktop_ON_ratio))
    if N_Desktops_ON >= n_desktops:
        N_Desktops_ON = n_desktops
    
    if N_Desktops_ON > students_in_lesson:
        N_Desktops_ON = students_in_lesson
        
    N_Desktops_SB = int (np.round(n_desktops - N_Desktops_ON))

    return N_Desktops_ON, N_Desktops_SB

def gen_laptop_power(state):
    laptop_power = -1

    # State: 0 - SB, 1 - ON
    if state == 0:
        laptop_power = abs(np.random.normal(40, 20))
    else:
        laptop_power = abs(np.random.normal(100, 50))

    return laptop_power

def gen_desktop_power(state):
    desktop_power = -1
    
    # State: 0 - SB, 1 - ON
    if state == 0:
        desktop_power = abs(np.random.normal(100, 50))
    else:
        desktop_power = abs(np.random.normal(800, 200))
    
    return desktop_power

def calc_laptop_power(N_Laptops_ON, N_Laptops_SB, hour_ratio):
    Laptops_ON_Power = 0
    Laptops_SB_Power = 0

    if N_Laptops_ON > 0:
        for i in range(N_Laptops_ON):
            Laptop_i = gen_laptop_power(1) * hour_ratio
            Laptops_ON_Power += abs(Laptop_i + np.random.normal(0, 0.2 * Laptop_i))

    if N_Laptops_SB > 0:
        for i in range(N_Laptops_SB):
            Laptop_i = gen_laptop_power(0) * hour_ratio
            Laptops_SB_Power += abs(Laptop_i + np.random.normal(0, 0.1 * Laptop_i))
    
    return Laptops_ON_Power, Laptops_SB_Power

def calc_desktop_power(N_Desktops_ON, N_Desktops_SB, hour_ratio):
    Desktops_ON_Power = 0
    Desktops_SB_Power = 0

    if N_Desktops_ON > 0:
        for i in range(N_Desktops_ON):
            Desktop_i = gen_desktop_power(1) * hour_ratio
            Desktops_ON_Power += abs(Desktop_i + np.random.normal(0, 0.1 * Desktop_i))

    if N_Desktops_SB > 0:
        for i in range(N_Desktops_SB):
            Desktop_i = gen_desktop_power(0) * hour_ratio
            Desktops_SB_Power += abs(Desktop_i + np.random.normal(0, 0.05 * Desktop_i))
    
    return Desktops_ON_Power, Desktops_SB_Power

if __name__ == '__main__':
    time_sample = generate_time_sample(start_date, end_date)
    df_time = pd.DataFrame(time_sample, columns=['Year', 'Month', 'Day', 'Week_Num', 'Week_Day', 'Working_Day', 'Hour'])
    df_courses = pd.read_csv('Courses.csv')

    for root, dirs, files in os.walk(base_path):
        if files:
            path = root.replace('\\', '/')
            print(f"Exploring {path} ...")
            
            building, floor = analyze_path(path)
            # print(f"Building: {building}, Floor: {floor}")
            
            if building and floor:
                info_path = f'{path}/_{building}_{floor}_info.csv'
                save_info_path = info_path.replace('IST_Schedule', 'IST_Power_Load')

            selected_buildings = [sanitize_name(building).lower() for building in selected_buildings]

            if building and (not selected_buildings or building.replace('_', ' ').lower() in selected_buildings):
                if floor and (not selected_floors or floor in selected_floors):
                    df_floor_flag = False
                    try:
                        df_floor = pd.read_csv(info_path)
                    except pd.errors.EmptyDataError:
                        print(info_path, 'is empty!')

                    for file in files:
                        if file.startswith("_"):
                                continue
                        elif file.endswith(".csv"):
                            room_name = file.replace('_', ' ').replace('.csv', '')
                            selected_rooms = [sanitize_name(room).lower() for room in selected_rooms]

                            if not selected_rooms or room_name.lower() in selected_rooms:
                                print(f"Processing {file} ...")

                                file_path = os.path.join(root, file).replace('\\', '/')
                                save_file_path = file_path.replace('IST_Schedule', 'IST_Power_Load').replace('.csv', '_power.csv')

                                df_csv = pd.read_csv(file_path)
                                
                                n_desktops, group_elements = analyse_room(df_csv)

                                df_floor.loc[df_floor['Space_Name'] == room_name, 'Estimated_Desktops'] = n_desktops
                                df_floor.loc[df_floor['Space_Name'] == room_name, 'Estimated_Group_Elements'] = group_elements
                                df_floor_flag = True
                                
                                df = pd.merge(df_time, df_csv, on=['Year', 'Month', 'Day', 'Week_Num', 'Week_Day', 'Hour'], how='left')
                                df['Room_in_use'] = df['Course_Name'].notnull().astype(int)
                                df = df.sort_values(by=['Year', 'Month', 'Week_Num', 'Day', 'Hour'])

                                for index, row in df.iterrows():
                                    if pd.notna(row['Hour_Ratio']):
                                        if row['Hour_Ratio'] != 1 and index != 0:
                                            if pd.isna(df.at[index - 1, 'Hour_Ratio']):
                                                new_row = df.loc[index - 1].copy()
                                                new_row['Hour'] = row['Hour']
                                                new_row['Hour_Ratio'] = row['Hour_Ratio']
                                                df.loc[index - 0.5] = new_row
                                            elif pd.isna(df.at[index + 1, 'Hour_Ratio']):
                                                new_row = df.loc[index + 1].copy()
                                                new_row['Hour'] = row['Hour']
                                                new_row['Hour_Ratio'] = row['Hour_Ratio']
                                                df.loc[index + 0.5] = new_row
                                df = df.sort_index().reset_index(drop=True)

                                df.fillna({'Hour_Ratio': 1}, inplace=True)
                                
                                for index, row in df.iterrows():
                                    if row['Room_in_use'] == 1:
                                        course = df_courses.loc[df_courses['Course_id'] == row['Course_id']]

                                        laptop_usage = get_course_laptop_usage(course, row)
                                        desktop_usage = get_course_desktop_usage(course, row)
                                        attendance_ratio = get_course_quc_attendance(course, row)

                                        if (laptop_usage or desktop_usage or attendance_ratio) == -1:
                                            print(f'Error getting course info ({row["Course_id"]})!')
                                            exit(0)
                                        
                                        students_in_lesson = estimate_students_in_room(row['Enrolled_Students'], attendance_ratio)
                                        N_Laptops_ON, N_Laptops_SB = estimate_laptops_in_use(students_in_lesson, laptop_usage)
                                        N_Desktops_ON, N_Desktops_SB = estimate_desktops_in_use(row, students_in_lesson, desktop_usage, n_desktops, group_elements)

                                        # print('Course:\t', row['Course_Name'],
                                        #       '\nLesson_Type:\t', row['Lesson_Type'],
                                        #       '\nStudents in lesson:', students_in_lesson,
                                        #       '\tN_Laptops_ON:', N_Laptops_ON,
                                        #       '\tN_Laptops_SB:', N_Laptops_SB,
                                        #       '\tN_Desktops_ON:', N_Desktops_ON,
                                        #       '\tN_Desktops_SB:', N_Desktops_SB,
                                        #       '\n')
                                    
                                    else:
                                        students_in_lesson = 0
                                        N_Laptops_ON , N_Laptops_SB = 0, 0
                                        N_Desktops_ON, N_Desktops_SB = 0, n_desktops

                                    hour_ratio = row['Hour_Ratio']

                                    Desktops_ON_Power, Desktops_SB_Power = calc_desktop_power(N_Desktops_ON, N_Desktops_SB, hour_ratio)
                                    Laptops_ON_Power, Laptops_SB_Power = calc_laptop_power(N_Laptops_ON, N_Laptops_SB, hour_ratio)

                                    Desktops_Total_Power = Desktops_ON_Power + Desktops_SB_Power
                                    Laptops_Total_Power = Laptops_ON_Power + Laptops_SB_Power

                                    Total_Power = Desktops_Total_Power + Laptops_Total_Power
                                    
                                    df.at[index, 'Room_Estimated_Students'] = students_in_lesson
                                    
                                    df.at[index, 'N_Desktops_ON'] = N_Desktops_ON
                                    df.at[index, 'N_Desktops_SB'] = N_Desktops_SB

                                    df.at[index, 'Desktops_ON_Power'] = Desktops_ON_Power
                                    df.at[index, 'Desktops_SB_Power'] = Desktops_SB_Power

                                    df.at[index, 'N_Laptops_ON'] = N_Laptops_ON
                                    df.at[index, 'N_Laptops_SB'] = N_Laptops_SB

                                    df.at[index, 'Laptops_ON_Power'] = Laptops_ON_Power
                                    df.at[index, 'Laptops_SB_Power'] = Laptops_SB_Power
                                    
                                    df.at[index, 'Desktops_Total_Power'] = Desktops_Total_Power
                                    df.at[index, 'Laptops_Total_Power'] = Laptops_Total_Power
                                    df.at[index, 'Total_Power'] = Total_Power

                                os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
                                df.to_csv(save_file_path, index=False)
                    
                    if df_floor_flag:
                        df_floor.dropna(subset=['Estimated_Desktops'], inplace=True)
                        os.makedirs(os.path.dirname(save_info_path), exist_ok=True)
                        df_floor.to_csv(save_info_path, index=False)
                        print(f"Saved {save_info_path} ...")
    
    print('\nDone!')
