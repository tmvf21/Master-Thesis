import os
import pandas as pd
import requests

# Base path of the root of the schedule csv files
base_path = './IST_Schedule/Alameda'

if __name__ == '__main__':
    df_unique_courses = pd.DataFrame(columns=['Course_Acronym', 'Course_Name', 'Academic_Term', 'Enrolled_Students', 'Course_url', 'Course_id',])
    
    course_url = f"https://fenix.tecnico.ulisboa.pt/api/fenix/v1/courses/"
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.startswith("_") or not file.endswith(".csv"):
                continue
            
            print(f"Processing {file}...")
            
            file_path = os.path.join(root, file).replace('\\', '/')
            df = pd.read_csv(file_path)

            unique_combinations = df[['Course_Acronym', 'Course_Name', 'Academic_Term', 'Course_id']].drop_duplicates()
            
            df_unique_courses = pd.concat([df_unique_courses, unique_combinations])
    
    df_unique_courses = df_unique_courses.drop_duplicates()

    print('\nUnique courses found:', len(df_unique_courses))

    print('\nRetrieving courses info data...')

    for index, row in df_unique_courses.iterrows():
        course_id = row['Course_id']
        response = requests.get(course_url + str(course_id))
        if response.status_code == 200:
            course_info = response.json()
        else:
            print(f"Failed to retrieve course info data: {response.status_code} for course ID {course_id}")
            exit(0)
        
        df_unique_courses.at[index, 'Course_Enrolled_Students'] = course_info['numberOfAttendingStudents']
        df_unique_courses.at[index, 'Course_url'] = course_info['url']
        
    df_unique_courses = df_unique_courses.sort_values(by=['Course_Acronym'])
    
    new_columns = [
        'QUC_T_attendance', 'QUC_TP_attendance', 'QUC_PB_attendance', 'QUC_Lab_attendance',
        'T_Laptops', 'TP_Laptops', 'PB_Laptops', 'Lab_Laptops',
        'T_Desktops', 'TP_Desktops', 'PB_Desktops', 'Lab_Desktops'
    ]
    
    for col in new_columns:
        df_unique_courses[col] = None
    
    df_unique_courses.to_csv("Courses.csv", index=False)

    print('\nDone!')
    