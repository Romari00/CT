import pandas as pd
from sklearn.preprocessing import StandardScaler

metadata = pd.read_excel("D:/datasets/updated_metadata.xlsx")
print("Columns before renaming:", metadata.columns)
metadata.columns = metadata.columns.str.strip()
print("Columns after renaming:", metadata.columns)

metadata['Age (year)'] = pd.to_numeric(metadata['Age (year)'], errors='coerce')
metadata['Calcium_score'] = pd.to_numeric(metadata['Calcium_score'], errors='coerce')
metadata['Epicardial Tissue Volume (ml)'] = pd.to_numeric(metadata['Epicardial Tissue Volume (ml)'], errors='coerce')
metadata['Pericardial (Mediastinal) Tissue Volume(ml)'] = pd.to_numeric(metadata['Pericardial (Mediastinal) Tissue Volume(ml)'], errors='coerce')
metadata['Sum of Cardiac Fats(ml) or Paracardial (Thoracic) Tissue Volume'] = pd.to_numeric(metadata['Sum of Cardiac Fats(ml) or Paracardial (Thoracic) Tissue Volume'], errors='coerce')

metadata = metadata.dropna()

def classify_risk(cac_score):
    if cac_score <= 10:
        return 0
    elif 11 <= cac_score <= 100:
        return 1
    elif 101 <= cac_score <= 400:
        return 2
    else:
        return 3

metadata['Risk'] = metadata['Calcium_score'].apply(classify_risk)

scaler = StandardScaler()
normalized_data = scaler.fit_transform(metadata[['Age (year)', 'Calcium_score', 'Epicardial Tissue Volume (ml)',
                                                 'Pericardial (Mediastinal) Tissue Volume(ml)',
                                                 'Sum of Cardiac Fats(ml) or Paracardial (Thoracic) Tissue Volume']])

normalized_df = pd.DataFrame(normalized_data, columns=['Age (year)', 'Calcium_score', 'Epicardial Tissue Volume (ml)',
                                                      'Pericardial (Mediastinal) Tissue Volume(ml)',
                                                      'Sum of Cardiac Fats(ml) or Paracardial (Thoracic) Tissue Volume'])

normalized_df['Health'] = metadata['Health'].values
normalized_df['Risk'] = metadata['Risk'].values

print(normalized_df.head())

normalized_df.to_excel("D:/datasets/normalized_patients_with_health_and_risk.xlsx", index=False)
