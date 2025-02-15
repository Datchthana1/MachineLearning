import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# โหลดชุดข้อมูล
df = pd.read_csv(
    rf"Y:\ALLfile\Real-Data (Processed)\Concat - Result - ResampleMax - Normalization\SR1600_1s_Scaled.csv")
selected_participants = [f'S{str(i).zfill(2)}' for i in range(1, 31)]

# เลือกแถวที่มีผู้เข้าร่วมการทดลองตามที่เลือก
df = df[df['SubjectCode'].isin(selected_participants)]
print(df.columns)

# แยกฟีเจอร์และตัวแปรเป้าหมาย
X = df[['Cleaned_EDA_EmotiBit',
        'EDA_Tonic_EmotiBit', 'EDA_Phasic_EmotiBit', 'PPG_Rate', 'PPG_Clean',
        'SkinTemp_Emo', 'Gender', 'hf_PPG', 'lf_PPG', 'lf_hf_PPG', 'Height', 'Weight', 'WHR', 'BMI',
        'FatPercentage', 'FatMass', 'MuscleMass', 'MusclePercentage',
        'WeightNonFat', 'BodyAge', 'BoneMass', 'BMR', 'WaterBody',
        'ProteinPercentage', 'Striated', 'VFI', 'RoomTemp', 'TempHandNormal',
        'TempHandBeforeHot', 'TempHotGel', 'TempHandAfterHot',
        'TempHandBeforeCold', 'TempColdGel', 'TempHandAfterCold', ]]  # ฟีเจอร์
y = df['PainLevel']  # ตัวแปรเป้าหมาย

# คำนวณ Information Gain โดยใช้ mutual_info_classif
information_gain = mutual_info_classif(X, y)
print(information_gain)

# จับคู่ฟีเจอร์กับค่า Information Gain
info_gain_pairs = zip(X.columns, information_gain)

# เรียงลำดับค่า Information Gain จากมากไปน้อย
sorted_info_gain = sorted(info_gain_pairs, key=lambda x: x[1], reverse=True)

# แสดงผลลัพธ์เรียงลำดับจากมากไปน้อย
print("\nInformation Gain sorted from highest to lowest:")
result = {"Features": [], "Information Gain": []}

for feature, ig in sorted_info_gain:
    result['Features'].append(feature)
    result['Information Gain'].append(ig)

# สร้าง DataFrame จากผลลัพธ์
df_result = pd.DataFrame(result)
df_result.to_csv(
    "D:\OneFile\WorkOnly\AllCode\Python\Project - B (PainManagement)\AllCode\โค้ดที่ไม่จำเป็นต้องใช้\OutputCV\Information_Gain.csv")

# แสดงผลลัพธ์ในรูปแบบ DataFrame
print(df_result)
