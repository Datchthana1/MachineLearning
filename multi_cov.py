import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Deep EDA\\Data\\CleanedData\\Biopac\\15s\\15Sec\\00-Complete.csv",
#                  index_col=0, encoding="utf8")
# df = pd.read_csv(r"Y:\ALLfile\Real-Data (Processed)\Concat\SR1600_1s.csv")
#
# print(f'{df.keys()}')
#
# # df['ComfortLevel'] = LabelEncoder().fit_transform(df['ComfortLevel'])
# print(df['PainLevel'].unique())
# #
# #
# # x_columns = ['EDA_Tonic_BIOPAC', 'EDA_Phasic_BIOPAC']
# #
# print(f'{df.keys()}')
#
# # predictor = 'np.log(HF)'
# # predictor = 'np.log(LF)'
# # predictor = 'SkinTemp'
# # predictor = 'TN_TNPS'
# # outcome = 'np.log(LFHF)'
# # outcome = 'np.log(Air_Temperature)'
# cov1 = 'bmi'
# # cov1 = 'SkinTemp'
# # # cov1 = 'bmi' # 'LFHF', 'LF', 'HF'
# cov2 = 'SkinTemp'
# # cov3 = 'bmi'# 'RespirationRate', 'PulseRate', 'SkinTemp','AirTemperature' ,'AirHumidity'
# # eq_1 = '' + outcome + ' ~ ' + predictor + ''
# # eq_1 = '' + outcome + ' ~ ' + predictor + ' * ' + cov1 + ''
#
#

print(f"------------------- Procressing --------------------")
# y_columns = "PainLevel"
# x = df[x_columns]
# y = df[y_columns]

# equation = "PainLevel ~ EDA_Phasic_BIOPAC + EDA_Tonic_BIOPAC"
columns = ['Cleaned_EDA_EmotiBit', 'EDA_Tonic_EmotiBit',
           'EDA_Phasic_EmotiBit', 'PPG_Rate', 'PPG_Clean', 'SkinTemp_Emo',
           'hf_PPG', 'lf_PPG', 'lf_hf_PPG',
           'Height', 'Weight', 'WHR', 'BMI', 'FatPercentage',
           'MuscleMass', 'FatMass', 'BMR', 'MusclePercentage', 'WaterBody',
           'BoneMass', 'ProteinPercentage', 'Striated', 'VFI', 'RoomTemp',
           'TempHandNormal', 'TempHandBeforeHot', 'TempHotGel', 'TempHandAfterHot',
           'TempHandBeforeCold', 'TempColdGel', 'TempHandAfterCold']

models = [smf.mnlogit]
# columns = ["EDA_Phasic_BIOPAC", "EDA_Tonic_BIOPAC", "EDA_Phasic_BIOPAC"]0

# statsmodels utibility : mnlogit,


all_results = pd.DataFrame()
# for nmodel in models:
#     for column in columns:
#         equation = f"PainLevel ~ {column}"
#         model = nmodel(equation, data=df).fit()
#         print(f"Summary : \n{model.summary()}")
#         print(f"P-values : \n{model.pvalues}")
#         print(f"------------------- {column} -----------------------")
#     print(f"-----------------------------  -----------------------------------\n")

# equation = f"PainLevel ~ RAW_EDA_EmotiBit * EDA_Phasic_EmotiBit * EDA_Tonic_EmotiBit * PPG_Rate * hf_PPG * BMI * lf_PPG * lf_hf_PPG * FatPercentage * MuscleMass * FatMass * BMR * MusclePercentage * WaterBody * BoneMass * ProteinPercentage * Striated * VFI"
# equation = f"PainLevel ~ EDA_Phasic_EmotiBit + EDA_Tonic_EmotiBit + PPG_Rate + hf_PPG + lf_PPG"
# equation = f"PainLevel ~ EDA_Phasic_EmotiBit * EDA_Tonic_EmotiBit * BMI"
# equation = f"PainLevel ~ EDA_Phasic_EmotiBit * EDA_Tonic_EmotiBit + PPG_Rate + Weight"
# for samplingRate in range(1600, 999, -200):
df = pd.read_csv(rf"Y:\ALLfile\Real-Data (Processed)\Concat - Result - ResampleMax - Normalization\SR1600_1s_Scaled.csv")
print(df.keys())
# for columns in columns:
for columns in columns:
    # print(f"SR : {samplingRate}")
    # print(f'{df.keys()}')
    #
    # # df['ComfortLevel'] = LabelEncoder().fit_transform(df['ComfortLevel'])
    # print(df['PainLevel'].unique())
    # #
    # #
    # # x_columns = ['EDA_Tonic_BIOPAC', 'EDA_Phasic_BIOPAC']
    # #
    # print(f'{df.keys()}')
    # columns = ['Hight', 'Weight', "BMI", 'hf_PPG', 'lf_PPG', 'lf_hf_PPG']
    # columns = ['BMR']
    # for column in columns:
    #     equation = f"PainLevel ~ {column}"
    #     #     model = smf.mnlogit(equation, data=df).fit()
    #     #     print(f"Summary : \n{model.summary()}")
    #     #     print(f"P-values : \n{model.pvalues}")
    # equation = f"PainLevel ~ {columns}"
    # equation = f"PainLevel ~ EDA_Phasic_EmotiBit + EDA_Tonic_EmotiBit + PPG_Rate + BMI + lf_hf_PPG"
    # equation = f"PainLevel ~ EDA_Phasic_EmotiBit + EDA_Tonic_EmotiBit + BMI + lf_hf_PPG"
    #     model = smf.mnlogit(equation, data=df).fit()
    #     print(f"Summary : \n{model.summary()}")
    #     print(f"P-values : \n{model.pvalues}")
    # equation1 = f"PainLevel ~ EDA_Phasic_EmotiBit + EDA_Tonic_EmotiBit + lf_hf_PPG + SkinTemp_Emo + FatPercentage + PPG_Rate" #FatPercentage == 0.000
    # equation2 = f"PainLevel ~ EDA_Phasic_EmotiBit + EDA_Tonic_EmotiBit + lf_hf_PPG + SkinTemp_Emo + BMI "
    # equation4 = f"PainLevel ~ EDA_Phasic_EmotiBit + EDA_Tonic_EmotiBit + lf_hf_PPG + SkinTemp_Emo + MusclePercentage : FatPercentage + PPG_Rate"
    # equation = [equation1, equation2, equation4]
    equation4 = f"PainLevel ~ {columns}"
    equation = [equation4]
    for equation in equation:
        model = smf.mnlogit(equation, data=df).fit()
        print(f"Summary : \n{model.summary()}")
        print(f"P-values : \n{model.pvalues.to_string()}")
        params_df = pd.DataFrame(model.params)

    # for series_name, series in params_df.items():
    #     for i, v in series.items():
    #         print("class: ", series_name, " | key: ", i, " | odd value: ", np.exp(v))
    #         print(f'class: {series_name} | key: {i} Impact: {(float(np.exp(v)) - 1) * 100}')
    #     print()

# for columns in columns:
#     equation = f"PainLevel ~ {columns}"
#     model = smf.mnlogit(equation, data=df).fit()
#     print(f"Summary : \n{model.summary()}")
#     print(f"P-values : \n{model.pvalues}")
# all_results.to_csv("Test.csv", index=False)

# # print(rp.summary_cont(df).to_string())
# # df = df[(df["caseStatus"] == 0)]
# # df = df[(df["ComfortLevel"] == 0) | (df["ComfortLevel"] == 1)]
# for x in x_columns:
#     print(f'\n ---------- started loop {x}  ----------\n')
#     # eq_1 = '' + y_columns + ' ~ np.log(' + x + ') '
#     # eq_1 = '' + y_columns + ' ~ np.log(' + x + ') + ' + cov1 + ' + ' + cov2 + '+ ' + cov3 + ''
#     # eq_1 = '' + y_columns + ' ~ np.log(' + x + ')  + ' + cov1 + ''
#     # eq_1 = '' + y_columns + ' ~ np.log(' + x + ') + ' + cov1 + ' + np.log(' + x + ') : ' + cov1 + ''
#     # eq_1 = '' + y_columns + ' ~ ' + x + ' + np.log(' + x + ') : ' + cov1 + ' + np.log(' + x + ') :' + cov2 + ''
#     # eq_1 = '' + y_columns + ' ~ np.log(' + x + ') +  np.log(' + cov1 + ') : ' + cov2 + ''
#     eq_1 = '' + y_columns + ' ~ np.log(' + x + ') + ' + cov1 + ' : ' + cov2 + ''
#     model = smf.mnlogit(eq_1, data=df).fit()
#     # print(model.summary())
#     # print('params:', model.params)
#     # # print('pvalues:', model.pvalues)
#     # print('conf_int:', model.conf_int())
#
#     conf_int_df = model.conf_int().reset_index()
#     pvalues_df = pd.DataFrame(model.pvalues)
#     params_df = pd.DataFrame(model.params)
#
#     # conf_int_df.rename(columns={conf_int_df.columns[0]: "class"}, inplace=True)
#     # print('---------------- model.conf_int ----------------')
#     # for i in conf_int_df["class"].unique():
#     #     q = conf_int_df.loc[(conf_int_df["class"] == str(i))].reset_index(drop=True).drop('class', axis=1).set_index(
#     #         'level_1')
#     #     for k, v in q.iterrows():
#     #         print("class: ", i, " | key: ", k, " | odd lower: ", np.exp(v['lower']), " | odd upper: ", np.exp(v['upper']))
#     #     print()
#
#     #
#     # #
#     # print('---------------- model.params ----------------')
#     # for series_name, series in params_df.items():
#     #     for i, v in series.items():
#     #         print("class: ", series_name, " | key: ", i, " | odd value: ", np.exp(v))
#     #         # print(f'class: {series_name} | key: {i} Impact: {(float(np.exp(v)) - 1) * 100}')
#     #     print()
#     #
#
#
#     #
#     print('---------------- model.pvalues ----------------')
#     for series_name, series in pvalues_df.items():
#         for i, v in series.items():
#             print("class: ", series_name, " | key: ", i, " | p value: ", v)
#         print()
