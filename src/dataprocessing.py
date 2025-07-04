import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def process_data(input_filename="gladiator_data.csv", output_filename="gladiador_data_procesado.csv"):
    # --- TODO ESTE BLOQUE DEBE ESTAR INDENTADO DENTRO DE LA FUNCIÓN ---

    print("--- Iniciando el script data_processing.py ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    print(f"Ruta raíz del proyecto detectada: {project_root}")

    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')

    input_filepath = os.path.join(raw_data_dir, input_filename)
    output_filepath = os.path.join(processed_data_dir, output_filename)

    print(f"Ruta completa esperada para el archivo de entrada: {input_filepath}")
    
    if not os.path.exists(input_filepath):
        print(f"¡ADVERTENCIA CRÍTICA!: El archivo RAW no existe en la ruta esperada: {input_filepath}")
        print("Por favor, verifica que 'gladiator_data.csv' esté en 'tu_proyecto_raiz/data/raw/'.")
        return

    print(f"Cargando datos desde: {input_filepath}")
    try:
        df = pd.read_csv(input_filepath)
        print("Datos cargados exitosamente.")
    except Exception as e:
        print(f"Error al leer el archivo CSV {input_filepath}: {e}")
        return

    df = pd.read_csv("../data/raw/gladiator_data.csv")
    df.info()
    df['Name'].value_counts(normalize=True)
    df['Name'].nunique()
    df = df.drop('Name', axis=1)
    df['WinLossRatio'] = df['Wins'] / (df['Wins'] + df['Losses'])
    df['WinLossRatio'] = df['WinLossRatio'].fillna(0.5)
    columnas_para_heatmap = ['WinLossRatio','Survived']

    df_seleccionado = df[columnas_para_heatmap]

    matriz_correlacion = df_seleccionado.corr()
    plt.figure(figsize=(6, 4)) # Ajusta el tamaño de la figura para una mejor lectura

    sns.heatmap(matriz_correlacion,annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    df = df.drop('WinLossRatio', axis=1)
    df['Origin'].nunique()
    df['Origin'].unique()
    df_encoded = pd.get_dummies(df, columns=['Origin'], drop_first=False, dtype=int)
    columnas_para_heatmap1 = ['Origin_Gaul','Origin_Germania','Origin_Greece','Origin_Numidia','Origin_Rome','Origin_Thrace','Survived']

    df1_seleccionado = df_encoded[columnas_para_heatmap1]

    matriz_correlacion1 = df1_seleccionado.corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(matriz_correlacion1,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar = ['Origin_Gaul', 'Origin_Germania', 'Origin_Greece', 'Origin_Numidia', 'Origin_Rome', 'Origin_Thrace']
    df1 = df_encoded.drop(columnas_a_eliminar, axis=1)
    df1['Height_m'] = df1['Height'] / 100 #Convertir cm a metros
    df1['BMI'] = df1['Weight'] / (df1['Height_m'] ** 2)
    columnas_para_heatmap2 = ['BMI','Height_m','Survived']

    df2_seleccionado = df1[columnas_para_heatmap2]

    matriz_correlacion2 = df2_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion2,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    df1 = df1.drop('BMI', axis=1)
    df1 = df1.drop('Height_m', axis=1)
    # Age * Battle Experience
    df1['Age_x_BattleExperience'] = df1['Age'] * df1['Battle Experience']
    columnas_para_heatmap3 = ['Age_x_BattleExperience','Survived']

    df3_seleccionado = df1[columnas_para_heatmap3]

    matriz_correlacion3 = df3_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion3,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    df1 = df1.drop('Age_x_BattleExperience', axis=1)
    df1['Category'].nunique()
    df1['Category'].unique()
    df1_encoded = pd.get_dummies(df1, columns=['Category'], drop_first=False, dtype=int)
    columnas_para_heatmap4 = ['Category_Hoplomachus','Category_Murmillo','Category_Provocator','Category_Retiarius','Category_Secutor','Category_Thraex','Survived']

    df4_seleccionado = df1_encoded[columnas_para_heatmap4]

    matriz_correlacion4 = df4_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion4,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar1 = ['Category_Hoplomachus', 'Category_Murmillo', 'Category_Provocator', 'Category_Retiarius', 'Category_Secutor', 'Category_Thraex']
    df2 = df1_encoded.drop(columnas_a_eliminar1, axis=1)
    df2['Special Skills'].nunique()
    df2['Special Skills'].unique()
    df2['Special Skills'] = df2['Special Skills'].str.replace('...', '', regex=False)
    df2_encoded = pd.get_dummies(df2, columns=['Special Skills'], drop_first=False, dtype=int)
    columnas_para_heatmap4 = ['Special Skills_Agility','Special Skills_Endurance','Special Skills_Novice','Special Skills_Speed','Special Skills_Strength','Special Skills_Tactics','Survived']

    df4_seleccionado = df2_encoded[columnas_para_heatmap4]

    matriz_correlacion4 = df4_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion4,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar2 = ['Special Skills_Agility', 'Special Skills_Endurance', 'Special Skills_Novice', 'Special Skills_Speed', 'Special Skills_Strength', 'Special Skills_Tactics']
    df3 = df2_encoded.drop(columnas_a_eliminar2, axis=1)
    df3['Weapon of Choice'].nunique()
    df3['Weapon of Choice'].unique()
    df3['Weapon of Choice'].value_counts()
    df3_encoded = pd.get_dummies(df3, columns=['Weapon of Choice'], drop_first=False, dtype=int)
    columnas_para_heatmap5 = ['Weapon of Choice_Dagger','Weapon of Choice_Gladius (Sword)','Weapon of Choice_Net','Weapon of Choice_Sica (Curved Sword)','Weapon of Choice_Spear','Weapon of Choice_Trident','Survived']

    df5_seleccionado = df3_encoded[columnas_para_heatmap5]

    matriz_correlacion5 = df5_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion5,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar2 = ['Weapon of Choice_Dagger', 'Weapon of Choice_Gladius (Sword)', 'Weapon of Choice_Net', 'Weapon of Choice_Sica (Curved Sword)', 'Weapon of Choice_Spear', 'Weapon of Choice_Trident']
    df4 = df3_encoded.drop(columnas_a_eliminar2, axis=1)
    columnas_a_ver = ["Patron Wealth","Equipment Quality","Public Favor","Injury History","Mental Resilience","Diet and Nutrition","Tactical Knowledge","Allegiance Network"]

    print("\nColumnas seleccionadas con sus registros:")
    print(df4[columnas_a_ver])
    df4['Patron Wealth'].nunique()
    df4['Patron Wealth'].unique()
    health_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    training_map = {'Low': 1, 'Medium': 2, 'High': 3}
    patron_wealth_map = {'Low': 1, 'Medium': 2, 'High': 3} # Asumiendo este orden

    df4['Health_Ordinal'] = df4['Health Status'].map(health_map)
    df4['Training_Ordinal'] = df4['Training Intensity'].map(training_map)
    df4['Patron_Wealth_Ordinal'] = df4['Patron Wealth'].map(patron_wealth_map)
    columnas_para_heatmap6 = ['Health_Ordinal','Training_Ordinal','Patron_Wealth_Ordinal','Survived']

    df6_seleccionado = df4[columnas_para_heatmap6]

    matriz_correlacion6 = df6_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion6,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar3 = ['Health_Ordinal', 'Training_Ordinal', 'Patron_Wealth_Ordinal']
    df5 = df4.drop(columnas_a_eliminar3, axis=1)
    df4_encoded = pd.get_dummies(df5, columns=['Patron Wealth'], drop_first=False, dtype=int)
    columnas_para_heatmap7 = ['Patron Wealth_High','Patron Wealth_Low','Patron Wealth_Medium','Survived']
    df7_seleccionado = df4_encoded[columnas_para_heatmap7]
    matriz_correlacion7 = df7_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion7,annot=True, cmap="coolwarm",fmt=".2f",linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar4 = ['Patron Wealth_Low', 'Patron Wealth_Medium']
    df6 = df4_encoded.drop(columnas_a_eliminar4, axis=1)
    df6['Equipment Quality'].nunique()
    df6['Equipment Quality'].unique()
    df5_encoded = pd.get_dummies(df6, columns=['Equipment Quality'], drop_first=False, dtype=int)
    columnas_para_heatmap8 = ['Equipment Quality_Basic','Equipment Quality_Standard','Equipment Quality_Superior','Survived']

    df8_seleccionado = df5_encoded[columnas_para_heatmap8]

    matriz_correlacion8 = df8_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion8,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1,cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar5 = ['Equipment Quality_Basic', 'Equipment Quality_Standard']
    df7 = df5_encoded.drop(columnas_a_eliminar5, axis=1)
    df7['Injury History'].nunique()
    df7['Injury History'].unique()
    df6_encoded = pd.get_dummies(df7, columns=['Injury History'], drop_first=False, dtype=int)
    columnas_para_heatmap9 = ['Injury History_High','Injury History_Low','Survived']

    df9_seleccionado = df6_encoded[columnas_para_heatmap9]

    matriz_correlacion9 = df9_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion9,annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar6 = ['Injury History_High', 'Injury History_Low']
    df8 = df6_encoded.drop(columnas_a_eliminar6, axis=1)
    df8['Diet and Nutrition'].nunique()
    df8['Diet and Nutrition'].unique()
    df7_encoded = pd.get_dummies(df8, columns=['Diet and Nutrition'], drop_first=False, dtype=int)
    columnas_para_heatmap10 = ['Diet and Nutrition_Adequate','Diet and Nutrition_Excellent','Diet and Nutrition_Poor','Survived']

    df10_seleccionado = df7_encoded[columnas_para_heatmap10]

    matriz_correlacion10 = df10_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion10,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar7 = ['Diet and Nutrition_Adequate', 'Diet and Nutrition_Poor']
    df9 = df7_encoded.drop(columnas_a_eliminar7, axis=1)
    df9['Tactical Knowledge'].nunique()
    df9['Tactical Knowledge'].unique()
    df8_encoded = pd.get_dummies(df9, columns=['Tactical Knowledge'], drop_first=False, dtype=int)
    columnas_para_heatmap11 = ['Tactical Knowledge_Advanced','Tactical Knowledge_Basic','Tactical Knowledge_Expert','Tactical Knowledge_Intermediate','Survived']

    df11_seleccionado = df8_encoded[columnas_para_heatmap11]

    matriz_correlacion11 = df11_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion11,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar8 = ['Tactical Knowledge_Advanced', 'Tactical Knowledge_Basic', 'Tactical Knowledge_Expert', 'Tactical Knowledge_Intermediate']
    df10 = df8_encoded.drop(columnas_a_eliminar8, axis=1)
    df10['Allegiance Network'].nunique()
    df10['Allegiance Network'].unique()
    df9_encoded = pd.get_dummies(df10, columns=['Allegiance Network'], drop_first=False, dtype=int)
    columnas_para_heatmap12 = ['Allegiance Network_Moderate','Allegiance Network_Strong','Allegiance Network_Weak','Survived']

    df12_seleccionado = df9_encoded[columnas_para_heatmap12]

    matriz_correlacion12 = df12_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion12,annot=True, cmap="coolwarm",fmt=".2f",linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar9 = ['Allegiance Network_Moderate', 'Allegiance Network_Weak']
    df11 = df9_encoded.drop(columnas_a_eliminar9, axis=1)
    df11['Psychological Profile'].nunique()
    df11['Psychological Profile'].unique()
    df10_encoded = pd.get_dummies(df11, columns=['Psychological Profile'], drop_first=False, dtype=int)
    columnas_para_heatmap13 = ['Psychological Profile_Aggressive','Psychological Profile_Calculative','Psychological Profile_Fearful','Psychological Profile_Stoic','Survived']

    df13_seleccionado = df10_encoded[columnas_para_heatmap13]

    matriz_correlacion13 = df13_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion13,annot=True, cmap="coolwarm",fmt=".2f",linewidths=.5,vmin=-1,vmax=1,cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar10 = ['Psychological Profile_Aggressive', 'Psychological Profile_Calculative', 'Psychological Profile_Fearful', 'Psychological Profile_Stoic']
    df12 = df10_encoded.drop(columnas_a_eliminar10, axis=1)
    df12['Health Status'].nunique()
    df12['Health Status'].unique()
    df11_encoded = pd.get_dummies(df12, columns=['Health Status'], drop_first=False, dtype=int)
    columnas_para_heatmap14 = ['Health Status_Excellent','Health Status_Fair','Health Status_Good','Survived']

    df14_seleccionado = df11_encoded[columnas_para_heatmap14]

    matriz_correlacion14 = df14_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion14,annot=True,cmap="coolwarm",fmt=".2f",linewidths=.5, vmin=-1,vmax=1,cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar11 = ['Health Status_Excellent', 'Health Status_Fair', 'Health Status_Good']
    df13 = df11_encoded.drop(columnas_a_eliminar11, axis=1)
    df13['Personal Motivation'].nunique()
    df13['Personal Motivation'].unique()
    df12_encoded = pd.get_dummies(df13, columns=['Personal Motivation'], drop_first=False, dtype=int)
    columnas_para_heatmap15 = ['Personal Motivation_Freedom','Personal Motivation_Glory','Personal Motivation_Survival','Personal Motivation_Vengeance','Personal Motivation_Wealth','Survived']

    df15_seleccionado = df12_encoded[columnas_para_heatmap15]

    matriz_correlacion15 = df15_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion15,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar12 = ['Personal Motivation_Freedom', 'Personal Motivation_Glory', 'Personal Motivation_Survival', 'Personal Motivation_Vengeance', 'Personal Motivation_Wealth']
    df14 = df12_encoded.drop(columnas_a_eliminar12, axis=1)
    df14['Previous Occupation'].nunique()
    df14['Previous Occupation'].unique()
    df13_encoded = pd.get_dummies(df14, columns=['Previous Occupation'], drop_first=False, dtype=int)
    columnas_para_heatmap16 = ['Previous Occupation_Criminal','Previous Occupation_Entertainer','Previous Occupation_Laborer','Previous Occupation_Soldier','Previous Occupation_Unemployed','Survived']

    df16_seleccionado = df13_encoded[columnas_para_heatmap16]

    matriz_correlacion16 = df16_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion16,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar13 = ['Previous Occupation_Criminal', 'Previous Occupation_Entertainer', 'Previous Occupation_Laborer', 'Previous Occupation_Soldier', 'Previous Occupation_Unemployed']
    df15 = df13_encoded.drop(columnas_a_eliminar13, axis=1)
    df15['Training Intensity'].nunique()
    df14_encoded = pd.get_dummies(df15, columns=['Training Intensity'], drop_first=False, dtype=int)
    columnas_para_heatmap17 = ['Training Intensity_High','Training Intensity_Low','Training Intensity_Medium','Survived']

    df17_seleccionado = df14_encoded[columnas_para_heatmap17]

    matriz_correlacion17 = df17_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion17,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'})
    columnas_a_eliminar14 = ['Training Intensity_High', 'Training Intensity_Low', 'Training Intensity_Medium']
    df16 = df14_encoded.drop(columnas_a_eliminar14, axis=1)
    df16['Battle Strategy'].nunique()
    df16['Battle Strategy'].unique()
    df15_encoded = pd.get_dummies(df16, columns=['Battle Strategy'], drop_first=False, dtype=int)
    columnas_para_heatmap18 = ['Battle Strategy_Aggressive','Battle Strategy_Balanced','Battle Strategy_Defensive','Survived']

    df18_seleccionado = df15_encoded[columnas_para_heatmap18]

    matriz_correlacion18 = df18_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion18,annot=True,cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar15 = ['Battle Strategy_Aggressive', 'Battle Strategy_Balanced', 'Battle Strategy_Defensive']
    df17 = df15_encoded.drop(columnas_a_eliminar15, axis=1)
    df17['Social Standing'].nunique()
    df17['Social Standing'].unique()
    social_standing_map = {'Low': 1, 'Medium': 2, 'High': 3} # Asume este orden
    df17['Social_Standing_Ordinal'] = df17['Social Standing'].map(social_standing_map)
    #Public Favor * Social Standing
    # Y usa la versión ordinal de Social Standing
    df17['PublicFavor_x_Standing'] = df17['Public Favor'] * df17['Social_Standing_Ordinal']
    df17.head()
    columnas_para_heatmap19 = ['PublicFavor_x_Standing','Social_Standing_Ordinal','Survived']

    df19_seleccionado = df17[columnas_para_heatmap19]

    matriz_correlacion19 = df19_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion19,annot=True,cmap="coolwarm",fmt=".2f", linewidths=.5,vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    df16_encoded = pd.get_dummies(df17, columns=['Social Standing'], drop_first=False, dtype=int)
    df16_encoded.head()
    columnas_para_heatmap20 = ['Social Standing_High','Social Standing_Low','Social Standing_Medium','Survived']

    df20_seleccionado = df16_encoded[columnas_para_heatmap20]

    matriz_correlacion20 = df20_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion20,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar16 = ['Social Standing_Low', 'Social Standing_Medium']
    df18 = df16_encoded.drop(columnas_a_eliminar16, axis=1)
    df18['Crowd Appeal Techniques'].nunique()
    df18['Crowd Appeal Techniques'].unique()
    df17_encoded = pd.get_dummies(df18, columns=['Crowd Appeal Techniques'], drop_first=False, dtype=int)
    columnas_para_heatmap21 = ['Crowd Appeal Techniques_Charismatic','Crowd Appeal Techniques_Flamboyant','Crowd Appeal Techniques_Humble','Crowd Appeal Techniques_Intimidating','Survived']

    df21_seleccionado = df17_encoded[columnas_para_heatmap21]

    matriz_correlacion21 = df21_seleccionado.corr()
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(matriz_correlacion21,annot=True, cmap="coolwarm",fmt=".2f", linewidths=.5, vmin=-1,vmax=1,cbar_kws={'label': 'Coeficiente de Correlación'} )
    columnas_a_eliminar17 = ['Crowd Appeal Techniques_Charismatic', 'Crowd Appeal Techniques_Humble', 'Crowd Appeal Techniques_Intimidating']
    df19 = df17_encoded.drop(columnas_a_eliminar17, axis=1)
    plt.figure(figsize=(14, 14))
    sns.heatmap(df19.corr(numeric_only= True), annot= True, cmap= "coolwarm", vmin=-1)
    columnas_a_eliminar18 = ['Age', 'PublicFavor_x_Standing', 'Social_Standing_Ordinal', 'Birth Year', 'Height', 'Weight', 'Losses', 'Mental Resilience', 'Battle Experience', 'Patron Wealth_High', 'Equipment Quality_Superior', 'Diet and Nutrition_Excellent', 'Social Standing_High', 'Crowd Appeal Techniques_Flamboyant']
    df20 = df19.drop(columnas_a_eliminar18, axis=1)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df20.corr(numeric_only= True), annot= True, cmap= "coolwarm", vmin=-1)
    plt.figure(figsize=(1, 1))
    sns.pairplot(df20, hue='Survived')
    plt.close('all') 

    df20.to_csv('../data/processed/gladiador_data_procesado.csv', index=False)

    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
        print(f"Directorio de salida creado: {processed_data_dir}")

    print(f"Guardando datos procesados en: {output_filepath}")
    
    if 'df20' in locals():
        df20.to_csv(output_filepath, index=False)
        print(f"Datos procesados (df20) guardados exitosamente en: {output_filepath}")
    elif 'df' in locals():
        df.to_csv(output_filepath, index=False)
        print(f"Datos procesados (df) guardados exitosamente en: {output_filepath}")
    else:
        print("¡ADVERTENCIA!: No se encontró un DataFrame 'df' o 'df20' para guardar.")

    print("--- Script data_processing.py finalizado ---")

# Esta parte DEBE ESTAR SIN INDENTACIÓN, en el nivel superior del script
if __name__ == "__main__":
    process_data()