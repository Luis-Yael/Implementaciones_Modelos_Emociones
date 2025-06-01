# Este script se encarga de procesar el dataset para obtener un Análisis Exploratorio de Datos (EDA)
#Este script es importante para conocer el dataset antes de preprocesar y construir el modelo
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm  # Importar tqdm para ver el progreso

# Descargar recursos de NLTK (si no los tienes instalados)
nltk.download('punkt')

# Definir carpeta de origen y archivo de salida
input_folder = "Clasificacion_Manual_Dataset"
output_file = os.path.join(input_folder, "Dataset_Emociones_Completo.csv")

# Orden de emociones deseado
orden_emociones = [
    "Pregunta_1_6_Felicidad_Etiquetado.csv",
    "Pregunta_2_Tristeza_Etiquetado.csv",
    "Pregunta_3_9_Estres_Etiquetado.csv",
    "Pregunta_4_5_Preocupacion_Etiquetado.csv",
    "Pregunta_7_10_Miedo_Etiquetado.csv",
    "Pregunta_8_Ira_Etiquetado.csv"
]

# Filtrar archivos disponibles en el orden deseado
csv_files = [f for f in orden_emociones if f in os.listdir(input_folder)]

# Lista para almacenar los DataFrames
dfs = []

# Cargar y concatenar los datasets con barra de progreso
for file in tqdm(csv_files, desc="Cargando archivos CSV", unit="archivo"):
    file_path = os.path.join(input_folder, file)
    df = pd.read_csv(file_path, encoding='utf-8')
    dfs.append(df)

# Unir todos los DataFrames en uno solo
if dfs:
    df_complete = pd.concat(dfs, ignore_index=True)
    df_complete.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Dataset unificado guardado en: {output_file}")
else:
    print("No se encontraron archivos CSV en la carpeta especificada.")

# Cargar el dataset consolidado
df = pd.read_csv(output_file, encoding='utf-8')

# Mostrar primeras filas para revisar la estructura
print(df.head())  # Reemplazo de display() por print()

# Ver información general del dataset
df.info()

# Ver cantidad de datos por cada emoción
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Emocion', order=df['Emocion'].value_counts().index, hue='Emocion', palette='viridis', legend=False)
plt.title("Distribución de Clases (Emociones)")
plt.xlabel("Emoción")
plt.ylabel("Cantidad de Ejemplos")
plt.xticks(rotation=45)
plt.show()

# Verificar valores nulos en el dataset
print("Valores nulos en el dataset:\n", df.isnull().sum())

# Longitud de los textos preprocesados con barra de progreso
df['Longitud_Texto'] = [len(str(x).split()) for x in tqdm(df['Texto_Procesado'], desc="Calculando longitudes", unit="texto")]

# Histograma de longitudes de los textos
plt.figure(figsize=(8,5))
sns.histplot(df['Longitud_Texto'], bins=30, kde=True, color='blue')
plt.title("Distribución de Longitudes de los Textos Preprocesados")
plt.xlabel("Cantidad de Palabras")
plt.ylabel("Frecuencia")
plt.show()

# Palabras más frecuentes en los textos preprocesados
all_words = ' '.join(df['Texto_Procesado'].dropna()).split()
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(20)

# Graficar las palabras más comunes
plt.figure(figsize=(10,5))
sns.barplot(x=[word[0] for word in most_common_words], y=[word[1] for word in most_common_words], palette='viridis')
plt.title("Palabras Más Frecuentes en los Textos Preprocesados")
plt.xlabel("Palabra")
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
plt.show()

# Mostrar las palabras más comunes con su frecuencia
print("Palabras más comunes:\n", most_common_words)