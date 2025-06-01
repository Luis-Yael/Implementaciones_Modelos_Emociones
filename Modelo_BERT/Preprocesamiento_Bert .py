#Este script sirve para preprocesar el dataset, se divide en 80% de training y 20% de test.
#Se carga el tokenizador con BERT
# Se convierten los datos tokenizados a tensores de PyTorch
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
import os

# Cargar el dataset consolidado desde la carpeta donde se guardó previamente
file_path = "Clasificacion_Manual_Dataset/Dataset_Emociones_Completo.csv"
print(f"Cargando dataset desde: {file_path}")
df = pd.read_csv(file_path, encoding='utf-8')
print(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")

# Codificar etiquetas de emociones a valores numéricos para que el modelo pueda interpretarlas
print("Codificando etiquetas de emociones...")
emocion_labels = {label: idx for idx, label in enumerate(df["Emocion"].unique())}  # Asigna un número a cada emoción única
df["Emocion"] = df["Emocion"].map(emocion_labels)  # Reemplaza los nombres de emociones por su número correspondiente
print(f"Etiquetas codificadas: {emocion_labels}")

# Dividir el dataset en conjunto de entrenamiento (80%) y prueba (20%)
print("Dividiendo el dataset en entrenamiento y prueba...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Texto_Procesado"].tolist(),  # Se toma la columna con los textos preprocesados
    df["Emocion"].tolist(),  # Se toma la columna con las emociones codificadas
    test_size=0.2,  # El 20% de los datos se usará para la evaluación
    random_state=42,  # Se fija una semilla para reproducibilidad
    stratify=df["Emocion"]  # Se asegura que la distribución de clases sea balanceada en ambos conjuntos
)
print(f"Conjunto de entrenamiento: {len(train_texts)} muestras, Conjunto de prueba: {len(test_texts)} muestras")

# Guardar los datasets en archivos CSV
train_df = pd.DataFrame({"Texto_Procesado": train_texts, "Emocion": train_labels})
test_df = pd.DataFrame({"Texto_Procesado": test_texts, "Emocion": test_labels})

train_path = "Clasificacion_Manual_Dataset/Train_Dataset.csv"
test_path = "Clasificacion_Manual_Dataset/Test_Dataset.csv"

train_df.to_csv(train_path, index=False, encoding='utf-8')
test_df.to_csv(test_path, index=False, encoding='utf-8')

print(f"Train dataset guardado en: {train_path}")
print(f"Test dataset guardado en: {test_path}")

# Cargar el tokenizador de BERT
MODEL_NAME = "bert-base-multilingual-cased"  # Se puede cambiar dependiendo del idioma del dataset
print(f"Cargando tokenizador: {MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Tokenizar los textos para convertirlos en secuencias numéricas que BERT pueda interpretar
print("Tokenizando textos...")
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=128  # Se establece un máximo de 128 tokens por texto
)
test_encodings = tokenizer(
    test_texts, truncation=True, padding=True, max_length=128  # Se realiza el mismo procesamiento para el conjunto de prueba
)
print("Tokenización completada.")

# Convertir los datos tokenizados a tensores de PyTorch
print("Convirtiendo datos a tensores de PyTorch...")
train_inputs = torch.tensor(train_encodings["input_ids"])  # IDs de los tokens
train_masks = torch.tensor(train_encodings["attention_mask"])  # Máscara de atención (1 para tokens válidos, 0 para padding)
train_labels = torch.tensor(train_labels)  # Etiquetas de las emociones

test_inputs = torch.tensor(test_encodings["input_ids"])  # IDs de los tokens en el conjunto de prueba
test_masks = torch.tensor(test_encodings["attention_mask"])  # Máscara de atención para el conjunto de prueba
test_labels = torch.tensor(test_labels)  # Etiquetas del conjunto de prueba
print("Conversión a tensores completada.")

# Crear conjuntos de datos para PyTorch con los tensores generados
print("Creando TensorDatasets...")
train_data = TensorDataset(train_inputs, train_masks, train_labels)
test_data = TensorDataset(test_inputs, test_masks, test_labels)
print("TensorDatasets creados exitosamente.")

# Definir el tamaño de lote (batch size) para el entrenamiento y la evaluación
BATCH_SIZE = 16  # Ajustable según la capacidad de memoria de la GPU
print(f"Configurando DataLoaders con batch size de {BATCH_SIZE}...")

# Crear DataLoaders para manejar los datos en lotes y facilitar el entrenamiento
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)  # Se barajan los datos en cada época
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)  # No se barajan los datos de prueba
print("DataLoaders creados correctamente.")

print("Preprocesamiento completado. Datos listos para entrenamiento.")