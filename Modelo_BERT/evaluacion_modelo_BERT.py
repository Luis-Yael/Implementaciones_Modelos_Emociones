import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import numpy as np

# Cargar el dataset consolidado para obtener las etiquetas de emociones
file_path = "Clasificacion_Manual_Dataset/Dataset_Emociones_Completo.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# Codificar etiquetas de emociones a valores numéricos si no están definidas
emocion_labels = {label: idx for idx, label in enumerate(df["Emocion"].unique())}
reverse_labels = {v: k for k, v in emocion_labels.items()}  # Diccionario inverso para convertir números a etiquetas

# Verificar si hay GPU disponible para acelerar el procesamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Cargar el modelo BERT preentrenado con una capa de clasificación ajustada
NUM_LABELS = len(emocion_labels)  # Número de clases en la clasificación
MODEL_NAME = "bert-base-multilingual-cased"

print("Cargando modelo BERT preentrenado...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(device)  # Enviar el modelo a GPU si está disponible

# Cargar el tokenizador de BERT
print("Cargando tokenizador...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Cargar el modelo entrenado previamente
MODEL_PATH = "Clasificacion_Manual_Dataset/BERT_Emociones_Model.pth"
print(f"Cargando modelo entrenado desde {MODEL_PATH}...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Poner el modelo en modo evaluación
print("Modelo cargado y listo para predicciones.")

# Cargar los datasets preprocesados
train_dataset_path = "Clasificacion_Manual_Dataset/Train_Dataset.csv"
test_dataset_path = "Clasificacion_Manual_Dataset/Test_Dataset.csv"
train_df = pd.read_csv(train_dataset_path, encoding='utf-8')
test_df = pd.read_csv(test_dataset_path, encoding='utf-8')

# Tokenizar los textos y convertirlos en tensores
def convert_to_tensor_dataset(df):
    encodings = tokenizer(
        df['Texto_Procesado'].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    labels = torch.tensor(df['Emocion'].values)
    return TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)

train_data = convert_to_tensor_dataset(train_df)
test_data = convert_to_tensor_dataset(test_df)

# Crear DataLoaders
BATCH_SIZE = 16
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Función para predecir la emoción de un texto
def predecir_emocion(texto):
    """
    Tokeniza el texto ingresado, lo pasa por el modelo y devuelve la emoción predicha.
    """
    inputs = tokenizer(texto, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    
    with torch.no_grad():  # Desactiva gradientes ya que no estamos entrenando
        outputs = model(**inputs)
    
    pred_label = torch.argmax(outputs.logits, dim=1).item()
    emocion_predicha = reverse_labels[pred_label]  # Convertir índice numérico a etiqueta de emoción
    return emocion_predicha

# Evaluación del modelo en el conjunto de prueba
def evaluar_modelo(test_dataloader):
    """Evalúa el modelo y calcula métricas de desempeño."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluando", unit="batch"):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calcular métricas de clasificación
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    
    print("\n===== Métricas de Evaluación =====")
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print(f"Precisión (Precision - Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    print(f"Área bajo la curva ROC-AUC: {auc_score:.4f}\n")
    
    # Matriz de Confusión Normalizada
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=[reverse_labels[i] for i in range(NUM_LABELS)], yticklabels=[reverse_labels[i] for i in range(NUM_LABELS)])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión Normalizada")
    plt.show()

# Llamar a la evaluación del modelo
evaluar_modelo(test_dataloader)

# Interfaz para ingresar textos y obtener predicciones
print("\nIngrese una frase para evaluar su emoción. Escriba 'salir' para terminar.\n")
while True:
    texto_usuario = input("Texto: ")
    if texto_usuario.lower() == "salir":
        print("Saliendo del programa...")
        break
    
    emocion = predecir_emocion(texto_usuario)
    print(f"La emoción predicha es: {emocion}\n")