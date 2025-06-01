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

# Definir el optimizador y la función de pérdida
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
loss_function = nn.CrossEntropyLoss()

EPOCHS = 5  # Número de épocas de entrenamiento

# Almacenar métricas de entrenamiento y prueba
history = {"train_loss": [], "test_loss": [], "train_accuracy": [], "test_accuracy": [], "precision": [], "recall": [], "f1_score": [], "roc_auc": []}

# Función de entrenamiento
def train_epoch(model, train_dataloader, optimizer, loss_function):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(train_dataloader, desc="Entrenando", unit="batch"):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_function(outputs.logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    avg_loss = total_loss / len(train_dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# Función de evaluación con métricas adicionales
def evaluate_epoch(model, test_dataloader, loss_function):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluando", unit="batch"):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_function(outputs.logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    avg_loss = total_loss / len(test_dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    
    print("\n===== Métricas de Evaluación =====")
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print(f"Precisión (Precision): {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Área bajo la curva ROC-AUC: {roc_auc:.4f}\n")
    
    # Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[reverse_labels[i] for i in range(NUM_LABELS)], yticklabels=[reverse_labels[i] for i in range(NUM_LABELS)])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.show()
    
    return avg_loss, accuracy

# Entrenamiento del modelo con actualización de pesos
for epoch in range(EPOCHS):
    print(f"\n===== Época {epoch+1}/{EPOCHS} =====")
    train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, loss_function)
    test_loss, test_acc = evaluate_epoch(model, test_dataloader, loss_function)
    
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["train_accuracy"].append(train_acc)
    history["test_accuracy"].append(test_acc)

    print(f"Pérdida Entrenamiento: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
    print(f"Pérdida Prueba: {test_loss:.4f} | Accuracy: {test_acc:.4f}")