import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Cargar el dataset consolidado para obtener las etiquetas de emociones
file_path = "Clasificacion_Manual_Dataset/Dataset_Emociones_Completo.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# Codificar etiquetas de emociones a valores numéricos si no están definidas
emocion_labels = {label: idx for idx, label in enumerate(df["Emocion"].unique())}

# Verificar si hay GPU disponible para acelerar el entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Cargar el modelo BERT preentrenado con una capa de clasificación ajustada
NUM_LABELS = len(emocion_labels)  # Número de clases en la clasificación
MODEL_NAME = "bert-base-multilingual-cased"

print("Cargando modelo BERT preentrenado...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(device)  # Enviar el modelo a GPU si está disponible

# Cargar el tokenizador de BERT
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Definir la función de pérdida y el optimizador
loss_function = nn.CrossEntropyLoss()  # Función de pérdida adecuada para clasificación multiclase
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # Optimizador adaptado a Transformers

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

# Número de épocas para el entrenamiento (ajustable según rendimiento)
EPOCHS = 5

# Listas para almacenar métricas de entrenamiento y evaluación
epochs_list = []
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Función de entrenamiento
def train_model(model, train_dataloader):
    model.train()  # Poner el modelo en modo entrenamiento
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for batch in tqdm(train_dataloader, desc="Entrenando", unit="batch", leave=False):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]  # Mover datos a GPU si está disponible
        optimizer.zero_grad()  # Resetear gradientes
        
        # Realizar la predicción
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_function(outputs.logits, labels)  # Calcular la pérdida
        total_loss += loss.item()
        
        # Backpropagation y optimización
        loss.backward()
        optimizer.step()
        
        # Calcular exactitud de predicciones
        preds = torch.argmax(outputs.logits, dim=1)  # Obtener la clase con mayor probabilidad
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
    
    avg_loss = total_loss / len(train_dataloader)  # Promedio de pérdida por batch
    accuracy = correct_predictions / total_samples  # Exactitud del modelo
    return avg_loss, accuracy

# Función de evaluación (sin actualización de gradientes)
def evaluate_model(model, test_dataloader):
    model.eval()  # Poner el modelo en modo evaluación
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  # Desactivar el cálculo de gradientes para ahorrar memoria y acelerar la evaluación
        for batch in tqdm(test_dataloader, desc="Evaluando", unit="batch", leave=False):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_function(outputs.logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)  # Obtener la clase predicha
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(test_dataloader)  # Promedio de pérdida en la evaluación
    accuracy = correct_predictions / total_samples  # Exactitud en el conjunto de prueba
    return avg_loss, accuracy

# Entrenamiento del modelo por el número de épocas definido
for epoch in range(EPOCHS):
    print(f"\n===== Época {epoch+1} de {EPOCHS} =====")
    train_loss, train_acc = train_model(model, train_dataloader)
    test_loss, test_acc = evaluate_model(model, test_dataloader)
    
    print(f"Pérdida en entrenamiento: {train_loss:.4f}, Exactitud: {train_acc:.4f}")
    print(f"Pérdida en prueba: {test_loss:.4f}, Exactitud: {test_acc:.4f}")
    
    # Guardar métricas
    epochs_list.append(epoch+1)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Guardar el modelo entrenado para futuras predicciones
MODEL_PATH = "Clasificacion_Manual_Dataset/BERT_Emociones_Model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Modelo guardado en {MODEL_PATH}")
