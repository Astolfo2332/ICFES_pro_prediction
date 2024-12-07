import pandas as pd
from transformers import pipeline
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
#Cargamos un modelo de clasificación de texto
#La ventaja de usar pipeline es que no se requiere establecer el tokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = pipeline("zero-shot-classification",
                        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=device)

    #Definimos las categorías

    categories = ["ciencias", "ingeniería", "arte", "salud", "economía", "derecho", "educación", "humanidades", "agropecuaria", "tecnología"]
    #Definimos un dataset para mejorar la eficiencia de la gpu
    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            return self.texts[idx]

    lf_epoch = None

    if os.path.exists("data/program_test.csv"):
        df_load = pd.read_csv("data/program_test.csv")
        lf_epoch = df_load["epoch"].max()

    df = pd.read_csv("data/test.csv")

    batch_size = 1024
    if lf_epoch is None:
        pgrm_colum = TextDataset(df["ESTU_PRGM_ACADEMICO"].tolist())
        lf_epoch = -1
    else:
        pgrm_colum = TextDataset(df["ESTU_PRGM_ACADEMICO"].tolist()[(lf_epoch + 1) * batch_size:])

    #Creamos un dataloader
    pgrm_loader = DataLoader(pgrm_colum, batch_size=batch_size, shuffle=False)

    #Definimos una función que clasifica el texto
    def classify_text(text):
        results = classifier(text, categories)
        return [result["labels"][0] for result in results]

    def save_text(epoch, text, file_path= "data/program_test.csv"):
        prog_df = pd.DataFrame({"epoch": [epoch] * len(text), "class_labels": text})
        if os.path.exists(file_path):
            prog_df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            prog_df.to_csv(file_path, index=False)

    class_labels = []

    #Iteramos en el dataset creado guardando los resultados en una lista

    with torch.inference_mode():
        for epoch, bath_texts in enumerate(tqdm(pgrm_loader, desc="Clasificando textos")):
            labels = classify_text(bath_texts)
            class_labels.extend(labels)
            if lf_epoch == -1:
                save_text(epoch, labels)
            else:
                save_text(epoch + lf_epoch + 1, labels)

if __name__ == "__main__":
    main()
