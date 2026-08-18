import os
import platform
import subprocess
import torch
import open_clip
from qdrant_client import QdrantClient

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "xlm-roberta-base-ViT-B-32"
pretrained = "laion5b_s13b_b90k"

print(f"Ładowanie modelu {model_name} na urządzeniu: {device}...")
model, _, _ = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained
)
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)

client = QdrantClient(path="./qdrant_db")

COLLECTIONS = {
    "1. Wizualny": "variant_1_visual",
    "2. Opis (Mistral)": "variant_2_text_desc",
    "3. Hybrydowy": "variant_3_hybrid"
}
def encode_query(query_text: str) -> list[float]:
    tokens = tokenizer([query_text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy().tolist()

def open_image(file_path: str):
    if not file_path or not os.path.exists(file_path):
        print(f"\n[BŁĄD] Nie można otworzyć pliku (ścieżka nie istnieje): {file_path}")
        return

    sys_name = platform.system()
    try:
        print(f"\n Otwieranie pliku: {file_path} ...")
        if sys_name == "Windows":
            os.startfile(file_path)
        elif sys_name == "Darwin":  # macOS
            subprocess.run(["open", file_path])
        else:  # Linux
            subprocess.run(["xdg-open", file_path])
    except Exception as e:
        print(f"[BŁĄD przy otwieraniu obrazu]: {e}")
        def main():
    print("\n" + "=" * 70)
    print(" Wpisz prompt i naciśnij Enter.")
    print(" Wpisz 'q', 'exit' lub 'quit', aby zakończyć działanie.")
    print("=" * 70)

    TOP_K = 3 

    while True:
        try:
            query = input("\nWprowadź prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nZakończono działanie.")
            break

        if not query:
            continue

        if query.lower() in ["q", "exit", "quit"]:
            print("Zakończono działanie programu.")
            break