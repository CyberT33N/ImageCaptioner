import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

directory_path = "/home/t33n/Documents/pics me ai"

# Überprüfe, ob die GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("GPU wird verwendet:", torch.cuda.get_device_name(0))
else:
    print("GPU nicht verfügbar, CPU wird verwendet.")

# Kosmos-2 Model und Processor laden
kosmos_path = "/home/t33n/Projects/ai/resources/transformers/kosmos-2-patch14-224"
model = AutoModelForVision2Seq.from_pretrained(kosmos_path).to(device)  # Verschiebe das Modell auf die GPU
processor = AutoProcessor.from_pretrained(kosmos_path)

def generate_caption_for_image(image_path):
    # Bild öffnen
    image = Image.open(image_path)
    
    # Eingabeprompt vorbereiten
    prompt = "<grounding>An image of"
    
    # Bild und Text durch Processor verarbeiten
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Verschiebe Eingaben auf die GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Bildbeschreibung generieren
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )
    
    # Text dekodieren und generieren
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text, _ = processor.post_process_generation(generated_text)
    
    return processed_text

def process_images_in_directory(directory_path):
    # Alle Bilddateien im Verzeichnis durchsuchen
    for filename in os.listdir(directory_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory_path, filename)
            caption = generate_caption_for_image(image_path)
            
            # .txt Datei mit Bildnamen erstellen und Caption speichern
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(directory_path, txt_filename)

            with open(txt_path, "w") as txt_file:
                txt_file.write(caption)
                
            print(f"Caption für '{filename}' erstellt und in '{txt_filename}' gespeichert.")

# Beispiel für die Nutzung: Verzeichnispfad angeben
process_images_in_directory(directory_path)
