import os
import glob
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Specify the directory where your images are
image_dir = r"D:\Downloads\Training\Kurs8\TestBilder"

# Collect all image files (recursively, case-insensitive)
image_paths = glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True)
image_paths = [f for f in image_paths if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"Found {len(image_paths)} images")
for i, path in enumerate(image_paths[:5], 1):  # Show first few images
    print(f"[{i}] {path}")

# Output file
output_file = os.path.abspath("captions.txt")
print(f"Captions will be saved to: {output_file}")

# Generate captions
with open(output_file, "w", encoding="utf-8") as caption_file:
    for img_path in image_paths:
        try:
            raw_image = Image.open(img_path).convert("RGB")
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            caption_file.write(f"{os.path.basename(img_path)}: {caption}\n")
            print(f"✔ Captioned {os.path.basename(img_path)} -> {caption}")

        except Exception as e:
            print(f"⚠ Error with {img_path}: {e}")
