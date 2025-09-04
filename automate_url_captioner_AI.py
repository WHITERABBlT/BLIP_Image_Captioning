from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image, ImageFile
import requests
from bs4 import BeautifulSoup
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

url = "https://en.wikipedia.org/wiki/IBM"
headers = {"User-Agent": "Mozilla/5.0"}
soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')

img_elements = soup.find_all('img')

with open("captions.txt", "w") as caption_file:
    for img_element in img_elements:
        img_url = img_element.get('src')
        if not img_url:
            continue

        # Skip SVGs and tiny icons
        if img_url.endswith(".svg") or img_url.endswith(".svg.png") or "1x1" in img_url:
            continue

        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not img_url.startswith('http'):
            continue

        try:
            img_data = requests.get(img_url, headers=headers).content
            raw_image = Image.open(BytesIO(img_data)).convert("RGB")
            if raw_image.size[0] * raw_image.size[1] < 400:
                continue

            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.tokenizer.decode(out[0], skip_special_tokens=True)

            caption_file.write(f"{img_url}: {caption}\n")
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
