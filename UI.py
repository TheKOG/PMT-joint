import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import torch
import clip
from torchvision import transforms

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry("600x500")
        
        self.label = tk.Label(root, text="Upload or Drag an Image Here")
        self.label.pack(pady=20)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.canvas = tk.Canvas(root, width=400, height=300)
        self.canvas.pack()

        self.class_names = ["Dog", "Elephant", "Giraffe", "Guitar", "Horse", "House", "Person"]
        self.clip_model, _ = clip.load("ViT-B/16", "cuda")
        self.classifier = self.clip_model.visual
        self.classifier.load_state_dict(torch.load("./exp0/checkpoints/best-PACS.pth"))
        self.classifier.eval()
        self.text_features = self.get_text_features(self.clip_model, "a photo of a {}.", self.class_names, "cuda")

        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind('<<Drop>>', self.drop_image)

    def get_text_features(self, clip_model, template, class_names, device):
        with torch.no_grad():
            texts = torch.cat(
                [clip.tokenize(template.format(c.replace("_", " ")))
                for c in class_names]).to(device)
            text_features = clip_model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)
            self.process_image(file_path)

    def drop_image(self, event):
        file_path = event.data.strip('{}')
        self.display_image(file_path)
        self.process_image(file_path)

    def display_image(self, file_path):
        image_origin = Image.open(file_path)
        self.canvas.delete("all")
        image_tk = ImageTk.PhotoImage(image_origin)
        self.canvas.create_image(200, 150, image=image_tk)
        self.canvas.image = image_tk

    def process_image(self, file_path):
        image_origin = Image.open(file_path)
        image = self.transform_image(image_origin)
        image_features = self.classify_image(image)
        predicted_class = self.get_predicted_class(image_features)
        self.display_prediction(predicted_class)

    def transform_image(self, image_origin):
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        image = transform(image_origin).unsqueeze(0).to("cuda")
        return image

    def classify_image(self, image):
        with torch.no_grad():
            image_features = self.classifier(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_predicted_class(self, image_features):
        output_similarity = image_features @ self.text_features.T
        predicted_index = output_similarity.argmax()
        return self.class_names[predicted_index]

    def display_prediction(self, predicted_class):
        self.canvas.create_text(200, 275, text=f"Predicted Class: {predicted_class}", font=('Helvetica', 20), fill="red")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
