import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image, ImageTk

# Fungsi untuk memuat model
def load_trained_model():
    return load_model('D:\Projek\imageclassifier.h5')

# Fungsi untuk memuat gambar
def load_image():
    global img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((256, 256))
        img = ImageTk.PhotoImage(img)
        input_img.configure(image=img)
        input_img.image = img
        result_label.config(text="")

# Fungsi untuk memproses gambar dan melakukan prediksi
def process_and_predict():
    global img_path
    if img_path:
        # Resize gambar
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        # Prediksi kelas
        result = model.predict(np.expand_dims(img / 255, 0))

        if result > 0.5:
            result_text = "Gambar yang diprediksi adalah Sedih"
        else:
            result_text = "Gambar yang diprediksi adalah Bahagia"

        result_label.config(text=result_text)

# Buat jendela utama
root = tk.Tk()
root.title("Klasifikasi Gambar")

# Muat model
model = load_trained_model()

# Kotak gambar input
input_img = tk.Label(root)
input_img.pack(side="left", padx=10, pady=10)

# Kotak hasil prediksi
result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
result_label.pack(side="top", pady=10)

# Tombol "Mulai Prediksi"
predict_button = tk.Button(root, text="Mulai Prediksi", command=process_and_predict, bg="#add8e6", fg="black")
predict_button.pack(side="bottom", pady=10, padx=10)

# Tombol "Masukkan Gambar"
load_button = tk.Button(root, text="Masukkan Gambar", command=load_image, bg="#add8e6", fg="black")
load_button.pack(side="bottom", pady=10, padx=10)



# Jalankan aplikasi
root.mainloop()
