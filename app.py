import streamlit as st
from PIL import Image
import os
from model import train_model
from utils import load_model, predict_image, load_label_map
import matplotlib.pyplot as plt
import cv2

st.title("🪴 Deteksi Penyakit Daun - MobileNetV2 - Kecerdasan Tiruan")

menu = st.sidebar.radio("Navigasi", ["Train Model", "Prediksi Gambar"])

if menu == "Train Model":
    st.header("🔧 Training Model")
    dataset_path = st.text_input("Masukkan path folder dataset:", "dataset/")
    epochs = st.slider("Jumlah Epochs", 1, 100, 10)

    if st.button("Mulai Training"):
        if os.path.exists(dataset_path):
            with st.spinner("Melatih model menggunakan MobileNetV2..."):
                model, class_indices, history = train_model(dataset_path, epochs)
            st.success("✅ Model berhasil dilatih dan disimpan!")

            st.subheader("📁 Label yang Terdeteksi:")
            st.write(class_indices)

            st.subheader("📊 Akurasi & Loss Selama Training")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history['accuracy'], label='Train Accuracy')
            ax[0].plot(history['val_accuracy'], label='Val Accuracy')
            ax[0].legend(); ax[0].set_title('Accuracy')

            ax[1].plot(history['loss'], label='Train Loss')
            ax[1].plot(history['val_loss'], label='Val Loss')
            ax[1].legend(); ax[1].set_title('Loss')

            st.pyplot(fig)
        else:
            st.error("❌ Path folder dataset tidak ditemukan.")

elif menu == "Prediksi Gambar":
    st.header("🔍 Prediksi Penyakit Daun")

    if os.path.exists("saved_model/leaf_model.h5"):
        model = load_model()
        class_indices = load_label_map()
    else:
        st.warning("Model belum ada. Silakan latih terlebih dahulu.")
        st.stop()

    option = st.radio("Pilih metode input:", ["Upload Gambar", "Gunakan Kamera"])

    if option == "Upload Gambar":
        uploaded_file = st.file_uploader("Upload gambar daun", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption='Gambar yang Diupload', use_container_width=True)
            label, conf = predict_image(model, img, class_indices)
            st.success(f"Hasil: **{label}** ({conf*100:.2f}%)")

    elif option == "Gunakan Kamera":
        st.info("Tekan tombol di bawah untuk ambil gambar dari kamera.")
        if st.button("📸 Ambil Gambar"):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                os.makedirs("images", exist_ok=True)
                image_path = os.path.join("images", "capture.jpg")
                cv2.imwrite(image_path, frame)
                st.image(image_path, caption="Gambar dari Kamera", use_container_width=True)
                img = Image.open(image_path)
                label, conf = predict_image(model, img, class_indices)
                st.success(f"Hasil: **{label}** ({conf*100:.2f}%)")
            else:
                st.error("Tidak bisa akses kamera.")
