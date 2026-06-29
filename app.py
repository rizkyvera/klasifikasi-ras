import streamlit as st
from PIL import Image
import re
_PIL_IMAGE_OPEN = Image.open
from ultralytics import YOLO
import os
import zipfile

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.title("🐱 Klasifikasi Kucing vs Anjing 🐶")
st.write("Unggah foto hanya dari folder `dataset` untuk diprediksi menggunakan YOLO11!")

# 1. Load Model (Pastikan file 'best.pt' berada di folder 'model')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'best_breed.pt')
DATA_DIR = r'D:\Semester 8\CatsAndDogs\dataset_ras_validation'
os.makedirs(DATA_DIR, exist_ok=True)


def _parse_species_and_breed(label: str):
    """Heuristic parser: from a class label try to extract species and breed.

    Returns (species_display, breed_display) where species_display is one of
    'Kucing 🐱', 'Anjing 🐶', or a title-cased label; breed_display is a title-cased
    breed string or None when not detectable.
    """
    if label is None:
        return None, None
    lbl = str(label).strip()
    low = lbl.lower().replace('_', ' ')

    # List of known cat breeds
    cat_breeds = {
        'abyssinian', 'bengal', 'birman', 'bombay', 'british shorthair',
        'egyptian mau', 'maine coon', 'persian', 'ragdoll', 'russian blue',
        'siamese', 'sphynx', 'mainecoon', 'exotic', 'balinese', 'scottish fold', 'norwegian forest'
    }

    breed = lbl.replace('_', ' ').title()
    species = None

    if low in cat_breeds or any(cb in low for cb in cat_breeds):
        species = 'Kucing 🐱'
    elif 'cat' in low or 'kucing' in low:
        species = 'Kucing 🐱'
    else:
        # Default to dog if not identified as cat
        species = 'Anjing 🐶'

    return species, breed

@st.cache_resource
def load_custom_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

@st.cache_data
def list_dataset_images():
    image_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                abs_path = os.path.join(root, file)
                image_files.append(abs_path)
    return sorted(image_files)


model = load_custom_model(model_path)

if model is None:
    st.error(f"Model '{model_path}' tidak ditemukan! Pastikan file .pt sudah diletakkan di folder 'model' di samping skrip ini.")
else:
    st.subheader("Dataset Testing")

    st.info("Pilih gambar internal dari folder `dataset`. Tidak perlu mengunggah gambar dari perangkat Anda.")

    tabs = st.tabs(["Internal Dataset", "Upload Eksternal"])
    dataset_images = list_dataset_images()

    with tabs[0]:
        st.subheader("Pilih Gambar Dataset")
        if len(dataset_images) == 0:
            st.warning("Folder `dataset` kosong. Masukkan gambar ke folder `dataset` terlebih dahulu.")
        else:
            selected_image = st.selectbox(
                "Pilih gambar dari folder dataset", 
                dataset_images, 
                format_func=lambda x: os.path.splitext(os.path.basename(x))[0].replace('_', ' ')
            )
            if selected_image:
                image_path = selected_image
                # Display without using PIL.Image.open to avoid ultralytics patch side-effects
                try:
                    with open(image_path, 'rb') as f:
                        img_bytes = f.read()
                    st.image(img_bytes, caption=os.path.splitext(os.path.basename(image_path))[0].replace('_', ' '), use_container_width=True)
                except Exception:
                    st.error("Gagal menampilkan gambar dari path. Periksa file dan izin.")

                if st.button("Prediksi Sekarang", key="predict_internal"):
                    with st.spinner("Sedang menganalisis gambar..."):
                        # Pass file path to model to let it load the image internally
                        results = model(image_path)
                        result = results[0]
                        # Get predicted label and confidence robustly
                        try:
                            if hasattr(result, 'probs') and result.probs is not None:
                                top_class_idx = int(result.probs.top1)
                                top_class_name = result.names[top_class_idx]
                                confidence = float(result.probs.top1conf.item() * 100)
                            else:
                                cls_list = result.boxes.cls.tolist() if hasattr(result.boxes, 'cls') else []
                                conf_list = result.boxes.conf.tolist() if hasattr(result.boxes, 'conf') else []
                                if len(cls_list) > 0:
                                    top_class_name = result.names[int(cls_list[0])]
                                    confidence = float(conf_list[0] * 100) if len(conf_list) > 0 else 0.0
                                else:
                                    top_class_name = result.names[0] if hasattr(result, 'names') else 'Unknown'
                                    confidence = 0.0
                        except Exception:
                            top_class_name = 'Unknown'
                            confidence = 0.0

                        species, breed = _parse_species_and_breed(top_class_name)
                        st.success("Analisis Selesai!")
                        st.metric(label="Hasil Prediksi", value=species)
                        if breed:
                            st.markdown(f"### Ras: {breed}")
                        st.info(f"Tingkat Kepercayaan (Confidence): {confidence:.2f}%")

    with tabs[1]:
        st.subheader("Unggah Gambar Eksternal")
        uploaded_file = st.file_uploader("Pilih file gambar dari perangkat Anda", type=["jpg", "jpeg", "png"], key="upload_external")
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="Gambar Eksternal", use_container_width=True)
            except Exception as e:
                st.error(f"Gagal memuat atau menampilkan gambar unggahan: {e}")

            # Run model once to gather candidate labels and default prediction
            with st.spinner("Mengambil prediksi awal..."):
                results = model(img)
            result = results[0]

            # Try to get a sensible default predicted label and confidence
            try:
                if hasattr(result, 'probs') and result.probs is not None:
                    top_class_idx = int(result.probs.top1)
                    default_label = result.names[top_class_idx]
                    default_conf = float(result.probs.top1conf.item() * 100)
                else:
                    cls_list = result.boxes.cls.tolist() if hasattr(result.boxes, 'cls') else []
                    conf_list = result.boxes.conf.tolist() if hasattr(result.boxes, 'conf') else []
                    if len(cls_list) > 0:
                        default_label = result.names[int(cls_list[0])]
                        default_conf = float(conf_list[0] * 100) if len(conf_list) > 0 else 0.0
                    else:
                        default_label = list(result.names.values())[0] if hasattr(result, 'names') else 'Unknown'
                        default_conf = 0.0
            except Exception:
                default_label = 'Unknown'
                default_conf = 0.0

            # Automatically show predicted label (species + breed) using default_label
            species, breed = _parse_species_and_breed(default_label)
            st.success("Analisis Otomatis Selesai!")
            st.metric(label="Hasil Prediksi", value=species)
            if breed:
                st.markdown(f"### Ras: {breed}")
            st.info(f"Tingkat Kepercayaan (Confidence): {default_conf:.2f}%")

            # Optional: allow manual label override inside an expander
            with st.expander("Ubah label (opsional)"):
                # Build candidate list: detected class names if any, otherwise all model class names
                candidates = []
                cls_list = result.boxes.cls.tolist() if hasattr(result.boxes, 'cls') else []
                if len(cls_list) > 0:
                    for ci in cls_list:
                        name = result.names[int(ci)]
                        if name not in candidates:
                            candidates.append(name)

                if not candidates:
                    candidates = [v for k, v in sorted(result.names.items())] if hasattr(result, 'names') else [default_label]

                if default_label in candidates:
                    default_idx = candidates.index(default_label)
                else:
                    candidates.insert(0, default_label)
                    default_idx = 0

                selected_label = st.selectbox("Pilih/konfirmasi label (ras)", candidates, index=default_idx)
                if st.button("Gunakan Label Terpilih", key="use_selected_label"):
                    species, breed = _parse_species_and_breed(selected_label)
                    st.success("Analisis Selesai (Manual)!")
                    st.metric(label="Hasil Prediksi", value=species)
                    if breed:
                        st.markdown(f"### Ras: {breed}")
                    conf_to_show = default_conf if selected_label == default_label else 0.0
                    st.info(f"Tingkat Kepercayaan (Confidence): {conf_to_show:.2f}%")