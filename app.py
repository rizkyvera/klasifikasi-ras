import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.title("🐱 Klasifikasi Kucing vs Anjing 🐶")
st.write("Unggah foto dari perangkat Anda untuk diprediksi menggunakan YOLO11!")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'best_breed.pt')


def _parse_species_and_breed(label: str):
    if label is None:
        return None, None
    lbl = str(label).strip()
    low = lbl.lower().replace('_', ' ')

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
        species = 'Anjing 🐶'

    return species, breed

@st.cache_resource
def load_custom_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_custom_model(model_path)

if model is None:
    st.error(f"Model '{model_path}' tidak ditemukan! Pastikan file .pt sudah diletakkan di folder 'model' di samping skrip ini.")
else:
    supported_breeds = [v.replace('_', ' ').title() for k, v in model.names.items()]
    supported_breeds.sort()
    
    with st.expander("Lihat Daftar Ras Kucing dan Anjing yang Didukung"):
        st.write(", ".join(supported_breeds))
    
    st.subheader("Unggah Gambar untuk Prediksi")
    uploaded_file = st.file_uploader("Pilih file gambar dari perangkat Anda", type=["jpg", "jpeg", "png"], key="upload_external")
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Gambar Unggahan", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memuat atau menampilkan gambar unggahan: {e}")

        with st.spinner("Mengambil prediksi awal..."):
            results = model(img)
        result = results[0]

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

        species, breed = _parse_species_and_breed(default_label)
        
        if default_conf < 50.0:
            st.warning("⚠️ Gambar tidak dikenali sebagai salah satu dari 37 ras hewan yang didukung. Mohon masukkan gambar kucing/anjing yang jelas.")
        else:
            st.success("Analisis Otomatis Selesai!")
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Hasil Prediksi", value=species)
            if breed:
                st.markdown(f"### Ras: {breed}")
            st.info(f"Tingkat Kepercayaan (Confidence): {default_conf:.2f}%")
        
        with col2:
            ground_truth_dir = os.path.join(BASE_DIR, 'ground_truth')
            gt_image_path = None
            if os.path.exists(ground_truth_dir):
                for f in os.listdir(ground_truth_dir):
                    name, _ = os.path.splitext(f)
                    if name.lower() == default_label.lower():
                        gt_image_path = os.path.join(ground_truth_dir, f)
                        break
            
            if gt_image_path:
                st.markdown("#### Pembanding (Ground Truth)")
                try:
                    gt_img = Image.open(gt_image_path).convert("RGB")
                    st.image(gt_img, caption=f"Contoh {breed}", use_container_width=True)
                except Exception as e:
                    st.error(f"Gagal memuat gambar pembanding: {e}")
