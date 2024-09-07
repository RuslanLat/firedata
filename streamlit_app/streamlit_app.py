import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from catboost import CatBoostClassifier
from tempfile import NamedTemporaryFile

st.set_page_config(
    page_title="Министерство природных ресурсов и экологии Российской Федерации",
    page_icon="streamlit_app/images/favicon.ico",
)

model = CatBoostClassifier()
model.load_model("streamlit_app/firemlmodel.json", format="json")

st.subheader("🌳 Сервис сегментирования изображения заданной территории")

tiff_form = st.form("rosatom")
uploaded_file = tiff_form.file_uploader(
    "Загрузка файла", type=["tiff"], help="загрузите файл"
)
submitted = tiff_form.form_submit_button("Сегментировать территорию", type="primary")
if submitted and uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with rasterio.open(tmp_path) as src:
        test_shape = src.read(1).shape
        red = src.read(1)  # B02 - Blue
        green = src.read(2)  # B03 - Green
        blue = src.read(3)  # B04 - Red
        ik = src.read(4)
        df = pd.DataFrame(red.reshape(-1, 1))
        df.columns = ["band1"]
        for i in range(2, 5):
            df[f"band{i}"] = src.read(i).reshape(-1, 1)
    st.success("Файл загружен", icon="✅")
elif submitted and not uploaded_file:
    st.error("Загрузите файл", icon="❌")
else:
    st.warning("Загрузите файл", icon="⚠️")

if submitted and uploaded_file:
    res = model.predict_proba(df)[:, 1]
    img = np.array([1 if i > 0.45 else 0 for i in res]).reshape(test_shape)
    photo_full = np.stack([red, green, blue], axis=-1)  # Отрисовка всего изображения
    photo_ik = np.stack([ik], axis=-1)  # Отрисовка ИК-слоя изображения
    photo_mask = np.stack([img], axis=-1)  # Отрисовка маски изображения

    photo_full = photo_full.astype(np.uint8)
    photo_ik = photo_ik.astype(np.uint8)
    photo_mask = photo_mask.astype(np.uint8)
    # Create a temporary file to save the uploaded file
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].imshow(photo_full)
    ax[1].imshow(photo_ik)
    ax[2].imshow(photo_mask)

    ax[0].set_title("Сегментируемая территория")
    ax[1].set_title("ИК канал изображения")
    ax[2].set_title("Очаги вероятного пожара")

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")

    st.subheader("Результат сегментации заданной территории")
    st.pyplot(fig)
