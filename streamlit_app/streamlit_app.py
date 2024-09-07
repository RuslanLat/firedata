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
model.load_model("streamlit_app/firemlmodel_v1.json", format="json")

st.subheader("🌳 Сервис сегментирования изображения заданной территории")

tiff_form = st.form("firedata")
uploaded_img_file = tiff_form.file_uploader(
    "Загрузка файла изображения", type=["tiff"], help="загрузите файл изображения"
)
uploaded_csv_file = tiff_form.file_uploader(
    "Загрузка файла данных погоды", type=["csv"], help="загрузите файл данных погоды"
)
prop = tiff_form.slider(
    "Уровень доверия к модели, %",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    help="Выбирете порог",
)
submitted = tiff_form.form_submit_button("Сегментировать территорию", type="primary")
if submitted and uploaded_img_file and uploaded_csv_file:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_img_file.getvalue())
        tmp_path = tmp.name

    with rasterio.open(tmp_path) as src:
        test_shape = src.read(1).shape

        df = pd.DataFrame(src.read(5).reshape(-1, 1))
        df.columns = ["target"]
        for i in range(1, 5):
            df[f"band{i}"] = src.read(i).reshape(-1, 1)

        df_csv = pd.read_csv(uploaded_csv_file)

        df_csv.columns = [
            "date",
            "t_min",
            "t_avg",
            "t_max",
            "rains",
            "wind_way",
            "wind_speed",
            "wind_s",
            "pressure",
        ]
        for i, j in enumerate(df_csv["t_max"].tolist()):
            df[f"t_max_day{-62 + i}"] = j
        for i, j in enumerate(df_csv["wind_way"].tolist()[-10:]):
            df[f"t_wind_way_day{-10 + i}"] = j
        for i, j in enumerate(df_csv["wind_speed"].tolist()[-10:]):
            df[f"t_win_speed_day{-10 + i}"] = j
        for i, j in enumerate(df_csv["pressure"].tolist()[-10:]):
            df[f"t_pressure_day{-10 + i}"] = j

        red = src.read(1)  # B02 - Blue
        green = src.read(2)  # B03 - Green
        blue = src.read(3)  # B04 - Red
        ik = src.read(4)

    st.success("Файл загружен", icon="✅")
elif submitted and not uploaded_img_file or not uploaded_csv_file:
    st.error("Загрузите файл", icon="❌")
else:
    st.warning("Загрузите файл", icon="⚠️")

if submitted and uploaded_img_file and uploaded_csv_file:
    res = model.predict_proba(df)[:, 1]
    img = np.array([1 if i > prop else 0 for i in res]).reshape(test_shape)
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
