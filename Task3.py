from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image

# =========================================
# LOAD MODEL
# =========================================

model = ResNet50(weights="imagenet")

# =========================================
# MAIN WINDOW
# =========================================

root = Tk()
root.title("AI Image Classification")
root.geometry("1100x850")
root.config(bg="#0f172a")

# =========================================
# IMAGE LABEL
# =========================================

image_label = Label(
    root,
    bg="#0f172a"
)

image_label.pack(pady=20)

# =========================================
# CAPTION DATABASE
# =========================================

captions = {
    "Pembroke": "A Pembroke Welsh Corgi dog running on grass.",
    "golden_retriever": "A Golden Retriever standing outdoors.",
    "Labrador_retriever": "A Labrador Retriever dog.",
    "tabby": "A tabby cat sitting calmly.",
    "tiger_cat": "A cat looking at the camera.",
    "Persian_cat": "A Persian cat resting peacefully.",
    "sports_car": "A sports car parked on the road.",
    "school_bus": "A yellow school bus on the street.",
    "airliner": "A large passenger airplane in the sky.",
    "mountain_bike": "A mountain bike ready for riding."
}

# =========================================
# UPLOAD IMAGE FUNCTION
# =========================================

def upload_image():

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files",
             "*.jpg *.jpeg *.png")
        ]
    )

    if file_path:

        # DISPLAY IMAGE

        img = Image.open(file_path)
        img = img.resize((350, 350))

        photo = ImageTk.PhotoImage(img)

        image_label.config(image=photo)
        image_label.image = photo

        # PROCESS IMAGE

        img_for_model = image.load_img(
            file_path,
            target_size=(224, 224)
        )

        img_array = image.img_to_array(
            img_for_model
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = preprocess_input(
            img_array
        )

        # PREDICT

        predictions = model.predict(
            img_array,
            verbose=0
        )

        decoded = decode_predictions(
            predictions,
            top=1
        )[0]

        object_name = decoded[0][1]
        confidence = decoded[0][2] * 100

        # BETTER CAPTION

        caption_text = captions.get(
            object_name,
            f"A photo containing {object_name.replace('_', ' ')}."
        )

        # SHOW RESULT

        result_label.config(
            text=
            f"Detected Object: {object_name.replace('_',' ').title()}\n\n"
            f"Caption: {caption_text}\n\n"
            f"Confidence: {confidence:.2f}%",
            fg="#22c55e"
        )

# =========================================
# RESET FUNCTION
# =========================================

def reset_app():

    image_label.config(image="")
    image_label.image = None

    result_label.config(text="")

# =========================================
# HEADER
# =========================================

top_frame = Frame(
    root,
    bg="#111827",
    height=120
)

top_frame.pack(fill="x")

Label(
    top_frame,
    text="AI IMAGE CLASSIFICATION",
    font=("Arial", 34, "bold"),
    bg="#111827",
    fg="#38bdf8"
).pack(pady=20)

Label(
    top_frame,
    text="Computer Vision + Deep Learning",
    font=("Arial", 14),
    bg="#111827",
    fg="#94a3b8"
).pack()

# =========================================
# BUTTON FRAME
# =========================================

button_frame = Frame(
    root,
    bg="#0f172a"
)

button_frame.pack(pady=25)

# UPLOAD BUTTON

Button(
    button_frame,
    text="UPLOAD IMAGE",
    command=upload_image,
    font=("Arial", 16, "bold"),
    bg="#06b6d4",
    fg="black",
    padx=25,
    pady=12,
    bd=0,
    cursor="hand2"
).grid(
    row=0,
    column=0,
    padx=15
)

# RESET BUTTON

Button(
    button_frame,
    text="RESET",
    command=reset_app,
    font=("Arial", 16, "bold"),
    bg="#ef4444",
    fg="white",
    padx=25,
    pady=12,
    bd=0,
    cursor="hand2"
).grid(
    row=0,
    column=1,
    padx=15
)

# =========================================
# RESULT LABEL
# =========================================

result_label = Label(
    root,
    text="Upload an image to start detection",
    font=("Arial", 20, "bold"),
    bg="#0f172a",
    fg="white",
    wraplength=900,
    justify=CENTER
)

result_label.pack(pady=30)

# =========================================
# FOOTER
# =========================================

Label(
    root,
    text="Developed Using Python, TensorFlow & ResNet50",
    font=("Arial", 12),
    bg="#0f172a",
    fg="#94a3b8"
).pack(side=BOTTOM, pady=20)

# =========================================
# RUN APP
# =========================================

root.mainloop()