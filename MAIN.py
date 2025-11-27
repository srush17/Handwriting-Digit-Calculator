import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageDraw, ImageTk
import PIL
import customtkinter as ctk

# Load the model
model = tf.keras.models.load_model('model.h5')  # ✅ correct


# Set dimensions
width, height = 1200, 300

# Define symbol map
def num_to_sym(x):
    return {
        10: '+',
        11: '-',
        12: '*',
        13: '/',
        14: '(',
        15: ')',
        16: '.'
    }.get(x, str(x))

# Preprocess and predict
def testing(img):
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0
    return model.predict(img)

# Expression solver
def solve_exp(preds):
    expression = "".join(ind for ind, acc in preds)
    try:
        result = eval(expression)
        result = float(f"{result:.4f}")
        txt.delete('1.0', ctk.END)
        sol.delete('1.0', ctk.END)
        txt.insert(ctk.INSERT, f"{expression}")
        sol.insert(ctk.INSERT, f"= {result}")
    except Exception:
        txt.delete('1.0', ctk.END)
        sol.delete('1.0', ctk.END)
        txt.insert(ctk.INSERT, f"{expression}")
        sol.insert(ctk.INSERT, "Invalid Expression")

# Save, segment, and predict
def mod():
    image1.save('image.png')
    img = cv2.imread('image.png')

    pad = 5
    h, w = img.shape[:2]
    padded_img = ~(np.ones((h + 2*pad, w + 2*pad, 3), dtype=np.uint8))
    padded_img[pad:pad + h, pad:pad + w] = img
    img = padded_img

    img = cv2.GaussianBlur(img, (5, 5), 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    bw = cv2.bitwise_not(bw)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    preds = []
    i = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_img = gray[y:y + h, x:x + w]

        if h > 1.25 * w:
            pad = 3 * (h // w) ** 3
            cropped_img = cv2.copyMakeBorder(cropped_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=255)
        if w > 1.1 * h:
            pad = 3 * (w // h) ** 3
            cropped_img = cv2.copyMakeBorder(cropped_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=255)

        resized = cv2.resize(cropped_img, (28, 28))
        padded = cv2.copyMakeBorder(resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)

        cv2.imwrite(f"imgs/img_{i}.png", padded)
        prediction = testing(padded)
        ind = np.argmax(prediction[0])
        acc = float(f"{prediction[0][ind] * 100:.2f}")
        preds.append((num_to_sym(ind), acc))

        # Visual feedback on image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(img, f"{num_to_sym(ind)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        i += 1

    cv2.imwrite("Contours.png", img)
    img_change()
    solve_exp(preds)

# Canvas paint
def paint(event):
    d = 15
    x1, y1 = event.x - d, event.y - d
    x2, y2 = event.x + d, event.y + d
    canv.create_oval(x1, y1, x2, y2, fill="black", width=25)
    draw.line([x1, y1, x2, y2], fill="black", width=35)

# Clear function
def clear():
    canv.delete('all')
    draw.rectangle((0, 0, width, height), fill=(255, 255, 255, 0))
    txt.delete('1.0', ctk.END)
    sol.delete('1.0', ctk.END)
    image_label.configure(image=blank_image)

# Refresh Contours Image
def img_change():
    img = Image.open('Contours.png')
    img = ctk.CTkImage(dark_image=img, size=(width//2.5, height//1.5))
    image_label.configure(image=img)
    image_label.image = img

# Setup GUI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.geometry("1400x800")
root.title("Handwriting digit Calculator by Srushti Bankar")

# Canvas
canv = ctk.CTkCanvas(root, width=width, height=height, bg='white')
canv.grid(row=0, column=0, columnspan=2, padx=10, pady=17)
canv.bind("<B1-Motion>", paint)

# Image init
image1 = PIL.Image.new("RGB", (width, height), (255, 255, 255))
draw = ImageDraw.Draw(image1)

# Text boxes
txt_font = ctk.CTkFont(family="Bahnschrift", size=28)
txt = ctk.CTkTextbox(root, height=50, width=500, font=txt_font)
txt.grid(row=2, column=0, padx=10, pady=5)

sol_font = ctk.CTkFont(family="Bahnschrift", size=30, weight="bold")
sol = ctk.CTkTextbox(root, height=50, width=500, font=sol_font, text_color="#32aaff")
sol.grid(row=3, column=0, padx=10, pady=5)

# Image area
blank_img = Image.open("Blank.png")
blank_image = ctk.CTkImage(dark_image=blank_img, size=(width//2.5, height//1.5))
image_label = ctk.CTkLabel(root, image=blank_image, text="")
image_label.grid(row=2, column=1, rowspan=2, padx=10, pady=5)

# Buttons
btn_font = ctk.CTkFont(family="Bahnschrift", size=18, weight="bold")

Pred = ctk.CTkButton(root, text="✨ Calculate ✨", command=mod, fg_color="#2288ff", hover_color="#66bbff",
                     font=btn_font, corner_radius=25, height=50)
Pred.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

Clr = ctk.CTkButton(root, text="❌ Clear", command=clear, fg_color="#cc0000", hover_color="#ff4444",
                    font=btn_font, corner_radius=25, height=50)
Clr.grid(row=1, column=1, padx=10, pady=10, sticky='ew')

# Directory info

# Run
root.mainloop()