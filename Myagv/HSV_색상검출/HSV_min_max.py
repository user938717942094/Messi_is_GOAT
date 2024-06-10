import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

def update_hsv(event):
    global hsv_min, hsv_max
    hsv_min[0] = hue_min_slider.get()
    hsv_min[1] = saturation_min_slider.get()
    hsv_min[2] = value_min_slider.get()
    hsv_max[0] = hue_max_slider.get()
    hsv_max[1] = saturation_max_slider.get()
    hsv_max[2] = value_max_slider.get()

def update_image():
    global hsv_min, hsv_max
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    result = cv2.bitwise_and(image, image, mask=mask)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(result_rgb)
    img_tk = ImageTk.PhotoImage(image=img)
    label.config(image=img_tk)
    label.image = img_tk
    label.after(10, update_image)

# Load image
image = cv2.imread('ros4.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Initialize HSV min/max values
hsv_min = np.array([0, 0, 0])
hsv_max = np.array([180, 255, 255])

# Create main window
root = tk.Tk()
root.title("HSV Color Picker")

# Create sliders for HSV min values
hue_min_slider = tk.Scale(root, from_=0, to=180, label="Hue Min", orient=tk.HORIZONTAL, command=update_hsv)
hue_min_slider.set(hsv_min[0])
hue_min_slider.pack()

saturation_min_slider = tk.Scale(root, from_=0, to=255, label="Saturation Min", orient=tk.HORIZONTAL, command=update_hsv)
saturation_min_slider.set(hsv_min[1])
saturation_min_slider.pack()

value_min_slider = tk.Scale(root, from_=0, to=255, label="Value Min", orient=tk.HORIZONTAL, command=update_hsv)
value_min_slider.set(hsv_min[2])
value_min_slider.pack()

# Create sliders for HSV max values
hue_max_slider = tk.Scale(root, from_=0, to=180, label="Hue Max", orient=tk.HORIZONTAL, command=update_hsv)
hue_max_slider.set(hsv_max[0])
hue_max_slider.pack()

saturation_max_slider = tk.Scale(root, from_=0, to=255, label="Saturation Max", orient=tk.HORIZONTAL, command=update_hsv)
saturation_max_slider.set(hsv_max[1])
saturation_max_slider.pack()

value_max_slider = tk.Scale(root, from_=0, to=255, label="Value Max", orient=tk.HORIZONTAL, command=update_hsv)
value_max_slider.set(hsv_max[2])
value_max_slider.pack()

# Create label to display image
label = tk.Label(root)
label.pack()

# Update the image
update_image()

root.mainloop()
