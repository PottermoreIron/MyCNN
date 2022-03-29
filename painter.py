import tkinter as tk
from tkinter import *

import numpy as np
from PIL import ImageGrab, ImageOps
import torch


class Painter(tk.Tk):
    def __init__(self, model):
        tk.Tk.__init__(self)
        self.predict_button = None
        self.digit = None
        self.preset_button = None
        self.canvas = None
        self.width = 200
        self.height = 200
        self.background_color = "White"
        self.color = "Black"
        self.radius = 6
        self.createPainter()
        self.model = model

    def createPainter(self):
        self.canvas = Canvas(self, width=self.width, height=self.height, bg=self.background_color, cursor="cross")
        self.preset_button = tk.Button(self, text="Clear", command=self.preset)
        self.predict_button = tk.Button(self, text="Predict", command=self.predict)
        self.digit = tk.Label(self, text="", font="Helvetica")
        self.canvas.grid(row=0, column=0, pady=0, sticky=W, )
        self.preset_button.grid(row=1, column=0, pady=2)
        self.predict_button.grid(row=1, column=1, pady=2)
        self.digit.grid(row=0, column=1, padx=2, pady=2)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - self.radius), (event.y - self.radius)
        x2, y2 = (event.x + self.radius), (event.y + self.radius)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.color, outline=self.color)

    def preset(self):
        self.canvas.delete("all")

    def img2data(self):
        # need update or coordinates will be zero
        self.update()
        # twist a little
        x = self.winfo_rootx() + self.winfo_x() + 10
        y = self.winfo_rooty() + self.winfo_y() + 30
        x1 = x + self.width * 2
        y1 = y + self.height * 2
        # screenshot
        img = ImageGrab.grab().crop((x, y, x1, y1))
        img = ImageOps.invert(img)
        # 28x28
        img = img.resize((28, 28))
        # greyscale
        img = img.convert('L')
        img = torch.unsqueeze(torch.from_numpy(np.array(img)), dim=0).type(torch.FloatTensor)
        # 1x1x28x28
        img = img.unsqueeze(1)
        return img

    def predict(self):
        img = self.img2data()
        digit = torch.max(self.model(img)[0], 1)[1].data.numpy()
        self.digit.configure(text=str(digit))
