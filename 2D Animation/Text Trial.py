from tkinter import *
import numpy as np
import time
from tkinter.font import Font

from PIL import ImageTk, Image

# Window size
n = 10
window_w = int(2.1**n)
window_h = int(2**n)
np.set_printoptions(suppress=True)

# Tkinter Setup
root = Tk()
root.title("Animations")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = Canvas(root, width=window_w, height=window_h)
w.pack()

from io import BytesIO
from matplotlib.mathtext import math_to_image
import matplotlib
from matplotlib import pyplot as plt

# Goes from (0,0) coordinate system to typical annoying one.
def A(x, y):
    return x + window_w/2, -y + window_h/2

# Goes from typical annoying coordinate system to (0,0) one.
def A_inv(x, y):
    return -window_w/2 + x, window_h/2 - y

def prepare_laTeX():
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['savefig.facecolor'] = 'black'
    matplotlib.rcParams['savefig.bbox'] = 'tight'

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_alpha(0.5)

def laTeX(text, color, size):
    # Set text color.
    matplotlib.rcParams['text.color'] = color
    matplotlib.rcParams['font.size'] = size

    # Turn math to image with black background.
    buffer = BytesIO()
    math_to_image(text, buffer, dpi=75, format='svg')
    buffer.seek(0)
    pimage = Image.open(buffer)

    # Remove black background to make transparent.
    data = pimage.getdata()
    newData = []
    for item in data:
        T = 0
        if item[0] <= T and item[1] <= T and item[2] <= T:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    pimage.putdata(newData)

    # Return final PhotoImage object to display.
    return ImageTk.PhotoImage(pimage)


# def


prepare_laTeX()
first = True
x = 0
def run():
    global x, first, work_integral, b
    w.configure(background='black')

    k = 70

    f_text = 'Avenir Next Ultra Light'
    font = Font(family=f_text, size=k)
    text = 'The trig function is '
    c = font.measure(text)

    if first:
        work_integral = laTeX('$\\sin{(x+\\pi)}$', 'red', k)

        w.create_text(*A(0,0), text=text, font=(f_text, k), fill='red', tag='txt', anchor=W)
        w.create_image(*A(c, -2), image=work_integral, anchor=W, tag='integral')
        # w.create_text(*A(137, 0), text='.', font=('Avenir Next Ultra Light', k), fill='red')
        first = False
    else:
        img = w.find_withtag('integral')
        w.coords(img, *A(c, -2))
        txt = w.find_withtag('txt')
        w.itemconfig(txt, text=text)


    w.update()
    # time.sleep(0.001)
    time.sleep(0.5)


# Main function
if __name__ == '__main__':
    while True:
        run()

# Necessary line for Tkinter
mainloop()