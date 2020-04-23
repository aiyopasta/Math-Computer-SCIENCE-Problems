from tkinter import *
import numpy as np
import time

# Window size
n = 10
window_w = int(2**n)
window_h = int(2**n)
np.set_printoptions(suppress=True)

# Tkinter Setup
root = Tk()
root.title("Animations")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = Canvas(root, width=window_w, height=window_h)
w.pack()


# Goes from (0,0) coordinate system to typical annoying one.
def A(x, y):
    return x + window_w/2, -y + window_h/2


# Goes from typical annoying coordinate system to (0,0) one.
def A_inv(x, y):
    return -window_w/2 + x, window_h/2 - y


def sine(x):
    return np.sin(x * np.pi/2)**2

def tanh(x):
    return np.tanh(4*x)**2

def sine_two(x):
    return np.sin(x * np.pi/2)**10

# Create Apple Color Pallete
# From: https://developer.apple.com/design/human-interface-guidelines/ios/visual-design/color/
apple_colors = {
    'darkblue': (0, 122, 255),
    'lightblue': (10, 132, 255),
    'darkgreen': (52, 199, 89),
    'lightgreen': (48, 209, 88),
    'darkindigo': (88, 86, 214),
    'lightindigo': (94, 92, 230),
    'darkorange': (255, 149, 0),
    'lightorange': (255, 159, 10),
    'darkpink': (255, 45, 85),
    'lightpink': (255, 55, 95),
    'darkpurple': (175, 82, 222),
    'lightpurple': (191, 90, 242),
    'darkred': (255, 59, 48),
    'lightred': (255, 69, 58),
    'darkteal': (90, 200, 250),
    'lightteal': (100, 210, 255),
    'darkyellow': (255, 204, 0),
    'lightyellow': (255, 214, 10)
}

# From https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter
def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb

from io import BytesIO
from matplotlib.mathtext import math_to_image
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rc('text', usetex=True)

matplotlib.rcParams['text.color'] = _from_rgb(apple_colors['lightindigo'])
matplotlib.rcParams['savefig.facecolor'] = 'black'
matplotlib.rcParams['savefig.bbox'] = 'tight'

fig, ax = plt.subplots(1,1)
fig.patch.set_alpha(0.5)



from PIL import ImageTk, Image
# Creating buffer for storing image in memory
buffer = BytesIO()

# Writing png image with our rendered greek alpha to buffer
# math_to_image('$\\int_C \\vec{F}\\cdot \\vec{dr}$', buffer, dpi=600, format='png')
# math_to_image('$\\int f(x)\\ dx$', buffer, dpi=600, format='png')
math_to_image('$\\int_C \\vec{Shrish}\\cdot \\vec{dr}$', buffer, dpi=600, format='png')

# Remoting bufeer to 0, so that we can read from it
buffer.seek(0)

# Creating Pillow image object from it
pimage = Image.open(buffer)

# Remove all black ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
img = pimage
datas = img.getdata()

newData = []
for item in datas:
    if item[0] == 0 and item[1] == 0 and item[2] == 0:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)

img.putdata(newData)
# Remove all black ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# Creating PhotoImage object from Pillow image object
image = ImageTk.PhotoImage(img)

first_round = True
second_round = True
t = 0
def run():
    global first_round, second_round, t, image
    w.configure(background='black')

    r = 200
    start_color = (0,0,0)#apple_colors['darkteal']
    end_color = apple_colors['darkred']
    dark_end_color = match_color((0, 0, 0), end_color, 0.1)
    if first_round:
        w.create_arc(*A(-r, r), *A(r, -r), style='arc', outline='cyan', width=5, start=0, extent=t, tag='arc')
        w.create_oval(*A(-r, r), *A(r, -r), fill=_from_rgb(match_color((0, 0, 0), end_color, 0.05)), tag='circle')
        w.create_line(*A(-2*r, 2*r), *A(-2*r, 2*r), fill=_from_rgb(apple_colors['lightindigo']), width=5, tag='l1')
        w.create_line(*A(2*r, 2*r), *A(2*r, 2*r), fill=_from_rgb(apple_colors['lightindigo']), width=5, tag='l2')
        w.create_line(*A(2*r, -2*r), *A(2*r, -2*r), fill=_from_rgb(apple_colors['lightindigo']), width=5, tag='l3')
        w.create_line(*A(-2*r, -2*r), *A(-2*r, -2*r), fill=_from_rgb(apple_colors['lightindigo']), width=5, tag='l4')

        w.create_image(*A(-160, 0), image=image, anchor=CENTER)
        first_round = False
    else:
        t = min(t+0.015, 1)
        arc = w.find_withtag('arc')
        circle = w.find_withtag('circle')
        l1 = w.find_withtag('l1')
        l2 = w.find_withtag('l2')
        l3 = w.find_withtag('l3')
        l4 = w.find_withtag('l4')

        w.itemconfig(arc, extent=str(359.99999*tanh(t)), outline=_from_rgb(match_color(start_color, end_color, t)))
        w.itemconfig(circle, fill=_from_rgb(match_color((0, 0, 0), dark_end_color, t)))
        w.coords(l1, *A(-2*r, 2*r), *A(-2*r + (2*2*r*tanh(t)), 2*r))
        w.coords(l2, *A(2*r, 2*r), *A(2*r, - (2*2*r*tanh(t)) + 2*r))
        w.coords(l3, *A(2*r, -2*r), *A(2*r - (2*2*r*tanh(t)), -2*r))
        w.coords(l4, *A(-2*r, -2*r), *A(-2*r, + (2*2*r*tanh(t)) - 2*r))

        if t >= 1:
            t = 0
            show_arc = False

    if second_round:
        w.find_withtag('')

    w.update()
    time.sleep(0.001)


def match_color(rgb, to_rgb, intensity):
    """Transforms first color to second. Intensity is % of transformation."""
    dr = to_rgb[0] - rgb[0]
    db = to_rgb[1] - rgb[1]
    dg = to_rgb[2] - rgb[2]

    return int(rgb[0] + intensity*dr), int(rgb[1] + intensity*db), int(rgb[2] + intensity*dg)


# Main function
if __name__ == '__main__':
    while True:
        run()

# Necessary line for Tkinter
mainloop()
