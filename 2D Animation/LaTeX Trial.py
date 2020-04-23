import tkinter as tk
import matplotlib
from matplotlib.mathtext import math_to_image
from io import BytesIO
from PIL import ImageTk, Image

matplotlib.rc('text', usetex=True)


class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):

        # Creating buffer for storing image in memory
        buffer = BytesIO()

        # Writing png image with our rendered greek alpha to buffer
        math_to_image('$\\int_C \\vec{F}\\cdot \\vec{dr}\\int_C \\vec{F}\\cdot \\vec{dr}\\int_C \\vec{F}\\cdot \\vec{dr}$', buffer, dpi=1000, format='png')

        # Remoting bufeer to 0, so that we can read from it
        buffer.seek(0)

        # Creating Pillow image object from it
        pimage= Image.open(buffer)

        # Creating PhotoImage object from Pillow image object
        image = ImageTk.PhotoImage(pimage)

        # Creating label with our image
        self.label = tk.Label(self, image=image)

        # Storing reference to our image object so it's not garbage collected,
        # as TkInter doesn't store references by itself
        self.label.img = image

        self.label.pack(side="bottom")
        self.QUIT = tk.Button(self, text="QUIT", fg="red", command=root.destroy)
        self.QUIT.pack(side="top")

print(matplotlib.rcParams.keys())

matplotlib.rcParams['text.color'] = 'red'
# matplotlib.rcParams['savefig.facecolor'] = 'black'
matplotlib.rcParams['savefig.format'] = 'svg'
matplotlib.rcParams['font.size'] = 4
matplotlib.rcParams['savefig.bbox'] = 'tight'

root = tk.Tk()
root.attributes("-topmost", True)
app = Application(master=root)
app.mainloop()