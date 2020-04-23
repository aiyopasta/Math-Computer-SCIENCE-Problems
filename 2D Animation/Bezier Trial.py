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
    return np.array([x + window_w/2, -y + window_h/2])


# Goes from typical annoying coordinate system to (0,0) one.
def A_inv(x, y):
    return np.array([-window_w/2 + x, window_h/2 - y])


t = 0
radius = 90
p1 = A(-radius, -radius); p2 = A(-radius, radius)
p3 = A(-radius, radius); p4 = A(radius, radius)

s1 = None; s2 = None

first_run = True
def run():
    global first_run, t, p1, p2, p3, p4, s1, s2
    w.configure(background='black')

    if t <= 1:
        s1 = p1 + ((p2 - p1) * t)
        s2 = p3 + ((p4 - p3) * t)

        if first_run:
            w.create_line(*p1, *p2, fill='blue')
            w.create_line(*p3, *p4, fill='blue')

            w.create_line(*p1, *s1, fill='red', tag='l1')
            w.create_line(*p3, *s2, fill='red', tag='l2')

            w.create_line(*s1, *(s1+((s2-s1)*t)), fill='green', tag='s')

        else:
            l1 = w.find_withtag('l1')
            l2 = w.find_withtag('l2')
            w.coords(l1, *p1, *(p1 + ((p2 - p1) * t)))
            w.coords(l2, *p3, *(p3 + ((p4 - p3) * t)))

            s = w.find_withtag('s')
            q = s1+((s2-s1)*t)
            w.coords(s, *s1, *q)

            r = 2
            w.create_oval(q[0]-r, q[1]-r, q[0]+r, q[1]+r, fill='green', outline=None)

        t += 0.01
        first_run = False

    w.update()
    time.sleep(0.001)


# Main function
if __name__ == '__main__':
    while True:
        run()

# Necessary line for Tkinter
mainloop()
