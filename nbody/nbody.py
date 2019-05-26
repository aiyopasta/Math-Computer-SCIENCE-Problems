from tkinter import *
import time
from random import random
import math
import numpy as np

default_mass = 10
density = 0.002

frame_x, frame_y = 1500, 900

root = Tk()
root.title("n-body")
root.attributes("-topmost", True)
root.geometry(str(frame_x) + "x" + str(frame_y))  # window size hardcoded

w = Canvas(root, width=frame_x, height=frame_y)
w.pack()

class Body:
    CONFIG = {
        "position": None,
        "velocity": None,
        'mass': None,
        "radius": None
    }
    def __init__(self, init_pos, init_vel, mass=default_mass):
        self.position = init_pos
        self.velocity = init_vel
        self.mass = mass
        self.radius = 10

    def move_step(self):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

    def accelerate(self, dv):
        self.velocity[0] += dv[0]; self.velocity[1] += dv[1]

    def dist_to(self, Body):
        return Body.position[0] - self.position[0], Body.position[1] - self.position[1]

def run():

    w.configure(background='black')

    center_frame = [0,0]

    # 3-body system

    b1 = Body([frame_x/7, frame_y/2], [-3,-7], 1E14)
    b2 = Body([frame_x/7, frame_y/2 + 200], [3, 2], 1E14)
    b3 = Body([frame_x/2 + 200, frame_y/2 + 200], [-4, -3], 1E14)
    b4 = Body([frame_x/2, frame_y/2 + 250], [4, -1], 1E14)
    bodies = [b1, b2, b3, b4]

    # Planetary

    # sun = Body([frame_x/2, frame_y/2], [0,0], 1E14)
    # earth = Body([frame_x/2, frame_y/2 + 250], [5,0], 30)
    # moon = Body([frame_x/2, frame_y/2 - 360], [-5,0], 5)
    # jupiter = Body([frame_x/2, frame_y/2 - 400], [-4,0], 1E12)
    # bodies = [sun, jupiter, moon, earth]

    # Chaos Planets: change color to white otherwise error.

    # sun = Body([frame_x / 2, frame_y / 2], [0, 0], 1E14)
    # bodies = [Body([frame_x/2, random()*(frame_y)], [random()*10-5, 0], 1E10) for _ in range(100)]
    # bodies.append(sun)

    G = 6.67E-11

    spacing = 150
    vert_lines = []
    for i in range(int(frame_x / spacing)):
        line = w.create_line(i * spacing, 0, i * spacing, frame_y, fill='purple', tags='vert'+str(i))
        vert_lines.append(line)

    horiz_lines = []
    for i in range(int(frame_y / spacing)):
        line = w.create_line(0, i * spacing, frame_x, i * spacing, fill='purple', tags='horiz'+str(i))
        horiz_lines.append(line)

    colors = ['red', 'blue', 'orange', 'green']

    for number, body in enumerate(bodies):
        w.create_oval(body.position[0] - body.radius, body.position[1] - body.radius,
                      body.position[0] + body.radius, body.position[1] + body.radius, fill=colors[number], outline='', tags='b'+str(number))

    positions = [[] for _ in bodies]

    while (True):
        for body in bodies:
            F_x = 0; F_y = 0
            for body2 in bodies:
                if body2 is not body:
                    disp_x, disp_y = body.dist_to(body2)
                    r = math.sqrt(disp_x**2 + disp_y**2)
                    F_x += (G * body.mass * body2.mass / (r**3)) * disp_x
                    F_y += (G * body.mass * body2.mass / (r**3)) * disp_y

            body.accelerate((F_x/body.mass, F_y/body.mass))

        center_of_mass_x = sum([body.position[0] * body.mass for body in bodies]) / sum([body.mass for body in bodies])
        center_of_mass_y = sum([body.position[1] * body.mass for body in bodies]) / sum([body.mass for body in bodies])

        vel_center_of_mass_x = sum([body.velocity[0] * body.mass for body in bodies]) / sum([body.mass for body in bodies])
        vel_center_of_mass_y = sum([body.velocity[1] * body.mass for body in bodies]) / sum([body.mass for body in bodies])

        if not (frame_x / 3 < center_of_mass_x - center_frame[0] < 2 * frame_x / 3):
            center_frame[0] += abs(vel_center_of_mass_x) * np.sign(center_of_mass_x - center_frame[0] - 2 * frame_x / 3)

        if not (frame_y / 3 < center_of_mass_y - center_frame[1] < 2 * frame_y / 3):
            center_frame[1] += abs(vel_center_of_mass_y) * np.sign(center_of_mass_y - center_frame[1] - 2 * frame_y / 3)

        for number, body in enumerate(bodies):
            body.move_step()
            circle = w.find_withtag('b'+str(number))
            w.coords(circle, body.position[0] - body.radius - center_frame[0], body.position[1] - body.radius - center_frame[1],
                             body.position[0] + body.radius - center_frame[0], body.position[1] + body.radius - center_frame[1])

        for i, vert in enumerate(vert_lines):
            x = ((i * spacing) - center_frame[0]) % frame_x
            w.coords(vert, x, 0, x, frame_y)

        for i, horiz in enumerate(horiz_lines):
            y = ((i * spacing) - center_frame[1]) % frame_y
            w.coords(horiz, 0, y, frame_x, y)

        # Line tracing

        for i in range(len(positions)):
            if len(positions[i]) >= 100:
                positions[i].pop(0)

            new_pos = [[bodies[i].position[0], bodies[i].position[1]]]
            positions[i].append(new_pos)
            point_array = np.array(positions[i]).flatten()
            w.delete(w.find_withtag('line'+str(i)))

            # center stage compatibility
            adjustments = np.array([[-center_frame[0], -center_frame[1]] for i in range(int(len(point_array)/2))]).flatten()

            if len(point_array)>=4:
                w.create_line(*(point_array+adjustments), tags='line'+str(i), fill=colors[i])

        w.update()
        time.sleep(0.001)


if __name__ == '__main__':
    run()

mainloop()