from tkinter import *
import time
import math
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)

# Constants
frame_x, frame_y = 1500, 900

# Tkinter Setup
root = Tk()
root.title("Lorenz")
root.attributes("-topmost", True)
root.geometry(str(frame_x) + "x" + str(frame_y))  # window size hardcoded
root.configure(background='black')

w = Canvas(root, width=frame_x, height=frame_y)
w.pack()


# WaterWheel Class
class WaterWheel:

    def __init__(self, x, y, radius, n_cups, theta_0):
        """
        :param x: x coordinate of center of wheel
        :param y: y coordinate of center of wheel
        :param radius: radius of the circum-circle of the polygon
        :param n_cups: number of cups on the wheel; number of corners of wheel
        :param theta_0: initial angle from horizontal
        """

        self.x = x
        self.y = y
        self.radius = radius
        self.theta = theta_0
        self.n_cups = n_cups

        self.omega = 0
        self.cup_masses = np.zeros(n_cups)

        # TEMPORARY
        # self.cup_masses[0] = 5

        self.max_mass_per_cup = 10
        self.g = 0.0000005

        self.point_locs = self.get_point_locs()

    def get_point_locs(self):
        points = []
        for i in range(self.n_cups):
            points.append(self.x + (self.radius * math.cos((2 * np.pi / self.n_cups * i) + self.theta)))
            points.append(self.y - (self.radius * math.sin((2 * np.pi / self.n_cups * i) + self.theta)))

        return points

    def get_point_locs_2(self):
        '''
            Same as above method, but returns n x 2 matrix with [x,y] coordinate pair for the n cups.
            The first method instead returns a flattened out 1 x 2n vector instead.
        '''

        flattened = np.array(self.get_point_locs())
        n = len(flattened)
        return flattened.reshape((int(n/2), 2))

    def get_center_of_mass(self):
        '''Returns tuple (x,y) location of center of mass of water'''

        locs = self.get_point_locs_2()
        return np.dot(locs.T, self.cup_masses) / sum(self.cup_masses)

    def update_step(self):
        # Rotation
        v = 0.03 # Ideal: 0.03
        damping = -v*self.omega
        x = np.cos((2*np.pi/self.n_cups * np.arange(self.n_cups)) + self.theta)

        torque = - self.g * self.radius * np.dot(self.cup_masses, x)
        torque += damping

        self.omega += torque
        self.theta += self.omega
        self.point_locs = self.get_point_locs()

        # Water leaking (Ideal: 0.006)
        leak_rate = 0.006

        for i in range(len(self.cup_masses)):
            self.cup_masses[i] = max(self.cup_masses[i] - leak_rate, 0)

        # Water adding (Ideal: 0.08)
        fill_rate = 0.08

        for i in range(len(self.cup_masses)):
            if self.get_cup_points(i)[1] <= self.y - self.radius - 27:
                self.cup_masses[i] = min(self.cup_masses[i] + fill_rate, self.max_mass_per_cup)

    def get_cup_points(self, cup_num):
        x = self.point_locs[2*cup_num]
        y = self.point_locs[(2*cup_num)+1]

        base_ratio = 1.5 # top base length / bottom base length
        altitude = 60
        bot_base_len = 40
        top_base_len = base_ratio * bot_base_len

        return [x - (top_base_len/2), y - (altitude/2), x + (top_base_len/2), y - (altitude/2),
                x + (bot_base_len/2), y + (altitude/2), x - (bot_base_len/2), y + (altitude/2)]

    def get_water_points(self, cup_num):
        x = self.point_locs[2 * cup_num]
        y = self.point_locs[(2 * cup_num) + 1]

        mass_ratio = self.cup_masses[cup_num] / self.max_mass_per_cup

        base_ratio = 1.5 # top base length / bottom base length
        altitude = 60
        bot_base_len = 40
        top_base_len = bot_base_len * (1 + (mass_ratio * (base_ratio - 1)))

        h = mass_ratio * altitude

        return [x - (top_base_len / 2), y + (altitude / 2) - h, x + (top_base_len / 2), y + (altitude / 2) - h,
                x + (bot_base_len / 2), y + (altitude / 2), x - (bot_base_len / 2), y + (altitude / 2)]


# Model Stuff
wheel = WaterWheel(frame_x/2, frame_y/2, 200, 20, np.radians(3))
from collections import deque
com_positions = deque([frame_x/2, frame_y/2])
iter = 0

# GUI Stuff
poly = w.create_polygon(wheel.get_point_locs(), fill='', outline='black', width='10')
center_rad = 10
center = w.create_oval(wheel.x - center_rad, wheel.y - center_rad, wheel.x + center_rad, wheel.y + center_rad, fill='black')

for i in range(wheel.n_cups):
    w.create_polygon(wheel.get_cup_points(i), tags='cup'+str(i), fill='', outline='brown', width=3)
    w.create_polygon(wheel.get_water_points(i), tags='water'+str(i), fill='blue')

point_rad = 5
point = w.create_oval(wheel.x - point_rad, wheel.y - point_rad, wheel.x + point_rad, wheel.y + point_rad, fill='red')

# Saving to file setup
f = open('CenterOfMassLocations.txt', 'wb')
np.savetxt(f, [], header="x, y, omega\n")

# System Params
animate = True
com_tracking = True # Do we even wanna track at all
anti_resolution = 4 # The smaller the more points will be plotted

while iter < 20000 or not com_tracking:
    iter += 1

    # Update model
    wheel.update_step()

    com = wheel.get_center_of_mass()
    prev_com = [com_positions[len(com_positions)-2], com_positions[len(com_positions)-1]]

    if np.linalg.norm(com - prev_com) >= anti_resolution and com_tracking:
        com_positions.append(com[0]); com_positions.append(com[1])

        # Save each data point to file
        np.savetxt(f, np.column_stack((com[0], com[1], wheel.omega)), fmt='%f')

        # Update animation of com point
        if animate:
            w.coords(point, com[0] - point_rad, com[1] - point_rad, com[0] + point_rad, com[1] + point_rad)

    if len(com_positions) > 100:
        com_positions.popleft(); com_positions.popleft()

    if animate:
        # Path tracing animation
        if len(com_positions) >= 4:
            w.delete(w.find_withtag('line'))
            w.create_line(*com_positions, tags='line', fill='red', width=2)

        # Wheel animation
        w.coords(poly, wheel.get_point_locs())
        for i in range(wheel.n_cups):
            w.coords('water' + str(i), wheel.get_water_points(i))
            w.coords('cup'+str(i), wheel.get_cup_points(i))

    w.update()
    # time.sleep(0.01)

f.close()
root.destroy()

# Full 2D Plot of Projection
x = []
y = []
omegas = []
with open('CenterOfMassLocations.txt', 'r') as f:
    line = f.readline()
    header = []
    while line:
        if line.startswith('#'):
            header.append(line)
        else:
            data = line.split(' ')
            x.append(float(data[0]))
            y.append(float(data[1]))
            omegas.append(float(data[2][:len(data[1])-2]))

        line = f.readline()

plt.plot(x, y, linewidth=0.5, color='red')
plt.xlabel('x center')
plt.ylabel('y center')
plt.title('Center of Mass of System')

if com_tracking:
    plt.show()

# 3D full graph with angular velocity
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, omegas, linewidth=0.5, color='red')
plt.xlabel('x center')
plt.ylabel('y center')
ax.set_zlabel('Angular Velocity')
ax.grid(False)
plt.title('Full Trajectory of System in Phase Space')
plt.show()

mainloop()