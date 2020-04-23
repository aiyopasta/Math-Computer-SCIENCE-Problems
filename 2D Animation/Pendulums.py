from ghetto_manim import *
from ghetto_manim import __from_rgb__
from tkinter import *
import time

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Tkinter Setup
root = Tk()
root.title("Cewl Animations :)")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = Canvas(root, width=window_w, height=window_h)
w.pack()


# Special Shape Classes

class CircleGroup(ParamShapeGroup):
    def __init__(self, n_circles, radius, color, global_center_x=0, global_center_y=0, start=0, stop=0):
        circles = []
        sep_distance = 20
        group_width = (2*radius*n_circles) + (sep_distance*(n_circles - 1))

        a = -group_width / 2
        add_factor = (2*radius) + sep_distance
        for i in range(n_circles):
            x = a + (i * add_factor)

            # NOTE: Here, stop>0 does not necessarily mean that circle will show up! It also depends on where the global
            #       stop also is. If the global stop is greater than this stop, then this stop actually matters.

            circles.append(Circle(x + global_center_x, global_center_y, radius, color, start=start, stop=1))

        super().__init__(circles, start=start, stop=stop)


class Pendulum(ParamShapeGroup):
    def __init__(self, pos_x, pos_y, length, g, init_theta, color):
        rod = Rectangle(-5, length/2, 5, -length/2, color, start=0, stop=1, rot_theta=0, anchor_x=0, anchor_y=length/2)
        bob = Circle(0, -length/2, 30, color, start=0, stop=1, rot_theta=0, anchor_x=0, anchor_y=0)
        super().__init__([rod, bob], start=0, stop=0, global_x=pos_x, global_y=pos_y, global_theta=init_theta,
                         global_anchor_x=0, global_anchor_y=length/2)

        self.length = length
        self.g = g
        self.omega = 0
        self.alpha = 0

    def swing_step(self, t):
        self.alpha = -(self.g / self.length) * np.sin(self.offset_theta)
        self.omega += self.alpha
        self.offset_theta += self.omega

        self.rot_matrix = self.__rotation_matrix__(self.offset_theta)


# Time step
dt = 0.0005

# Object Creation
objects = []
# colorzz = [apple_colors['lightred'], apple_colors['lightblue'], apple_colors['lightorange'], apple_colors['lightindigo']]
n = 3
for i in range(n):
    p = Pendulum((-window_w/2) + (((i+1)/(n+1))*window_w), 0, length=250, g=i+1, init_theta=np.pi/4, color=None)
    objects.append(p)

# Animation Tree Construction
animator = Animator()
empty_anim = animator.get_root()

for p in objects:
    draw = animator.add_animation(p.draw_step, [1, smooth], duration=30, parent_animation=empty_anim, delay=0)
    coincide = animator.add_animation(p.translate_step, [-70, -100, smooth], duration=25, parent_animation=draw, delay=10)
    swing = animator.add_animation(p.swing_step, [], duration=500, parent_animation=coincide, delay=10)


# ... add more animations


# Idea: Can you 'yield' to solve connected shapes problem in groups.

# Scene parameters
first_runs = [[True for j in range(obj.n_subobjects)] for obj in objects]
def scene():
    global first_runs, dt, objects, animator
    w.configure(background='black')

    animator.update_step()

    for i, obj in enumerate(objects):
        for j, point_set in enumerate(obj.get_drawing_points(0.015)):
            if len(point_set) >= 4:
                if first_runs[i][j]:
                    first_runs[i][j] = False
                    w.create_line(*A_many(point_set), fill=__from_rgb__(obj.color), smooth=0, width=2, tag='obj'+str(i)+str(j))
                else:
                    line = w.find_withtag('obj'+str(i)+str(j))
                    w.itemconfig(line, fill=__from_rgb__(obj.color))
                    w.coords(line, *A_many(point_set))

    w.update()
    time.sleep(dt)


# Main function
if __name__ == '__main__':
    while True:
        scene()

# Necessary line for Tkinter
mainloop()