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
    def __init__(self, pos_x, pos_y, length, g, init_theta, rod_color, bob_color):
        # Remember again: `stop' has a different meaning for subobjects in a group. See CircleGroup class.
        rod = Rectangle(-5, length/2, 5, -length/2, rod_color, fill_p=0.2, start=0, stop=1, rot_theta=0, anchor_x=0, anchor_y=length/2)
        bob = Circle(0, -length/2, 30, bob_color, fill_p=0.1, start=0, stop=1, rot_theta=0, anchor_x=0, anchor_y=0)
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


def scene1():
    # Object Creation
    objects = []
    n = 3
    for i in range(n):
        p = Pendulum((-window_w / 2) + (((i + 1) / (n + 1)) * window_w), 0, length=250, g=i + 1, init_theta=np.pi / 4,
                     rod_color=apple_colors['lightorange'], bob_color=apple_colors['lightteal'])
        objects.append(p)

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()

    for p in objects:
        draw = animator.add_animation(p.draw_step, [1, smooth], duration=30, parent_animation=empty_anim, delay=0)
        coincide = animator.add_animation(p.translate_step, [-70, -100, smooth], duration=25, parent_animation=draw,
                                          delay=10)
        swing = animator.add_animation(p.swing_step, [], duration=500, parent_animation=coincide, delay=10)

    # ... add more animations

    # Scene parameters
    da_Vinci = Painter(w, objects)

    #  Play em'! ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    while True:
        w.configure(background='black')
        animator.update_step()
        da_Vinci.paint_step()

        time.sleep(dt)

def scene2():
    # Object Creation
    r = 100
    fake_rect = Rectangle(-r, r, r, -r, color=apple_colors['darkpurple'], stop=0)
    real_rect = Rectangle(-r, r, r, -r, color=(0, 0, 0), stop=0)
    circle = Circle(0, 0, r, color=apple_colors['lightindigo'], stop=0)

    pendulum = Pendulum(300, 0, 200, 3, np.pi/6, apple_colors['lightgreen'], apple_colors['lightred'])
    objects = [real_rect, fake_rect, circle, pendulum]

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()

    fade_in =  animator.add_animation(real_rect.fade_color_step, [apple_colors['lightred'], smooth], duration=10, parent_animation=empty_anim, delay=0)
    draw_real = animator.add_animation(real_rect.draw_step, [1, smooth], duration=25, parent_animation=empty_anim, delay=0)
    draw = animator.add_animation(fake_rect.draw_step, [1, smooth], duration=35, parent_animation=draw_real, delay=10)
    undraw = animator.add_animation(fake_rect.undraw, [1, smooth], duration=30, parent_animation=draw_real, delay=25)

    morph = animator.add_animation(real_rect.morph_step, [circle, smooth], duration=20, parent_animation=undraw, delay=10)
    flip = animator.add_animation(real_rect.scale_step, [-1, 1, smooth], duration=20, parent_animation=undraw, delay=10)
    fade_indigo = animator.add_animation(real_rect.fade_color_step, [apple_colors['lightindigo'], smooth], duration=15, parent_animation=undraw, delay=5)

    draw_pend = animator.add_animation(pendulum.draw_step, [1, smooth], duration=25, parent_animation=fade_indigo, delay=5)
    swing = animator.add_animation(pendulum.swing_step, [], duration=200, parent_animation=draw_pend, delay=5)

    # ... add more animations

    picasso = Painter(w, objects)

    # Run Scene!
    while True:
        w.configure(background='black')

        animator.update_step()
        picasso.paint_step()

        time.sleep(dt)


# Main function
if __name__ == '__main__':
    # scene1()
    scene2()

# Necessary line for Tkinter
mainloop()