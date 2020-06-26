from ghetto_manim import *
import time

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Special Shape Classes
class Arrow(ParamShapeGroup):
    def __init__(self, x0, y0, x1, y1, color, fill_p=0., curve_place=0.5, curve_amount=0, start=0, stop=1):
        curve = CurvedLine(x0, y0, x1, y1, color, -1, curve_place, curve_amount, stop=1)
        triangle = Triangle(x1, y1, color, fill_p=fill_p, stop=1, rot_theta=curve.curve_angle(), scale_x=10, scale_y=10)

        super().__init__([curve, triangle], start=start, stop=stop)


# Combined Animation Functions
    # ...


# Time Step
dt = 0.001

def scene():
    # Object Construction
    r = 40
    k = 1.05
    l = 0.83
    original_color = apple_colors['lightindigo']

    rect = Circle(0, 0, 2*r, color=original_color, fill_p=-1, stop=0)
    rect2 = Circle(0, 0, 2*k*r, color=apple_colors['lightyellow'], stop=0)
    rect_0 = Circle(0, 0, 2*l*r, color=original_color, fill_p=0, stop=0)

    circle = Circle(-400, 0, 2*r, color=apple_colors['lightpurple'], stop=1)

    oliver = Arrow(-400, 0, 0, 0, apple_colors['lightpurple'], curve_amount=0, fill_p=0.15, stop=0)
    queen = Arrow(0, 100, 35, 100, apple_colors['lightyellow'], curve_amount=400, fill_p=0.15, stop=0)
    thea = Arrow(-400, 2*r, -2*r, 0, apple_colors['lightteal'], curve_amount=200, fill_p=0.15, stop=0)
    objects = [oliver, queen, thea, rect, rect2, rect_0, circle]

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()

    draw = animator.add_animation(oliver.draw_step, [1, smooth], duration=30, parent_animation=empty_anim, delay=0)
    draw2 = animator.add_animation(queen.draw_step, [1, smooth], duration=30, parent_animation=empty_anim, delay=0)
    draw3 = animator.add_animation(rect.draw_step, [1, smooth], duration=30, parent_animation=empty_anim, delay=0)
    draw4 = animator.add_animation(rect_0.draw_step, [1, smooth], duration=30, parent_animation=empty_anim, delay=0)

    trace0 = animator.add_animation(thea.draw_step, [1, smooth], duration=20, parent_animation=draw4, delay=15)

        # Recognizing Vertex Reached
    trace = animator.add_animation(rect2.draw_step, [1, smooth], duration=30, parent_animation=trace0, delay=0)
    untrace = animator.add_animation(rect2.undraw, [1, smooth], duration=20, parent_animation=trace0, delay=15)
    remove_trace = animator.add_animation(rect2.abrupt_removal, [], duration=1, parent_animation=untrace, delay=0)
    scale_up = animator.add_animation(rect.scale_step, [1.1, 1.1, smooth], duration=20, parent_animation=trace0, delay=25)
    scale_up2 = animator.add_animation(rect2.scale_step, [1.1, 1.1, smooth], duration=20, parent_animation=trace0, delay=25)
    scale_up3 = animator.add_animation(rect_0.scale_step, [1.2, 1.2, smooth], duration=20, parent_animation=trace0, delay=25)
    fill_in = animator.add_animation(rect_0.fade_fill_step, [0.05, smooth], duration=20, parent_animation=trace0, delay=30)
    change_color = animator.add_animation(rect.fade_color_step, [apple_colors['lightyellow'], smooth], duration=20, parent_animation=trace0, delay=30)
    change_color = animator.add_animation(rect_0.fade_color_step, [apple_colors['lightyellow'], smooth], duration=20, parent_animation=trace0, delay=30)

        # Recognizing Vertex Left
    invisible_flip = animator.add_animation(rect2.scale_step, [-1, 1, smooth], duration=1, parent_animation=change_color, delay=0)
    invisible_change_color = animator.add_animation(rect2.fade_color_step, [(255,255,255), smooth], duration=1, parent_animation=change_color, delay=0)

    pause = 20

    scale_down = animator.add_animation(rect.scale_step, [1, 1, smooth], duration=20, parent_animation=change_color, delay=pause+10)
    scale_down2 = animator.add_animation(rect2.scale_step, [0.9, 0.9, smooth], duration=20, parent_animation=change_color, delay=pause+10)
    scale_down3 = animator.add_animation(rect_0.scale_step, [1, 1, smooth], duration=20, parent_animation=change_color, delay=pause+10)
    fill_out = animator.add_animation(rect_0.fade_fill_step, [0, smooth], duration=20, parent_animation=change_color, delay=pause+10)
    change_color_back = animator.add_animation(rect.fade_color_step, [original_color, smooth], duration=20, parent_animation=change_color, delay=pause+10)
    change_color_back2 = animator.add_animation(rect_0.fade_color_step, [original_color, smooth], duration=20, parent_animation=change_color, delay=pause+10)

    remove_trace = animator.add_animation(rect2.abrupt_removal, [], duration=1, parent_animation=untrace, delay=0)


    # Scene Parameters
    picasso = Painter(w, objects)

    # Play em'!
    while True:
        animator.update_step()
        picasso.paint_step()

        time.sleep(dt)


# Main Function
if __name__ == '__main__':
    scene()


# Necessary line for Tkinter
mainloop()