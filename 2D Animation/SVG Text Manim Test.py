from ghetto_manim import *
import time
from PIL import Image
import os
import cv2

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Special Shape Classes
# ---none


# Time step
dt = 0.01

# Global vars
save_anim = False
stop = False
img_list = []

# File Ouput Setup
main_dir = '/Users/adityaabhyankar/Desktop/Programming/2D Animation/output'
ps_files_dir = main_dir + '/Postscript Frames'
png_files_dir = main_dir + '/Png Frames'

i = 0
while save_anim and len(os.listdir(ps_files_dir)) != 1:
    os.remove(ps_files_dir + '/frame' + str(i) + '.ps')
    os.remove(png_files_dir + '/frame' + str(i) + '.png')
    i += 1

def scene():
    global stop, main_dir, ps_files_dir, png_files_dir, img_list

    # Object Creation
    text = Text('a', 17, apple_colors['lightblue'], spacing=10, stop=0,
                left_x=-window_w/2 + 30, bottom_y=window_h/2 - 300, drawing_delta_per_glyph=0.02)
    glyph = Glyph('g', 7, apple_colors['lightgreen'], stop=0, drawing_points_delta=0.01)
    square = Rectangle(-70, 70, 70, -70, apple_colors['lightyellow'], stop=0, drawing_point_delta=0.01)
    curve = CurvedLine(0, 0, 300, 40, apple_colors['lightindigo'], curve_place=0.5, curve_amount=200, stop=1)
    objects = [text, curve]

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()
    draw = animator.add_animation(text.draw_step, [1, smooth], duration=50, parent_animation=empty_anim, delay=0)
    draw2 = animator.add_animation(glyph.draw_step, [1, smooth], duration=50, parent_animation=empty_anim, delay=0)
    morph = animator.add_animation(text.morph_step, [curve, smooth], duration=50, parent_animation=draw, delay=0)

    # Create Painter Object
    picasso = Painter(w, objects)

    # Run Scene!
    while not stop:
        if save_anim:
            filename = '/frame' + str(animator.current_frame)
            w.postscript(file=ps_files_dir + filename + '.ps', colormode='color')
            img = Image.open(ps_files_dir + filename + '.ps')
            img_list.append(img)

        # img.save(png_files_dir + filename + '.png')

        animator.update_step()
        picasso.paint_step()
        time.sleep(dt)


def on_closing():
    global stop, main_dir, ps_files_dir, png_files_dir, img_list, save_anim
    if save_anim:
        stop = True
        img_list[1].save('out.gif', save_all=True, append_images=img_list)

    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)

# Main function
if __name__ == '__main__':
    scene()

# Necessary line for Tkinter
mainloop()