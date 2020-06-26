from ghetto_manim import *
import time
from PIL import Image
import os
from svgpathtools import svg2paths
import random

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Special Shape Classes
class LShapedSquareGroup(ParamShapeGroup):
    def __init__(self, corner_square, n, color, fill_p=-1, stop=1, drawing_point_delta=0.01):
        assert n > 1
        self.n = n

        sidelen = corner_square.length
        x0, _, y0, _ = corner_square.get_bbox()

        squares = []
        for i in range(n):
            x = x0 + (sidelen * (n-1))
            y = y0 - (i * sidelen)

            squares.append(Rectangle(x, y, x+sidelen, y+sidelen, color, fill_p=fill_p, stop=1, drawing_point_delta=drawing_point_delta))

        for i in range(n-2, -1, -1):
            x = x0 + (i * sidelen)
            y = y0 - (sidelen * (n-1))

            squares.append(Rectangle(x, y, x+sidelen, y+sidelen, color, fill_p=fill_p, stop=1, drawing_point_delta=drawing_point_delta))

        super().__init__(squares, start=0, stop=stop)

    def flatten_step(self, ease, t):
        pivot_square = self.shapes[self.n - 1]
        _, x0, _, y0 = pivot_square.get_bbox()

        for i in range(self.n-1):
            square = self.shapes[i]
            square.translate_step(x0 + (square.length * (self.n-1.5-i)), y0 - (square.length / 2), ease, t)

# Time step
dt = 0.01

# Global vars
save_anim = True
stop = False
img_list = []

# File Output Setup
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
    objects = []
    n = 5
    layer_colors = [apple_colors['lightorange'], apple_colors['lightyellow'], apple_colors['lightgreen'],
                    apple_colors['lightblue'], apple_colors['lightpurple'], apple_colors['lightindigo']]

    fill_p = 0.2
    drawing_delta = 0.01
    sidelen = 25
    duplicate_spacing = (n * sidelen) + 80

    for k in range(3):
        # Creation of 1x1 square
        x0, y0 = -600 + (k * duplicate_spacing), 400
        color = apple_colors['lightred'] if k == 0 else (0, 0, 0)

        one_by_one = Rectangle(x0, y0, x0+sidelen, y0-sidelen, color, fill_p=fill_p, stop=0, drawing_point_delta=drawing_delta)
        objects.append(one_by_one)

        # Creation of larger grid squares
        spacing = 30

        for i in range(1, n+1):
            y = y0 - (sidelen * (i*(i+1)/2)) - (i * spacing)
            color = apple_colors['lightred'] if k == 0 else (0, 0, 0)

            corner_square = Rectangle(x0, y, x0+sidelen, y-sidelen, color, fill_p=fill_p, stop=0, drawing_point_delta=drawing_delta)
            objects.append(corner_square)

            for j in range(2, i+2):
                if k == 0:
                    color = layer_colors[j-2]

                L_shape = LShapedSquareGroup(corner_square, j, color, fill_p=fill_p, stop=0, drawing_point_delta=drawing_delta)
                objects.append(L_shape)

    # Creation of Assembly Bounding Box
    x0, y0 = 100, 400
    n += 1
    x1, y1 = x0 + (sidelen * ((2*n)+1)), y0 - (sidelen * (n*(n+1)/2))

    box = Rectangle(x0, y0, x1, y1, (255, 255, 255), stop=0, drawing_point_delta=0.005)
    objects.append(box)

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()

    # Draw Main Squares
    draw = empty_anim
    for i in range(n):
        for j in range(i+1):
            idx = int((i*(i+1)/2) - 1 + (j+1))
            anim = animator.add_animation(objects[idx].draw_step, [1, smooth], duration=15, parent_animation=draw, delay=0)

            if j == i:
                draw = anim

    # Draw Copy of Squares
    k = int(n*(n+1)/2)
    draw_copy = None
    for i in range(k, len(objects)-1):
        draw_copy = animator.add_animation(objects[i].draw_step, [1, smooth], duration=15, parent_animation=draw, delay=20)
        fade_in = animator.add_animation(objects[i].fade_color_step, [(255, 255, 255), smooth], duration=15, parent_animation=draw, delay=20)
        slight_movein = animator.add_animation(objects[i].translate_step, [*(objects[i].offsets_xy)+np.array([20, 0]), smooth], duration=15,
                                               parent_animation=draw, delay=20)


    # Draw Bounding Assembly Box
    box = objects[-1]
    draw_box = animator.add_animation(box.draw_step, [1, smooth], duration=30, parent_animation=draw_copy, delay=20)
    box_xmin, box_xmax, box_ymin, box_ymax = box.get_bbox()

    # Assemble Copies
    copy_assembly = None
    for i in range(n):
        dx1, dy1, dx2, dy2 = None, None, None, None
        for j in range(i+1):
            k0 = n*(n+1)/2
            raw_idx = ((i * (i + 1) / 2) - 1 + (j + 1))

            idx1 = int(k0 + raw_idx)
            idx2 = int((2*k0) + raw_idx)

            if j == 0:
                target_x1 = box_xmin + (sidelen / 2)
                target_y = box_ymax - (sidelen * (i*(i+1)/2)) - (sidelen/2)

                dx1 = target_x1 - objects[idx1].offsets_xy[0]
                dy1 = target_y - objects[idx1].offsets_xy[1]

                target_x2 = box_xmax + (sidelen / 2) - ((i+1) * sidelen)

                dx2 = target_x2 - objects[idx2].offsets_xy[0]
                dy2 = target_y - objects[idx2].offsets_xy[1]

            animator.add_animation(objects[idx1].translate_step, [*(objects[idx1].offsets_xy + np.array([dx1, dy1])), smooth],
                                   duration=50, parent_animation=draw_box, delay=0)

            copy_assembly = animator.add_animation(objects[idx2].translate_step, [*(objects[idx2].offsets_xy + np.array([dx2, dy2])), smooth],
                                   duration=50, parent_animation=draw_box, delay=0)


    # Assemble Main Corner Squares
    for i in range(n):
        for j in range(i+1):
            idx = int(((i * (i + 1) / 2) - 1 + (j + 1)))

            if j == 0:
                target_y = box_ymin + (sidelen / 2) + (sidelen * i)
                animator.add_animation(objects[idx].translate_step, [box.center_x, target_y, smooth], duration=40,
                                       parent_animation=copy_assembly, delay=0)
            else:
                animator.add_animation(objects[idx].flatten_step, [smooth], duration=40, parent_animation=copy_assembly, delay=30)

                extra_offsets = objects[idx].shapes[int((len(objects[idx].shapes) + 1) / 2) - 1].offsets_xy
                target_y = box_ymax - extra_offsets[1] + (sidelen / 2) - (sidelen * (((n-j-1)*(n-j)/2) + 1)) - (sidelen * (i-j))
                target_x = box.center_x - extra_offsets[0]

                animator.add_animation(objects[idx].translate_step, [target_x, target_y, smooth], duration=40, parent_animation=copy_assembly, delay=60)


    # Create Painter Object
    picasso = Painter(w, objects)

    # Run Scene!
    while not stop:
        if save_anim:
            filename = '/frame' + str(animator.current_frame)
            w.postscript(file=ps_files_dir + filename + '.ps', colormode='color')
            img = Image.open(ps_files_dir + filename + '.ps')
            img_list.append(img)

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