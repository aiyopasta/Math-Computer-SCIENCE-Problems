from tkinter import *
import numpy as np
import copy
from svgpathtools import svg2paths


# Window size
n = 10
window_w = int(2.1**n)
window_h = int(2**n)
np.set_printoptions(suppress=True)

# Tkinter Setup
root = Tk()
root.title("SVG Test")
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


def A_many(l):
    '''
        l is a list of the form (x1, y1, x2, y2, ... , xn, yn)
    '''
    assert len(l) % 2 == 0
    l_prime = copy.copy(l)

    # 'ord' stands for ordinate
    for i, ord in enumerate(l):
        if i%2==0:
            l_prime[i] += window_w/2
        else:
            l_prime[i] = -l[i] + window_h/2

    return l_prime

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





def run():
    global points_collection, should_fill_collection, _from_rgb, change_color_intensity

    # Get path strings
    paths, _ = svg2paths(
        '/Users/adityaabhyankar/Desktop/knuth.001.svg')  # Good methods for path: continuous_subpaths, iscontinuous

    # Get path string for character
    text = 'The DFA did not recognize string!'

    # For the other svg image
    # letters_map = ['C', 'G', 'A', 'B', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'Q', 'S', 'N', 'P', 'R',
    #                'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'b', 'd', 'f', 'h', 'i1', 'j1', 'k', 'l', 'g', 'a', 'c',
    #                'e', 'i2', 'j2', 'm', 't', 'q', 'n', 'o', 'p', 'r', 's', 'u', 'v', 'w', 'x', 'y', 'z']

    # Knuth Computer Modern Font
    letters_map = ['A', 'C', 'G', 'B', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Q', 'S', 'P', 'R', 'T',
                   'U',
                   'V', 'W', 'X', 'Y', 'Z', 'a', 'f', 'b', 'd', 'h', 'k', 'l', 'i1', 'j1', 'c', 'g', 's', 'e', 'i2',
                   'j2',
                   'm', 'n', 'o', 'p', 'q', 'r', '4', '7', '0', '1', '2', '3', '5', '6', '8', 't', 'u', 'v', 'w', 'x',
                   'y', 'z', '[', ']', '\\', '/', '$', '!1', '@', '\'', '#', '~', '9', '`', ';1', '=1', '-', '=2', ';2',
                   ',', '.', '!2', '%1', '*', '(', ')', '{', '}', '|', '&', '?1', '"1', '"2', '^', '+', '<', '>', ':1',
                   '%2', ':2', '?2', '_']

    char_list = []
    for i, character in enumerate(text):
        if character == 'i':
            char_list.extend(['i1', 'i2'])
        elif character == 'j':
            char_list.extend(['j1', 'j2'])
        elif character == '!':
            char_list.extend(['!1', '!2'])
        elif character == '=':
            char_list.extend(['=1', '=2'])
        elif character == '?':
            char_list.extend(['?1', '?2'])
        elif character == '%':
            char_list.extend(['%1', '%2'])
        elif character == ':':
            char_list.extend([':1', ':2'])
        else:
            char_list.append(character)

    points_collection = []
    should_fill_collection = []
    spacing = 0
    d = 20

    n = 100  # you can distribute these points according to number of parts

    for k, character in enumerate(char_list):

        if character == ' ':
            spacing += 300 / d
            continue

        index = letters_map.index(character)
        path = paths[index]
        xmin, xmax, ymin, ymax = path.bbox()

        # Need to tailor these offsets on a character by character basis.
        offset_x = xmin + (d * window_w / 2.5)
        offset_y = ymin

        if character in ['q', 'y', 'p', 'g', 'j2']:
            offset_y += (ymax - ymin) * 0.3

        if character == 'i1':
            _, _, i_ymin, i_ymax = paths[letters_map.index('i2')].bbox()
            offset_x -= 30
            offset_y -= (i_ymax - i_ymin) + 50

        if character == 'j1':
            _, _, j_ymin, j_ymax = paths[letters_map.index('j2')].bbox()
            offset_x -= 150
            offset_y -= (j_ymax - j_ymin) + 50

        if character == '!1':
            offset_y -= 200

        if character == '\'':
            offset_x += 70
            offset_y -= 700

        if character == '?1':
            offset_y -= 250

        if character == '?2':
            offset_x -= 150

        if character == '=1':
            offset_y -= 250 + 270

        if character == '=2':
            offset_y -= 270

        if character == ':1':
            offset_x -= 60
            offset_y -= 350

        if character == ':2':
            offset_x -= 60

        if character == '%1':
            offset_x -= 60

        if character == '%2':
            offset_x += 200

        subpaths = path.continuous_subpaths()
        for l, subpath in enumerate(subpaths):
            points = []
            for i in range(0, n + 1):
                p = subpath.point(i / n)  # note: this method is quite slow. best to store these beforehand.
                points.extend([((p.real - offset_x) / d) + spacing, ((p.imag - offset_y) / d)])

            should_fill_collection.append(l > 0)
            points_collection.append(points)

        if character not in ['i1', 'j1', '!1', '=1', '?1', ':1']:
            spacing += (xmax - xmin + 50) / d

    w.configure(background='black')

    for i, points in enumerate(points_collection):
        raw_color = apple_colors['lightorange']
        color = change_color_intensity(raw_color, 1)
        dark_color = change_color_intensity(raw_color, 1)

        if should_fill_collection[i]:
            w.create_polygon(*A_many(points), fill=None, outline=_from_rgb(color), smooth=0, width=0, tag='shape')
        else:
            w.create_polygon(*A_many(points), fill=None, outline=_from_rgb(color), smooth=0, width=1, tag='shape')


    # w.create_text(*A(-220, 30), text='The DFA did not recognize string!', font=('CMU Serif', 59), fill=_from_rgb(dark_color))

    w.update()


# From https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter
def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb


def change_color_intensity(rgb, p):
    return int(rgb[0]*p), int(rgb[1]*p), int(rgb[2]*p)

# Main function
if __name__ == '__main__':
    run()


# Necessary line for Tkinter
mainloop()


# The conclusion:
#
# This SVG technique works well for regular text. LaTeX is an issue because it creates SVG files with
# transformations that was difficult to parse. So it seems like we can proceed by making it so that
# the regular text can draw with the usual way, but the LaTeX draws using the ghetto method depicted
# on the whiteboard. Furthermore, if we want to morph the LaTeX into a different ParamShape object,
# we can simultaneously morph the bounding box
