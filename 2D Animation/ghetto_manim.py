import numpy as np
import copy
from tkinter import *
from svgpathtools import svg2paths
from tkinter.font import Font
from PIL import ImageTk, Image

# Default Window Size
n = 10
window_w = int(2.1**n)
window_h = int(2**n)

# Tkinter Setup
root = Tk()
root.title("Cewl Animations :)")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = Canvas(root, width=window_w, height=window_h)
w.pack()

# Additional Imports (some imports cause problems if imported prior to above lines of code)
from io import BytesIO
from matplotlib.mathtext import math_to_image
import matplotlib
from matplotlib import pyplot as plt

# Color Palette
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


# Helper routines
def __from_rgb__(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb


def change_color_intensity(rgb, p):
    return int(rgb[0]*p), int(rgb[1]*p), int(rgb[2]*p)


# Ease Functions
def sine_squared(t):
    return np.power(np.sin(t)*np.pi/2, 2)


def smooth(t):
    return np.power(t, 2) / (np.power(t, 2) + np.power(1 - t, 2))


# Coordinate Shift
def A(x, y):
    return np.array([x + window_w/2, -y + window_h/2])


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


# Goes from typical annoying coordinate system to (0,0) one.
def A_inv(x, y):
    return np.array([-window_w/2 + x, window_h/2 - y])


# Primitive shapes / Objects to animate
# General Purpose Parametric Shape Class to extend
class ParamShape:
    def __init__(self, curve, color, fill_p=-1, start=0, stop=1, offset_x=0, offset_y=0, offset_theta=0,
                 anchor_point_x=0, anchor_point_y=0, offset_sx=1, offset_sy=1, drawing_point_delta=0.01):
        '''
            curve: a parametric function r: R -> R^2. Must be defined everywhere on t∈[0, 1].
            offset_x, offset_y: initial positional offsets
            offset_theta, offset_sx, offset_sy: initial rotational / scaling offsets (w.r.t anchor point)
            anchor_point_x/y: dictates what the offset_theta is with respect to. (Not scaling, for now.)
                              NOTE: anchor_point location is WRT local coordinates. For example,
                              to rotate / scale a rectangle from its top left corner, set the anchor
                              point to be (-length/2, +width/2).
            color: (r, g, b) of the border of the shape.
            fill_p: %-value by which the fill color will be changed, w.r.t the border color.
                    NOTE: fill_p < 0 means fill color is transparent.
            start: the lower bound on domain of r(t).
            stop: the upper bound on domain of r(t).

            e.g. If start=0.25 & stop=0.5, we only consider the points {r(t) : 0.25 <= t <= 0.5}.
        '''

        self.curve = curve
        self.color = color
        self.p = fill_p
        self.bounds = [start, stop]

        self.offsets_xy = np.array([offset_x, offset_y])

        self.offset_theta = offset_theta
        self.rot_matrix = self.__rotation_matrix__(offset_theta)
        self.anchor_point = np.array([anchor_point_x, anchor_point_y])

        self.offset_sxy = np.array([offset_sx, offset_sy])
        self.scale_matrix = self.__scale_matrix__(offset_sx, offset_sy)

        self.drawing_point_delta = drawing_point_delta

        # Additional Params
        self.n_subobjects = 1  # We call the non-group ParamShape a subobject of itself.
        self.text_array = None

    def add_text(self, text, colors_array, font_family, font_size):
        '''
            text: String which is formatted exactly in the following way:
                  – *ci at the beginning of a substring denotes the color of the entire substring until another
                    *cj is read, where i and j are indices in colors_array.
                  - *n anywhere in the string means to begin the rest of the string on a new line
                  – String in between two $ signs represents a LaTeX script.

                  Example String: '*c0 The *c1 graph *c0 of the function $f(x)=\\sin{\\pi}$ clearly *n matches the
                                   trajectory of the *c2 pendulum $P$.'

            colors_array: Array consisting of all colors used in the text given. The *ci's in the text call the ith
                          color in this colors_array.
            font_family: font name. only applies to non-LaTeX text.
            font_size: in pt
            ____

            Method generates a text_array object which is structured in the following way:
                – Each entry is a tuple (String or PhotoImage, color, location)
                – First entry is an image if the contents are in LaTeX; otherwise, it'll be a string.
                – Second entry will be None if first entry contains an image. Otherwise, it'll contain an rgb tuple.
                – Third entry will be an np.array([loc_x, loc_y]) containing where to place the image or text, based
                  on the assumption that both will be anchored 'W' in the Tkinter canvas.
        '''

        assert self.text_array is None
        self.text_array = []

        chunks = text.split('*')
        assert chunks[0] == ''

        # TODO: Implement
        pass


    # Update Methods
    def translate_step(self, target_x, target_y, ease, t):
        '''
            prior_x and prior_y: the coordinates for the  offset in x and y directions.
            target_x and target_y: the target coordinates for the offset in x and y directions.
            ease: ease function for the translation.
            t: parametrizes the translation from 0 to 1.
        '''

        if t == 0:
            self.prior_x = self.offsets_xy[0]
            self.prior_y = self.offsets_xy[1]

        t = ease(t)
        self.offsets_xy[0] = ((1-t)*self.prior_x) + (t*target_x)
        self.offsets_xy[1] = ((1-t)*self.prior_y) + (t*target_y)

    def rotate_step(self, target_theta, ease, t):
        if t == 0:
            self.prior_theta = self.offset_theta

        t = ease(t)
        self.offset_theta = ((1-t)*self.prior_theta) + (t*target_theta)
        self.rot_matrix = self.__rotation_matrix__(self.offset_theta)

    def scale_step(self, target_sx, target_sy, ease, t):
        if t == 0:
            self.prior_sx = self.offset_sxy[0]
            self.prior_sy = self.offset_sxy[1]

        t = ease(t)
        sx_new = ((1-t)*self.prior_sx) + (t*target_sx)
        sy_new = ((1-t)*self.prior_sy) + (t*target_sy)
        self.offset_sxy = np.array([sx_new, sy_new])
        self.scale_matrix = self.__scale_matrix__(*self.offset_sxy)

    def draw_step(self, target_ubound, ease, t):
        if t == 0:
            self.prior_ubound = self.bounds[1]

            if self.p != -1:
                self.target_fill_p = self.p

        t = ease(t)
        self.bounds[1] = ((1-t)*self.prior_ubound) + (t*target_ubound)

        if self.p != -1:
            self.p = t*self.target_fill_p

    def undraw(self, target_lbound, ease, t):
        if t == 0:
            self.prior_lbound = self.bounds[0]

        t = ease(t)
        self.bounds[0] = ((1-t)*self.prior_lbound) + (t*target_lbound)

    def morph_step(self, target_shape, ease, t):
        assert isinstance(target_shape, ParamShape)

        # Figure out a way to fix method to allow for >1 morphs in succession. Currently throws exception.
        if t == 0:
            self.prior_curve = self.curve

        t = ease(t)
        self.curve = lambda x: ((1-t)*self.prior_curve(x)) + (t*target_shape.curve(x))

        self.translate_step(*target_shape.offsets_xy, ease, t)
        self.rotate_step(target_shape.offset_theta, ease, t)
        self.scale_step(*target_shape.offset_sxy, ease, t)
        self.fade_color_step(target_shape.color, ease, t)
        # self.fade_definition(target_shape.drawing_point_delta, ease, t)

    def fade_definition(self, target_drawing_point_delta, ease, t):
        if t == 0:
            self.prior_drawing_point_delta = self.drawing_point_delta

        t = ease(t)
        self.drawing_point_delta = ((1-t)*self.prior_drawing_point_delta) + (t*target_drawing_point_delta)


    def fade_color_step(self, target_color, ease, t):
        '''target_color must be in rgb'''

        if t == 0:
            self.prior_color = self.color

        t = ease(t)

        dr = target_color[0] - self.prior_color[0]
        db = target_color[1] - self.prior_color[1]
        dg = target_color[2] - self.prior_color[2]
        self.color = int(self.prior_color[0] + (t*dr)), int(self.prior_color[1] + (t*db)), int(self.prior_color[2] + (t*dg))

    def fade_fill_step(self, target_p, ease, t):
        assert self.p >= 0
        if t == 0:
            self.prior_p = self.p

        t = ease(t)
        self.p = ((1-t)*self.prior_p) + (t*target_p)

    def drop_exit(self, direction, t):
        '''
            direction: `up', `down', `left', or `right'
        '''
        if t == 0:
            v_mag = 30.
            g_mag = 5.

            if direction == 'up':
                self.v = np.array([0, -v_mag])
            elif direction == 'down':
                self.v = np.array([0, v_mag])
            elif direction == 'left':
                self.v = np.array([v_mag, 0])
            elif direction == 'right':
                self.v = np.array([-v_mag, 0])

            self.a = -g_mag * (self.v / np.linalg.norm(self.v))

        self.v = np.add(self.v, self.a)
        self.offsets_xy = np.add(self.offsets_xy, self.v)

    def abrupt_removal(self, t):
        self.bounds = [0, 0]


    # ... add more methods

    # Generating sample points for draw / interpolation
    def get_drawing_points(self):
        dt = self.drawing_point_delta

        points = []
        for t in np.arange(self.bounds[0], self.bounds[1]+dt, dt):
            points.extend(self.__get_drawing_point__(min(t, 1)))

        yield points

    def __get_drawing_point__(self, t):
        v = self.curve(t)
        return np.matmul(self.rot_matrix, np.matmul(self.scale_matrix, v) - self.anchor_point) \
                         + self.anchor_point + self.offsets_xy

    @staticmethod
    def __rotation_matrix__(theta):
        return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T

    @staticmethod
    def __scale_matrix__(sx, sy):
        return np.array([[sx, 0], [0, sy]]).T


# Primitive shapes
class Ellipse(ParamShape):
    def __init__(self, center_x, center_y, a, b, color, fill_p=-1, start=0, stop=1, rot_theta=0,
                 anchor_x=0, anchor_y=0, drawing_point_delta=0.01):
        # a, b are parameters of the curve (major / minor axes)
        self.a = a
        self.b = b
        super().__init__(self.curve, color, fill_p, start, stop, center_x, center_y, rot_theta, anchor_x, anchor_y,
                         1, 1, drawing_point_delta=drawing_point_delta)

    def curve(self, t):
        t = 2*np.pi*t
        return np.array([self.a*np.cos(t), self.b*np.sin(t)])


class Circle(Ellipse):
    def __init__(self, center_x, center_y, radius, color, fill_p=-1, start=0, stop=1, rot_theta=0,
                 anchor_x=0, anchor_y=0, drawing_point_delta=0.01):
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y
        super().__init__(center_x, center_y, radius, radius, color, fill_p, start, stop, rot_theta,
                         anchor_x, anchor_y, drawing_point_delta=drawing_point_delta)


class Rectangle(ParamShape):
    def __init__(self, topleft_x, topleft_y, botright_x, botright_y, color, fill_p=-1, start=0, stop=1, rot_theta=0,
                 anchor_x=0, anchor_y=0, drawing_point_delta=0.01):

        self.center_x = (topleft_x + botright_x) / 2.
        self.center_y = (topleft_y + botright_y) / 2.
        self.length = abs(topleft_x - botright_x)
        self.height = abs(topleft_y - botright_y)
        super().__init__(self.curve, color, fill_p, start, stop, self.center_x, self.center_y, rot_theta,
                         anchor_x, anchor_y, 1, 1, drawing_point_delta=drawing_point_delta)

    def curve(self, t):
        t = 2 * np.pi * t
        sec = lambda x: 1. / np.cos(x)

        dist = sec(t - (np.pi / 2 * np.floor(2. / np.pi * (t + (np.pi / 4)))))
        return np.array([self.length*np.cos(t)*dist/2, self.height*np.sin(t)*dist/2])

    def get_bbox(self):
        xmin = self.center_x - self.length/2
        xmax = self.center_x + self.length/2
        ymin = self.center_y - self.height/2
        ymax = self.center_y + self.height/2

        return xmin, xmax, ymin, ymax

    def small_bounce_around(self, screen, t):  # note that this assumes square is tiny
        pass


class Triangle(ParamShape):
    def __init__(self, center_x, center_y, color, fill_p=-1, start=0, stop=1, rot_theta=0,
                 anchor_x=0, anchor_y=0, scale_x=1, scale_y=1, drawing_point_delta=0.01):
        self.scale_x = scale_x
        self.scale_y = scale_y
        super().__init__(self.curve, color, fill_p, start, stop, center_x, center_y, rot_theta,
                         anchor_x, anchor_y, 1, 1, drawing_point_delta=drawing_point_delta)

    def curve(self, t):
        t = 2 * np.pi * t
        sec = lambda x: 1. / np.cos(x)

        dist = sec(t - (np.pi / 3) - (2*np.pi/3 * np.floor(3*t / (2*np.pi))))
        return np.array([self.scale_x*np.cos(t)*dist/2, self.scale_y*np.sin(t)*dist/2])


class Cardoid(ParamShape):
    def __init__(self, color, center_x, center_y, fill_p=-1, scale=1, start=0, stop=1, rot_theta=0,
                 anchor_x=0, anchor_y=0, drawing_point_delta=0.01):
        super().__init__(self.curve, color, fill_p, start, stop, center_x, center_y, rot_theta, anchor_x, anchor_y,
                         scale, scale, drawing_point_delta=drawing_point_delta)

    def curve(self, t):
        t = 2*np.pi*t
        return np.array([40*np.cos(t)*(1 - 2*np.cos(t)), 40*np.sin(t)*(1 - 2*np.cos(t))])

class CurvedLine(ParamShape):
    def __init__(self, x0, y0, x1, y1, color, fill_p=-1, curve_place=1, curve_amount=0, start=0, stop=1,
                 drawing_point_delta=0.01):
        '''
            (x0, y0): point where the parametrized curve starts
            (x1, y1): point where the parametrized curve stops
            curve_place: Value between 0 and 1 which dictates which part of the curve is most curved.
            curve_amount: Number which dictates how much curvature there is.
                          NOTE: If this number is negative, we curve down. Otherwise, we curve up. 0 means straight line.
        '''

        self.p1 = np.array([x0, y0])
        self.p2 = np.array([x1, y1])
        self.t_k = curve_place
        self.curve_amount = curve_amount
        super().__init__(self.curve, color, fill_p, start, stop, offset_x=0, offset_y=0, offset_theta=0,
                         anchor_point_x=0, anchor_point_y=0, offset_sx=1, offset_sy=1,
                         drawing_point_delta=drawing_point_delta)

        self.__calculate_p_mid__()

    # Demonstration: https://www.desmos.com/calculator/jk7n16e52a
    def __calculate_p_mid__(self):
        # Calculate perpendicular line parametric equation.
        v = self.p2 - self.p1
        temp = copy.copy(v)

        r_0 = (v * self.t_k) + self.p1
        if self.curve_amount >= 0:
            v[0] = -temp[1]
            v[1] = temp[0]
        elif self.curve_amount < 0:
            v[0] = temp[1]
            v[1] = -temp[0]

        v = v / np.linalg.norm(self.p2 - self.p1)

        # Calculate third point and Bezier curve.
        self.p_mid = (abs(self.curve_amount) * v * self.t_k) + r_0

    def curve(self, t):
        # Calculate third point and Bezier curve.
        p3 = self.p_mid

        line1 = lambda x: ((1-x)*self.p1) + (x*p3)
        line2 = lambda x: ((1-x)*p3) + (x*self.p2)

        # Return point on Bezier curve at t.
        return ((1-t)*line1(t)) + (t*line2(t))

    def curve_angle(self):
        m = self.p2 - self.p_mid
        return np.arctan2(m[1], m[0])

    def translate_tip(self, target_tip, ease, t):
        if t == 0:
            self.prior_p2 = copy.copy(self.p2)

        t = ease(t)
        self.p2 = ((1-t)*self.prior_p2) + (t*target_tip)
        self.__calculate_p_mid__()

    def translate_tail(self, target_tail, ease, t):
        if t == 0:
            self.prior_p1 = copy.copy(self.p1)

        t = ease(t)
        self.p1 = ((1-t)*self.prior_p1) + (t*target_tail)


# Animation is a node in a tree structure
class Animation:
    def __init__(self, update_step_fn=None, fn_params=None, duration=1, parent_animation=None, delay=0):
        '''
            update_step_fn:  for example, square.translate, circle.expand, pendulum.swing (square, circle, pendulum are
                             ParamShape objects).  NOTE: The ONLY Animation for which this parameter should be NONE for
                             is the empty animation, which is the root of the tree.

            fn_params: additional parameters for the update_step_fn, in a list. for example: [dx, dy, ease].
                       NOTE: ALL fn_params will be sent their respective t parameter, which parametrizes the animation.
                       NOTE: The ONLY Animode for which this parameter should be NONE for is the empty animation,
                             which is the root of the tree.

            duration: the number of seconds to play the animation.

            parent_animation: Animation with which the subsequent delay parameter is stated with respect to.
                              NOTE: The ONLY Animode for which this parameter should be NONE for is the
                              empty animation, which is the root of the tree.

            delay: the number of seconds to wait after the end of the parent_animation, prior to starting this one.

        '''

        global dt

        self.update_step_fn = update_step_fn
        self.extra_fn_params = fn_params
        self.parent = parent_animation
        self.children = []

        if self.parent is not None:
            self.parent.children.append(self)
            self.start_frame = self.parent.end_frame + delay
        else:
            self.start_frame = delay

        self.duration = duration
        self.end_frame = self.start_frame + self.duration

        # Only for when this Animation is on the priority queue / heap
        self.h_parent = None
        self.h_right = None
        self.h_left = None

    def update_step(self, t):
        # t is the variable which parametrizes the animation
        if self.extra_fn_params is None:
            return

        # print(self.extra_fn_params)
        self.update_step_fn(*self.extra_fn_params, t)


class ParamShapeGroup(ParamShape):
    def __init__(self, param_shapes, start=0, stop=0, global_x=0, global_y=0, global_theta=0,
                 global_anchor_x=0, global_anchor_y=0, global_sx=1, global_sy=1, draw_time_weights='equal'):
        '''
            draw_time_weights: list of % of draw time given to each shape. ith draw time %age is given to the ith shape
                               in the list param_shapes (i.e. order of shapes matters). For example, draw_time_weights
                               [0.7, 0.2, 0.1] allot 70% of the drawing time to shape 1, 20% to shape 2, and 10% to
                               shape 3.

                               NOTE: draw_time_weights can be 'equal', or some hard-coded list like [0.4, 0.6],
                                     or some function defined on the interval [0,1].
        '''

        if draw_time_weights == 'equal':
            self.draw_time_weights = np.array([1./len(param_shapes) for _ in param_shapes])
        elif callable(draw_time_weights):
            # TODO: Implement
            pass
        else:
            self.draw_time_weights = draw_time_weights

        assert abs(sum(self.draw_time_weights) - 1) < 0.0001

        self.shapes = param_shapes
        draw_deltas = [shape.drawing_point_delta for shape in param_shapes]
        draw_time_delta = min(draw_deltas) / len(draw_deltas)  # lazy solution

        # Note that the `global color' for the entire group is None, as each of the shapes have their own color.
        # TODO: CHANGE THE HARDCODED LIGHTINDIGO TO NONE
        super().__init__(curve=self.curve, color=apple_colors['lightindigo'], start=start, stop=stop, offset_x=global_x, offset_y=global_y,
                         offset_theta=global_theta, anchor_point_x=global_anchor_x, anchor_point_y=global_anchor_y,
                         offset_sx=global_sx, offset_sy=global_sy, drawing_point_delta=draw_time_delta)

        self.n_subobjects = len(self.shapes)

    def curve(self, t):
        t, anim_slot = self.__remapped_t__(t)
        return self.shapes[anim_slot].curve(t)

    def __remapped_t__(self, t):
        tot = len(self.draw_time_weights)

        # Deciding which one piecewise curve to look at.
        # [0.1, 0.7, 0.1, 0.5, 0.5] --> [0, 0.1, 0.8, 0.9, 0.95, 1]
        anim_slot = None
        running_total = self.draw_time_weights[0]

        for i in range(1, tot + 1):
            if t < running_total:
                anim_slot = i - 1
                break
            elif abs(t - running_total) < 0.00001 and i == tot:
                anim_slot = i - 1
                break

            # If you got an error here, it means its a rounding error. Check the value of t vs. running_total.
            # print(t, running_total)
            running_total += self.draw_time_weights[i]

        # float() operation is there only to suppress some stupid highlight it is doing.
        endpoints = [running_total - float(self.draw_time_weights[anim_slot]), running_total]

        # We use these endpoints to transform the `t' into the interval [i_start, i_stop].
        m = (self.shapes[anim_slot].bounds[1] - self.shapes[anim_slot].bounds[0]) / (endpoints[1] - endpoints[0])
        b = self.shapes[anim_slot].bounds[0] - (m * endpoints[0])
        t = (m * t) + b

        return t, anim_slot

    def morph_step(self, target_shape, ease, t):
        for k, obj in enumerate(self.shapes):
            obj.morph_step(target_shape, ease, t)

            # if t > 0.99 and k > 0:
            #     obj.bounds[1] = 0

            # if t > 0.99 and k == 0:
            #     obj.drawing_point_delta = target_shape.drawing_point_delta

        super().morph_step(target_shape, ease, t)

    def fade_color_step(self, target_color, ease, t):
        for obj in self.shapes:
            obj.fade_color_step(target_color, ease, t)

    # Generating sample points for draw / interpolation
    def get_drawing_points(self):
        dt = self.drawing_point_delta

        point_set = []
        prev_shape_num = 0
        prev_subpath_num = 0
        for t in np.arange(self.bounds[0], self.bounds[1]+dt, dt):
            t_prime, curr_shape_num = self.__remapped_t__(min(t, 1))

            # Apply local shape-wise transformations
            v = self.shapes[curr_shape_num].__get_drawing_point__(min(t_prime, 1))

            # Apply global transformations
            v = np.matmul(self.rot_matrix, np.matmul(self.scale_matrix, v) - self.anchor_point) + self.anchor_point + self.offsets_xy

            # Special Glyph Accommodation
            curr_subpath_num = 0
            if isinstance(self.shapes[curr_shape_num], Glyph):
                curr_subpath_num = self.shapes[curr_shape_num].current_subpath_num

            # Should yield accumulated point_set?
            if prev_shape_num != curr_shape_num or t >= 1:
                prev_shape = self.shapes[prev_shape_num]
                if (not isinstance(prev_shape, CurvedLine) and prev_shape.p != -1) or isinstance(prev_shape, Glyph)\
                        and not prev_shape.morphed:
                    point_set.extend(point_set[0:2])

                yield prev_shape_num, point_set
                point_set = [*v]
                prev_shape_num = curr_shape_num
                prev_subpath_num = 0

            elif curr_subpath_num != prev_subpath_num:
                if not self.shapes[prev_shape_num].morphed:
                    point_set.extend(point_set[0:2])
                yield curr_shape_num, point_set
                point_set = [*v]
                prev_subpath_num = curr_subpath_num

            else:
                point_set.extend(v)

        yield prev_shape_num, point_set


class Glyph(ParamShape):
    def __init__(self, character, pseudo_font_size, color, start=0, stop=1, left_x=0, bottom_y=0,
                 angle=0, anchor_point_x=0, anchor_point_y=0, offset_sx=1, offset_sy=1, drawing_points_delta=0.01):
        '''
            character: character to display
            font_size: size of font
            n_points: number of points to sample across each svg image path.
        '''

        # Knuth Computer Modern Font
        self.letters_map = ['A', 'C', 'G', 'B', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Q', 'S', 'P',
                            'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'f', 'b', 'd', 'h', 'k', 'l', 'i1', 'j1', 'c',
                            'g', 's', 'e', 'i2', 'j2', 'm', 'n', 'o', 'p', 'q', 'r', '4', '7', '0', '1', '2', '3', '5',
                            '6', '8', 't', 'u', 'v', 'w', 'x', 'y', 'z', '[', ']', '\\', '/', '$', '!1', '@', '\'', '#',
                            '~', '9', '`', ';1', '=1', '-', '=2', ';2', ',', '.', '!2', '%1', '*', '(', ')', '{', '}',
                            '|', '&', '?1', '"1', '"2', '^', '+', '<', '>', ':1', '%2', ':2', '?2', '_']

        self.character = character

        # Use svg image to extract paths
        self.paths, _ = svg2paths('/Users/adityaabhyankar/Desktop/knuth.001.svg')
        self.path = self.paths[self.letters_map.index(self.character)]
        self.subpaths = self.path.continuous_subpaths()

        self.d = pseudo_font_size

        super().__init__(self.curve, color, fill_p=-1, start=start, stop=stop, offset_x=left_x, offset_y=bottom_y,
                         offset_theta=angle, anchor_point_x=anchor_point_x, anchor_point_y=anchor_point_y,
                         offset_sx=offset_sx, offset_sy=offset_sy, drawing_point_delta=drawing_points_delta)

        self.__calculate_possible_extra_offsets__()
        self.current_subpath_num = 0

        # Create time markers
        self.time_markers = [1]  # when to move on to drawing the next subpath
        if len(self.subpaths) > 1:
            t0 = 0.7
            self.time_markers = [t0]
            for i in range(len(self.subpaths) - 1):
                self.time_markers.append(((1 - t0) / (len(self.subpaths) - 1)) + self.time_markers[-1])

        # Additional Params
        self.morphed = False

    def __calculate_possible_extra_offsets__(self):
        xmin, xmax, ymin, ymax = self.path.bbox()

        # Need to tailor these offsets on a character by character basis.
        extra_offset_x = xmin
        extra_offset_y = ymin

        if self.character in ['q', 'y', 'p', 'g', 'j2']:
            extra_offset_y += (ymax - ymin) * 0.3

        if self.character == 'i1':
            _, _, i_ymin, i_ymax = self.paths[self.letters_map.index('i2')].bbox()

            extra_offset_x -= 30
            extra_offset_y -= (i_ymax - i_ymin) + 50

        if self.character == 'j1':
            _, _, j_ymin, j_ymax = self.paths[self.letters_map.index('j2')].bbox()

            extra_offset_x -= 150
            extra_offset_y -= (j_ymax - j_ymin) + 50

        if self.character == '!1':
            extra_offset_y -= 200

        if self.character == '\'':
            extra_offset_x += 70
            extra_offset_y -= 700

        if self.character == '?1':
            extra_offset_y -= 250

        if self.character == '?2':
            extra_offset_x -= 150

        if self.character == '=1':
            extra_offset_y -= 250 + 270

        if self.character == '=2':
            extra_offset_y -= 270

        if self.character == ':1':
            extra_offset_x -= 60
            extra_offset_y -= 350

        if self.character == ':2':
            extra_offset_x -= 60

        if self.character == '%1':
            extra_offset_x -= 60

        if self.character == '%2':
            extra_offset_x += 200

        self.extra_offset_x = extra_offset_x
        self.extra_offset_y = extra_offset_y

    def curve(self, t):
        p = self.subpaths[self.current_subpath_num].point(t)
        return np.array([(p.real - self.extra_offset_x) / self.d, (p.imag - self.extra_offset_y) / self.d])

    def __remapped_t__(self, t):
        tm = [0]; tm.extend(self.time_markers)
        slope = 1. / (tm[self.current_subpath_num + 1] - tm[self.current_subpath_num])
        yint = -slope * tm[self.current_subpath_num]

        if self.current_subpath_num > 0 and self.morphed:
            slope = yint = 0

        return (slope * t) + yint

    def morph_step(self, target_shape, ease, t):
        super().morph_step(target_shape, ease, t)

        if t > 0.45:
            self.morphed = True

    def get_drawing_points(self):
        dt = self.drawing_point_delta
        point_set = []

        self.current_subpath_num = 0
        for t in np.arange(self.bounds[0], self.bounds[1]+dt, dt):
            v = self.__get_drawing_point__(min(t, 1))
            point_set.extend(v)

            if t + dt > self.time_markers[self.current_subpath_num]:
                # self.current_subpath_num += 1
                point_set.extend(point_set[0:2])
                yield point_set
                point_set = []

        yield point_set

    def __get_drawing_point__(self, t):
        tm = [0]; tm.extend(self.time_markers)
        for k, marker in enumerate(tm):
            if t > marker:
                self.current_subpath_num = k
            elif t == 0:
                self.current_subpath_num = k
                break
            else:
                break

        t_prime = self.__remapped_t__(t)
        return super().__get_drawing_point__(min(t_prime, 1))

    def get_right_x(self):
        return ((self.path.bbox()[1] - self.extra_offset_x) / self.d) + self.offsets_xy[0]


class Text(ParamShapeGroup):  # implement to include LaTeX too.
    def __init__(self, text, pseudo_font_size, color, spacing=30, start=0, stop=1, left_x=0, bottom_y=0,
                 angle=0, anchor_point_x=0, anchor_point_y=0, offset_sx=1, offset_sy=1, drawing_delta_per_glyph=0.01):

        self.text = text

        space_adj = 0
        glyphs = []
        for k, char in enumerate(self.__get_char_list__()):
            if char == ' ':
                space_adj = spacing * 3
                continue

            if len(char)==2 and '2' in char:
                space_adj = -spacing * 3

            x = 0
            if k != 0:
                x = glyphs[-1].get_right_x() + spacing + space_adj

            if space_adj != 0:
                space_adj = 0

            glyph = Glyph(char, pseudo_font_size, color, stop=1, left_x=x, bottom_y=bottom_y,
                          drawing_points_delta=drawing_delta_per_glyph)
            glyphs.append(glyph)

        super().__init__(glyphs, start=start, stop=stop, global_x=left_x, global_y=bottom_y, global_theta=angle,
                         global_anchor_x=anchor_point_x, global_anchor_y=anchor_point_y, global_sx=offset_sx,
                         global_sy=offset_sy)

    def __get_char_list__(self):
        char_list = []
        for i, character in enumerate(self.text):
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

        return char_list


# Animation Tree Class
class Animator:
    def __init__(self):
        self.root = Animation(duration=10)  # the do-nothing animation
        self.playing_animations = []  # animations currently being played
        self.queue = self.AnimationQueue()  # after 11+ goddamn hours of blood sweat and tears. people died, you know

        self.current_frame = 0
        self.playing = False

    def get_root(self):
        return self.root

    def add_animation(self, update_fn, extra_fn_params, duration, parent_animation, delay):
        '''
            update_fn: ``update_step_fn'' parameter for Animation object
            extra_fn_params: ``fn_params'' parameter for Animation object
            duration: ``duration'' parameter for Animation object
            parent_animation: ``parent_animation'' parameter for Animation object
            delay: ``delay'' parameter for Animation object
        '''

        anim = Animation(update_step_fn=update_fn, fn_params=extra_fn_params, duration=duration,
                         parent_animation=parent_animation, delay=delay)

        return anim

    # Actual Animation. Should be only repeated called at each new time step.
    # Should only be called after entire animation tree has been built before hand.
    def update_step(self):
        global dt

        if not self.playing:
            self.playing = True
            self.queue.add_node(self.root)

        # See if first thing(s) in the queue need to be added to currently playing anims
        while self.queue.root is not None and self.queue.next_start_frame() <= self.current_frame:
            # The second entry in the array is the initial value of the `t' which parametrizes the animation.
            new_to_play = [self.queue.extract_first(), 0]
            self.playing_animations.append(new_to_play)

        # Play a step of all currently playing anims, and as you play each, check if any are finished.
        # If so, then add all its children to the queue, and discard it from the currently playing anims list.
        for i, anim_tup in enumerate(self.playing_animations):
            anim = anim_tup[0]
            t = anim_tup[1]
            anim.update_step(t)

            if t == 1:
                for child in anim.children:
                    self.queue.add_node(child)

                self.playing_animations.remove(anim_tup)
                continue

            total_frame_duration = anim.duration
            self.playing_animations[i][1] = min(1, t + (1. / total_frame_duration))

        self.current_frame += 1

    # Bad Priority Queue Implementation class
    class AnimationQueue:  # min-heap structure for maintaining sorted list of waiting animations
        '''
            Priority queue for animations waiting to be played next. Implemented by min-heap,
            organized by earliest start_frame of Animations. (Perhaps in the future, can reimplement
            this to be an implicit data-structure; i.e. using an array that simulates the bin. tree
            of the heap).

            root: the Animation which is the root of the heap (earliest start frame).
        '''

        def __init__(self):
            self.root = None
            self.leaf = self.root  # leaf node on the bottom level to the most right
            self.next_parent = self.root  # parent node for which next added node will be the child

        def next_start_frame(self):
            return self.root.start_frame

        def add_node(self, node):  # O(log n), total two main O(log(n)) steps: heapify and bad coding.
            if self.root is None:
                self.root = node
                self.leaf = node
                self.next_parent = node
            else:
                # First we insert it into the next available spot
                node.h_parent = self.next_parent
                self.leaf = node

                if self.next_parent.h_left is None:
                    self.next_parent.h_left = node
                else:
                    self.next_parent.h_right = node

                    # Updating next_parent
                    ancestor = self.next_parent
                    count = 0
                    left_move = False
                    while ancestor != self.root:
                        temp = ancestor
                        ancestor = ancestor.h_parent
                        count += 1

                        if ancestor.h_left == temp:  # if we made an upwards right move
                            left_move = True
                            break

                    if ancestor == self.root and not left_move:
                        self.next_parent = self.left_most_node()
                    else:
                        ancestor = ancestor.h_right
                        while count - 1 > 0:
                            ancestor = ancestor.h_left
                            count -= 1

                        self.next_parent = ancestor

                self.bubble_up()

        def extract_first(self):
            prev_root = copy.copy(self.root)
            prev_leaf = copy.copy(self.leaf)
            is_prev_leaf_right = 0

            if self.root == self.leaf:
                self.root = None
            else:
                # Updating next_parent and leaf
                if self.leaf.h_parent.h_right == self.leaf:
                    self.leaf = self.leaf.h_parent.h_left
                    self.next_parent = self.leaf.h_parent
                    is_prev_leaf_right = 1
                else:
                    ancestor = self.next_parent
                    count = 0
                    left_move = False
                    while ancestor != self.root:
                        temp = ancestor
                        ancestor = ancestor.h_parent
                        count += 1

                        if ancestor.h_right == temp:  # if we made an upwards left move
                            left_move = True
                            break

                    if ancestor == self.root and not left_move:
                        self.leaf = self.right_most_node()
                        self.next_parent = self.left_most_node().h_parent
                    else:
                        ancestor = ancestor.h_left
                        while count > 0:
                            ancestor = ancestor.h_right
                            count -= 1

                        self.leaf = ancestor

                # print('e', self.leaf)

                # Replacing the root
                if self.root == self.next_parent:
                    self.next_parent = prev_leaf

                if self.root == self.leaf:
                    self.leaf = prev_leaf

                self.root = prev_leaf

                if prev_root.h_right == prev_leaf.h_parent.h_right and is_prev_leaf_right:
                    self.root.h_right = None
                else:
                    self.root.h_right = prev_root.h_right

                    if prev_root.h_right is not None:
                        prev_root.h_right.h_parent = self.root

                if prev_root.h_left == prev_leaf.h_parent.h_left and not is_prev_leaf_right:
                    self.root.h_left = None
                else:
                    self.root.h_left = prev_root.h_left

                    if prev_root.h_left is not None:
                        prev_root.h_left.h_parent = self.root

                # Making leaf the new root, and erasing leaf out of existence
                if is_prev_leaf_right:
                    prev_leaf.h_parent.h_right = None
                else:
                    prev_leaf.h_parent.h_left = None

                self.root.h_parent = None

                self.sift_down()
            return prev_root

        def bubble_up(self):
            node = self.leaf
            while node != self.root and node.start_frame < node.h_parent.start_frame:
                self.switch_nodes(node.h_parent, node)

        def sift_down(self):
            node = self.root

            while not self.is_leaf(node):
                if node.h_right is None:
                    if node.start_frame >= node.h_left.start_frame:
                        self.switch_nodes(node, node.h_left)
                        continue
                    else:
                        break

                if node.h_left is None:
                    if node.start_frame >= node.h_right.start_frame:
                        self.switch_nodes(node, node.h_right)
                        continue
                    else:
                        break

                # If we've reached this part, it means node has 2 children.
                smaller = None
                if node.h_left.start_frame <= node.h_right.start_frame:
                    smaller = node.h_left
                else:
                    smaller = node.h_right

                if node.start_frame >= smaller.start_frame:
                    self.switch_nodes(node, smaller)
                else:
                    break

        def switch_nodes(self, parent, child):
            # Special nodes: Root, leaf, next_parent
            if parent == self.root:
                self.root = child
            elif child == self.root:
                self.root = parent

            if child == self.leaf:
                self.leaf = parent

            if child == self.next_parent:
                self.next_parent = parent
            elif parent == self.next_parent:
                self.next_parent = child

            # Temporary objects
            parents_left = parent.h_left
            parents_right = parent.h_right
            grandparent = parent.h_parent
            childs_left = child.h_left
            childs_right = child.h_right

            # Grandparent and other ancestor handling
            if grandparent is not None:
                if grandparent.h_left == parent:
                    grandparent.h_left = child
                else:
                    grandparent.h_right = child

            if childs_right is not None:
                childs_right.h_parent = parent

            if childs_left is not None:
                childs_left.h_parent = parent

            if parents_left == child and parents_right is not None:
                parents_right.h_parent = child
            elif parents_right == child and parents_left is not None:
                parents_left.h_parent = child

            # Actual switch
            parent.h_parent = child
            parent.h_right = childs_right
            parent.h_left = childs_left
            child.h_parent = grandparent

            if parents_left == child:
                child.h_left = parent
                child.h_right = parents_right
            else:
                child.h_right = parent
                child.h_left = parents_left

        # O(log n) stupid methods, but who cares for now
        def left_most_node(self):
            node = self.root
            while node.h_left is not None:
                node = node.h_left

            return node

        def right_most_node(self):
            node = self.root
            while node.h_right is not None:
                node = node.h_right

            return node

        def is_leaf(self, node):
            return node.h_right is None and node.h_left is None


class Painter:
    def __init__(self, canvas, objects):
        self.w = canvas
        # self.w.configure(background='black')

        self.objects = objects[:]

        self.first_runs = []
        for obj in objects:
            sublist = []
            if isinstance(obj, Text):
                for glyph in obj.shapes:
                    sublist.extend([True] * len(glyph.subpaths))
            elif isinstance(obj, Glyph):
                sublist.extend([True]*len(obj.subpaths))
            elif hasattr(obj, 'vx'):  # If it is a DVDLogoObject
                sublist.extend([True]*3)
            else:
                sublist = [True]*obj.n_subobjects

            self.first_runs.append(sublist)

        self.w.create_rectangle(0,0,window_w,window_h, fill='black')

    # TODO: Implement painting for arbitrarily nested ParamShapeGroup objects.
    def paint_step(self):
        w = self.w

        for i, obj in enumerate(self.objects):
            for j, point_set in enumerate(obj.get_drawing_points()):
                shape = obj
                points = point_set
                if isinstance(obj, ParamShapeGroup):  # reason we don't use n_subobjects here is bc if there's just
                    shape = obj.shapes[point_set[0]]  # 1 object in the group, it won't get colored then as we'll think
                    points = point_set[1]             # its just a single ParamShape and not a ParamShapeGroup.

                if shape.bounds == [0, 0] and len(points) < 4:
                    points = [0]*4

                if len(points) >= 4:
                    if self.first_runs[i][j]:
                        self.first_runs[i][j] = False

                        if shape.p >= 0:
                            fill_color = change_color_intensity(shape.color, shape.p)

                            if hasattr(obj, 'vx') and j == 2:
                                fill_color = (0, 0, 0)

                            w.create_polygon(*A_many(points), fill=__from_rgb__(fill_color), smooth=0, width=2,
                                             tag='shape' + str(i) + str(j))

                        w.create_line(*A_many(points), fill=__from_rgb__(shape.color), smooth=0, width=2,
                                      tag='boundary' + str(i) + str(j))
                    else:
                        if shape.p >= 0:
                            fill_color = change_color_intensity(shape.color, shape.p)
                            if hasattr(obj, 'vx') and j == 2:
                                fill_color = (0, 0, 0)

                            poly = w.find_withtag('shape' + str(i) + str(j))
                            w.itemconfig(poly, fill=__from_rgb__(fill_color))
                            w.coords(poly, *A_many(points))

                        line = w.find_withtag('boundary' + str(i) + str(j))
                        w.itemconfig(line, fill=__from_rgb__(shape.color))
                        w.coords(line, *A_many(points))

        # print(self.first_runs)
        w.update()
