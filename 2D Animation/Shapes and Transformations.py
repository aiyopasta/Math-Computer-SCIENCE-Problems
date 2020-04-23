# The philosophy for primitive shapes:
#
# Upon creation of the shapes, there are several offset parameters that are passed to the constructor. For each
# of these parameters, we have the choice of either fundamentally changing the parametric equation of the shape
# based on the parameter, or instead storing these offsets as additional translation, rotation, or scaling of
# the original shape. For example, when creating a circle, a radius is specified. Here, we have the choice of
# either writing the parametric equation for a circle which is dependant on the radius, or we can leave
# the equation to describe the unit circle and instead store the radius in the scaling matrix of the ParamShape object.
#
# So which choices do we make? The philosophy is as follows. If the offset parameter / property feels intrinsic to
# the default nature of the shape itself prior to transforming it (e.g. radius of a circle, length/width of a
# rectangle, or perhaps the direction of an arrow), then we make the parametric equations dependent on that parameter,
# and call it the ``default shape created by these parameters''.
# Otherwise, we simply store the parameters (e.g. rotation of a square) in the transformation matrices, and say that the
# default shape can been ``transformed'' by the parameters.

from tkinter import *
import numpy as np
import time
import copy

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

# Time step
dt = 0.0005

# Goes from (0,0) coordinate system to typical annoying one.
def A(x, y):
    return np.array([x + window_w/2, -y + window_h/2])

def A_many(l):
    '''
        l is a list of the form (x1, y1, x2, y2, ... , xn, yn)
    '''
    assert len(l) % 2 == 0

    # 'ord' stands for ordinate
    for i, ord in enumerate(l):
        if i%2==0:
            l[i] += window_w/2
        else:
            l[i] = -l[i] + window_h/2

    return l

# Goes from typical annoying coordinate system to (0,0) one.
def A_inv(x, y):
    return np.array([-window_w/2 + x, window_h/2 - y])


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


# Animation tree structure
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

    # Priority Queue Implementation class
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


# Shape Model
class ParamShape:
    def __init__(self, curve, color, start=0, stop=1, offset_x=0, offset_y=0, offset_theta=0, offset_sx=1, offset_sy=1):
        '''
            curve: a parametric function r: R -> R^2. Must be defined everywhere on t∈[0, 1].
            offset_x, offset_y, offset_theta, offset_sx, offset_sy: initial offsets (NOTE: scaling is w.r.t. local coords)
            color: (r, g, b)
            start: the lower bound on domain of r(t).
            stop: the upper bound on domain of r(t).

            e.g. If start=0.25 & stop=0.5, we only consider the points {r(t) : 0.25 <= t <= 0.5}.
        '''

        self.curve = curve
        self.color = color
        self.bounds = [start, stop]

        self.offsets_xy = np.array([offset_x, offset_y])

        self.offset_theta = offset_theta
        self.rot_matrix = self.__rotation_matrix__(offset_theta)

        self.offset_sxy = np.array([offset_sx, offset_sy])
        self.scale_matrix = self.__scale_matrix__(offset_sx, offset_sy)

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

    def draw_step(self, ease, t):
        t = ease(t)
        self.bounds[1] = t

    def morph_step(self, target_shape, ease, t):
        assert isinstance(target_shape, ParamShape)

        if t == 0:
            self.prior_curve = self.curve

        t = ease(t)
        self.curve = lambda x: ((1-t)*self.prior_curve(x)) + (t*target_shape.curve(x))

    def fade_color_step(self, target_color, ease, t):
        '''target_color must be in rgb'''

        if t == 0:
            self.prior_color = self.color

        t = ease(t)

        dr = target_color[0] - self.prior_color[0]
        db = target_color[1] - self.prior_color[1]
        dg = target_color[2] - self.prior_color[2]
        self.color = int(self.prior_color[0] + (t*dr)), int(self.prior_color[1] + (t*db)), int(self.prior_color[2] + (t*dg))


    # ... add more methods

    # Generating sample points for draw / interpolation
    def get_drawing_points(self, dt):
        '''
            dt: points are sampled from t∈self.bounds at every multiple of this parameter. e.g. dt=0.001
        '''

        points = []
        for t in np.arange(self.bounds[0], self.bounds[1]+dt, dt):
            v = self.curve(min(t, 1))
            points.extend(np.matmul(self.rot_matrix, np.matmul(self.scale_matrix, v)) + self.offsets_xy)

        return points

    @staticmethod
    def __rotation_matrix__(theta):
        return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T

    @staticmethod
    def __scale_matrix__(sx, sy):
        return np.array([[sx, 0], [0, sy]]).T


# Primitive shapes
class Ellipse(ParamShape):
    def __init__(self, center_x, center_y, a, b, color, start=0, stop=1, rot_theta=0):
        # a, b are parameters of the curve (major / minor axes)
        self.a = a
        self.b = b
        super().__init__(self.curve, color, start, stop, center_x, center_y, rot_theta, 1, 1)

    def curve(self, t):
        t = 2*np.pi*t
        return np.array([self.a*np.cos(t), self.b*np.sin(t)])


class Circle(Ellipse):
    def __init__(self, center_x, center_y, radius, color, start=0, stop=1, rot_theta=0):
        super().__init__(center_x, center_y, radius, radius, color, start, stop, rot_theta)


class Rectangle(ParamShape):
    def __init__(self, topleft_x, topleft_y, botright_x, botright_y, color, start=0, stop=1, rot_theta=0):

        self.center_x = (topleft_x + botright_x) / 2.
        self.center_y = (topleft_y + botright_y) / 2.
        self.length = abs(topleft_x - botright_x)
        self.height = abs(topleft_y - botright_y)
        super().__init__(self.curve, color, start, stop, self.center_x, self.center_y, rot_theta, 1, 1)

    def curve(self, t):
        t = 2 * np.pi * t
        sec = lambda x: 1. / np.cos(x)

        # That stupid square problem
        dist = sec(t - (np.pi / 2 * np.floor(2. / np.pi * (t + (np.pi / 4)))))
        return np.array([self.length*np.cos(t)*dist/2, self.height*np.sin(t)*dist/2])


class Cardoid(ParamShape):
    def __init__(self, color, center_x, center_y, scale=1, start=0, stop=1, rot_theta=0):
        super().__init__(self.curve, color, start, stop, center_x, center_y, rot_theta, scale, scale)

    def curve(self, t):
        t = 2*np.pi*t
        return np.array([40*np.cos(t)*(1 - 2*np.cos(t)), 40*np.sin(t)*(1 - 2*np.cos(t))])


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

# Helper Routines

# From https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter
def __from_rgb__(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb

# Ease Functions

def sin_squared(t):
    return np.power(np.sin(t)*np.pi/2, 2)

def ease2(t):
    return np.power(t, 2) / (np.power(t, 2) + np.power(1 - t, 2))

# Create shapes
apothem = 50
rx = -400
rect = Rectangle(-2*apothem + rx, apothem, apothem + rx, -apothem, apple_colors['lightred'], start=0, stop=0)
rect2 = Rectangle(-apothem, 3*apothem, apothem, -3*apothem, apple_colors['lightindigo'], stop=0)
scale = 2
cardoid = Cardoid(apple_colors['lightteal'], 0, 0, scale, rot_theta=np.pi/2, stop=0)

circle = Circle(0, 0, 60, apple_colors['lightorange'])

objects = [rect, cardoid, rect2, circle]

# Design animation tree
animator = Animator()
empty_anim = animator.get_root()

m0 = animator.add_animation(rect.draw_step, [ease2], duration=40, parent_animation=empty_anim, delay=1)
m01=animator.add_animation(rect2.draw_step, [ease2], duration=40, parent_animation=empty_anim, delay=15)
m02=animator.add_animation(cardoid.draw_step, [ease2], duration=40, parent_animation=empty_anim, delay=25)

m1 = animator.add_animation(cardoid.translate_step, [-500, 500, ease2], duration=30,
                            parent_animation=m02, delay=0)

m2 = animator.add_animation(rect.translate_step, [40, 40, ease2], duration=30,
                            parent_animation=m02, delay=0)

m3 = animator.add_animation(cardoid.translate_step, [300, -300, ease2], duration=60,
                            parent_animation=m1, delay=10)

m4 = animator.add_animation(rect2.translate_step, [100, -200, ease2], duration=100,
                            parent_animation=m2, delay=0)

m5 = animator.add_animation(rect2.rotate_step, [np.pi/2, ease2], duration=30, parent_animation=m2, delay=1)

m6 = animator.add_animation(cardoid.scale_step, [-1, -1, ease2], duration=50, parent_animation=m5, delay=1)

m7 = animator.add_animation(rect.scale_step, [3, -2, ease2], duration=20, parent_animation=m5, delay=5)

morph = animator.add_animation(circle.morph_step, [cardoid, ease2], duration=40, parent_animation=m7, delay=5)

morph2 = animator.add_animation(rect2.morph_step, [circle, ease2], duration=40, parent_animation=morph, delay=5)
scale = animator.add_animation(rect2.scale_step, [-1, -1, ease2], duration=40, parent_animation=morph, delay=10)

m11 = animator.add_animation(rect2.fade_color_step, [apple_colors['lightpink'], ease2], duration=30,
                             parent_animation=morph, delay=0)


first_runs = [True for i in range(len(objects))]
def run():
    global first_run, dt, objects, animator
    w.configure(background='black')

    animator.update_step()

    for i, obj in enumerate(objects):
        points = obj.get_drawing_points(0.01)
        if first_runs[i]:
            if len(points) >= 4:
                first_runs[i] = False
                w.create_line(*A_many(points), fill=__from_rgb__(obj.color), smooth=0, width=2, tag='obj'+str(i))
        else:
            if len(points) >= 4:
                line = w.find_withtag('obj'+str(i))
                w.itemconfig(line, fill=__from_rgb__(obj.color))
                w.coords(line, *A_many(points))

    w.update()
    time.sleep(dt)


# Main function
if __name__ == '__main__':
    while True:
        run()

# Necessary line for Tkinter
mainloop()
