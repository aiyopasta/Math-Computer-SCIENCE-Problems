import numpy as np
import copy

n = 10
window_w = int(2.1**n)
window_h = int(2**n)

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
                 anchor_point_x=0, anchor_point_y=0, offset_sx=1, offset_sy=1):
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

        self.n_subobjects = 1  # We call the non-group ParamShape a subobject of itself.

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
            self.target_fill_p = self.p

        t = ease(t)
        self.bounds[1] = ((1-t)*self.prior_ubound) + (t*target_ubound)
        self.p = t*self.target_fill_p

    def undraw(self, target_lbound, ease, t):
        if t == 0:
            self.prior_lbound = self.bounds[0]

        t = ease(t)
        self.bounds[0] = ((1-t)*self.prior_lbound) + (t*target_lbound)

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
    def __init__(self, center_x, center_y, a, b, color, fill_p=-1, start=0, stop=1, rot_theta=0, anchor_x=0, anchor_y=0):
        # a, b are parameters of the curve (major / minor axes)
        self.a = a
        self.b = b
        super().__init__(self.curve, color, fill_p, start, stop, center_x, center_y, rot_theta, anchor_x, anchor_y, 1, 1)

    def curve(self, t):
        t = 2*np.pi*t
        return np.array([self.a*np.cos(t), self.b*np.sin(t)])


class Circle(Ellipse):
    def __init__(self, center_x, center_y, radius, color, fill_p=-1, start=0, stop=1, rot_theta=0, anchor_x=0, anchor_y=0):
        super().__init__(center_x, center_y, radius, radius, color, fill_p, start, stop, rot_theta, anchor_x, anchor_y)


class Rectangle(ParamShape):
    def __init__(self, topleft_x, topleft_y, botright_x, botright_y, color, fill_p=-1, start=0, stop=1, rot_theta=0, anchor_x=0, anchor_y=0):

        self.center_x = (topleft_x + botright_x) / 2.
        self.center_y = (topleft_y + botright_y) / 2.
        self.length = abs(topleft_x - botright_x)
        self.height = abs(topleft_y - botright_y)
        super().__init__(self.curve, color, fill_p, start, stop, self.center_x, self.center_y, rot_theta, anchor_x, anchor_y, 1, 1)

    def curve(self, t):
        t = 2 * np.pi * t
        sec = lambda x: 1. / np.cos(x)

        # That stupid square problem
        dist = sec(t - (np.pi / 2 * np.floor(2. / np.pi * (t + (np.pi / 4)))))
        return np.array([self.length*np.cos(t)*dist/2, self.height*np.sin(t)*dist/2])


class Cardoid(ParamShape):
    def __init__(self, color, center_x, center_y, fill_p=-1, scale=1, start=0, stop=1, rot_theta=0, anchor_x=0, anchor_y=0):
        super().__init__(self.curve, color, fill_p, start, stop, center_x, center_y, rot_theta, anchor_x, anchor_y, scale, scale)

    def curve(self, t):
        t = 2*np.pi*t
        return np.array([40*np.cos(t)*(1 - 2*np.cos(t)), 40*np.sin(t)*(1 - 2*np.cos(t))])


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

        # Note that the `global color' for the entire group is None, as each of the shapes have their own color.
        # TODO: CHANGE THE HARDCODED LIGHTINDIGO TO NONE
        super().__init__(curve=self.curve, color=apple_colors['lightindigo'], start=start, stop=stop, offset_x=global_x, offset_y=global_y,
                         offset_theta=global_theta, anchor_point_x=global_anchor_x, anchor_point_y=global_anchor_y,
                         offset_sx=global_sx, offset_sy=global_sy)

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

    # Generating sample points for draw / interpolation
    def get_drawing_points(self, dt):
        '''
            dt: points are sampled from t∈self.bounds at every multiple of this parameter. e.g. dt=0.001
        '''

        point_set = []
        prev_shape_num = 0
        for t in np.arange(self.bounds[0], self.bounds[1]+dt, dt):
            t_prime, curr_shape_num = self.__remapped_t__(min(t, 1))

            # Apply local shape-wise transformations
            v = self.shapes[curr_shape_num].__get_drawing_point__(min(t_prime, 1))

            # Apply global transformations
            v = np.matmul(self.rot_matrix, np.matmul(self.scale_matrix, v) - self.anchor_point) + self.anchor_point + self.offsets_xy

            if prev_shape_num != curr_shape_num or t >= 1:
                point_set.extend(point_set[0:2])
                yield prev_shape_num, point_set
                point_set = [*v]
                prev_shape_num = curr_shape_num
            else:
                point_set.extend(v)

        yield prev_shape_num, point_set


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

    # Sh*tty Priority Queue Implementation class
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
        self.objects = objects
        self.first_runs = [[True for _ in range(obj.n_subobjects)] for obj in objects]

    def paint_step(self):
        w = self.w

        for i, obj in enumerate(self.objects):
            for j, point_set in enumerate(obj.get_drawing_points(0.01)):
                shape = obj
                points = point_set
                if isinstance(obj, ParamShapeGroup):  # reason we don't use n_subobjects here is bc if there's just
                    shape = obj.shapes[point_set[0]]  # 1 object in the group, it won't get colored then as we'll think
                    points = point_set[1]  # its just a single ParamShape and not a ParamShapeGroup.

                if len(points) >= 4:
                    if self.first_runs[i][j]:
                        self.first_runs[i][j] = False

                        if shape.p >= 0:
                            fill_color = change_color_intensity(shape.color, shape.p)
                            w.create_polygon(*A_many(points), fill=__from_rgb__(fill_color), smooth=0, width=2,
                                             tag='shape' + str(i) + str(j))
                        w.create_line(*A_many(points), fill=__from_rgb__(shape.color), smooth=0, width=2,
                                      tag='boundary' + str(i) + str(j))
                    else:
                        if shape.p >= 0:
                            fill_color = change_color_intensity(shape.color, shape.p)
                            poly = w.find_withtag('shape' + str(i) + str(j))
                            w.itemconfig(poly, fill=__from_rgb__(fill_color))
                            w.coords(poly, *A_many(points))

                        line = w.find_withtag('boundary' + str(i) + str(j))
                        w.itemconfig(line, fill=__from_rgb__(shape.color))
                        w.coords(line, *A_many(points))

        w.update()
