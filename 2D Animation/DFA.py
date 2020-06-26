from ghetto_manim import *
import time
import random

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Special Shape Classes
class Adjacent1sDFA(ParamShapeGroup):
    class Vertex:
        def __init__(self, center_x, center_y, radius, color, fill_p=0, stop=1, recognizing=False):
            self.recognizing = recognizing
            self.circle = Circle(center_x, center_y, radius, color, fill_p=fill_p, stop=stop)

            if recognizing:
                k = 1.2
                l = 1.3
                self.circle2 = Circle(center_x, center_y, k*radius, color, fill_p=fill_p, stop=stop)
                self.trace_circle = Circle(center_x, center_y, l*radius, apple_colors['lightyellow'], fill_p=-1, stop=0)

            # Categorize children by edge label needed to reach there.
            self.children = {'a': [], 'b': []}

            # Tag
            self.tag = 'c' + str(random.randint(10000, 90000))

            # Base color
            self.base_color = color

        def add_child(self, vertex, key):
            self.children[key].append(vertex)

    def __init__(self, read_string):
        r = 100
        sep_dist = 2*r
        self.edge_map = {}

        self.read_string = read_string
        self.current_char_index = 0

        start_vertex = self.Vertex((-2*r) - sep_dist, 0, r, apple_colors['lightblue'])
        middle_vertex = self.Vertex(0, 0, r, apple_colors['lightpurple'])
        recog_vertex = self.Vertex((2*r) + sep_dist, 0, r, apple_colors['lightindigo'], recognizing=True)

        self.original_curr_vert = copy.copy(start_vertex)

        start_vertex.add_child(middle_vertex, 'a')
        start_vertex.add_child(start_vertex, 'b')
        middle_vertex.add_child(start_vertex, 'b')
        middle_vertex.add_child(recog_vertex, 'a')
        recog_vertex.add_child(recog_vertex, 'a')
        recog_vertex.add_child(recog_vertex, 'b')

        self.edge_arrows = self.__create_edges__([start_vertex, middle_vertex, recog_vertex], '00,01,10,12,22',
                                                [(np.pi/3, np.pi/3), (np.pi/4, np.pi/4),
                                                 (np.pi+(np.pi/4), np.pi+(np.pi/4)),
                                                 (np.pi/4, np.pi/4), (np.pi/3, np.pi/3)])

        self.current_vertex = start_vertex

        objects = [start_vertex.circle, middle_vertex.circle, recog_vertex.circle2, recog_vertex.circle,
                   recog_vertex.trace_circle, *self.edge_arrows]

        super().__init__(objects, start=0, stop=0)

    def __create_edges__(self, vertices, edge_encoding, edge_angles):
        '''
            vertices: list of Vertex objects
            edge_encoding: a string of the form '00, 01, 10, 12, 22', where 0,1,2 refer to indices in vertex list,
                           and ij refers to an edge from the ith vertex to the jth vertex.
            edge_angle = list of pairs (alpha_k, beta_k) where alpha_i represents the angle from the horizontal
                         measured counter-clockwise that the kth edge arrow curve leaves a vertex, and beta_i
                         represents the angle from the horizontal measured CLOCKWISE that the edge arrow curve
                         enters a vertex.
        '''

        edge_comps = []
        for i, edge in enumerate(edge_encoding.split(',')):
            c1, c2 = vertices[int(edge[0])].circle, vertices[int(edge[1])].circle

            curvature = 250
            if edge[0] == edge[1]:
                curvature *= -1

            alpha_i, beta_i = edge_angles[i][0], edge_angles[i][1]
            curve = CurvedLine(c1.center_x + c1.radius*np.cos(alpha_i), c1.center_y + c1.radius*np.sin(alpha_i),
                               c2.center_x - c2.radius*np.cos(beta_i), c2.center_y + c2.radius*np.sin(beta_i),
                               (255, 255, 255), fill_p=-1, curve_place=0.5, curve_amount=curvature, stop=1)

            triangle = Triangle(*curve.p2, (255, 255, 255), fill_p=0.2, stop=1,
                                rot_theta=curve.curve_angle(), scale_x=10, scale_y=10)

            edge_tup = [curve, triangle]

            self.__set_edge_tup__(vertices[int(edge[0])], vertices[int(edge[1])], edge_tup)
            edge_comps.extend(edge_tup)

        return edge_comps

    def __set_edge_tup__(self, v1, v2, edge_tup):
        '''
            c1, c2: vertices that edge is connecting
            edge_tup: (CurvedLine object, Triangle object)
        '''
        tag = v1.tag + v2.tag
        self.edge_map[tag] = edge_tup

    def __get_edge_tup__(self, v1, v2):
        tag = v1.tag + v2.tag
        return self.edge_map[tag]

    def __get_edge_tups_tofrom__(self, v):
        # Lame idea but won't slow animations as this is done beforehand.
        edge_tups_in = []
        edge_tups_out = []
        for key in self.edge_map.keys():
            if key[:6] == v.tag:
                edge_tups_out.append(self.edge_map[key])

            if key[6:] == v.tag:
                edge_tups_in.append(self.edge_map[key])

        return edge_tups_in, edge_tups_out

    def expand_current_vertex_step(self, ease, t):
        if t == 0:
            self.prior_edge_tups_in_xy = []
            self.prior_edge_tups_out_xy = []

        v = self.current_vertex
        c = v.circle

        c.scale_step(1.2, 1.2, ease, t)
        if v.recognizing:
            v.circle2.scale_step(1.1, 1.1, ease, t)
            v.trace_circle.scale_step(1.1, 1.1, ease, t)

        edge_tups_in, edge_tups_out = self.__get_edge_tups_tofrom__(v)

        for i, edge_tup in enumerate(edge_tups_in):
            curve, triangle = edge_tup[0], edge_tup[1]

            center = np.array([c.center_x, c.center_y])
            if t == 0:
                self.prior_edge_tups_in_xy.append(((curve.p2 - center) * 1.2) + center)

            curve.translate_tip(self.prior_edge_tups_in_xy[i], ease, t)
            triangle.offset_theta = curve.curve_angle
            triangle.offsets_xy = curve.p2

        for i, edge_tup in enumerate(edge_tups_out):
            curve, triangle = edge_tup[0], edge_tup[1]

            center = np.array([c.center_x, c.center_y])
            if t == 0:
                self.prior_edge_tups_out_xy.append(((curve.p1 - center) * 1.2) + center)

            curve.translate_tail(self.prior_edge_tups_out_xy[i], ease, t)

    def contract_current_vertex_step(self, ease, t):
        if t == 0:
            self.prior_edge_tups_in_xy = []
            self.prior_edge_tups_out_xy = []

        v = self.current_vertex
        c = v.circle

        c.scale_step(1, 1, ease, t)
        if v.recognizing:
            v.circle2.scale_step(1, 1, ease, t)
            v.trace_circle.scale_step(1, 1, ease, t)

        edge_tups_in, edge_tups_out = self.__get_edge_tups_tofrom__(v)

        for i, edge_tup in enumerate(edge_tups_in):
            curve, triangle = edge_tup[0], edge_tup[1]

            center = np.array([c.center_x, c.center_y])
            if t == 0:
                self.prior_edge_tups_in_xy.append(((curve.p2 - center) / 1.2) + center)

            curve.translate_tip(self.prior_edge_tups_in_xy[i], ease, t)
            triangle.offset_theta = curve.curve_angle
            triangle.offsets_xy = curve.p2

        for i, edge_tup in enumerate(edge_tups_out):
            curve, triangle = edge_tup[0], edge_tup[1]

            center = np.array([c.center_x, c.center_y])
            if t == 0:
                self.prior_edge_tups_out_xy.append(((curve.p1 - center) / 1.2) + center)

            curve.translate_tail(self.prior_edge_tups_out_xy[i], ease, t)

        if t >= 1:
            char = self.read_string[self.current_char_index]
            self.current_vertex = self.current_vertex.children[char][0]
            self.current_char_index += 1

    def trace_arrow_step(self, trace_edge, trace_triangle, ease, t):
        # if t == 0:
        # char = self.read_string[self.current_char_index]
        # self.edge, self.triangle = self.edge_map[self.current_vertex.tag + self.current_vertex.children[char][0].tag]
        #
        # trace_edge.p1, trace_edge.p2 = self.edge.p1, self.edge.p2
        # trace_edge.t_k = self.edge.t_k
        # trace_edge.curve_amount = self.edge.curve_amount
        # trace_edge.bounds = [0, 0]
        # trace_edge.__calculate_p_mid__()
        #
        # trace_triangle.offsets_xy = self.triangle.offsets_xy
        # trace_triangle.offset_theta = self.triangle.offset_theta
        # trace_triangle.scale_x = self.triangle.scale_x
        # trace_triangle.scale_y = self.triangle.scale_y
        #
        # t = ease(t)
        #
        # trace_edge.draw_step(1, ease, t)
        # trace_triangle.draw_step(1, ease, t)
        # self.edge.undraw(1, ease, t)
        pass

    def untrace_arrow_step(self, trace_edge, trace_triangle, ease, t):
        # char = self.read_string[self.current_char_index]
        #
        # if t == 0:
        #     self.edge, self.triangle = self.edge_map[self.current_vertex.tag + self.current_vertex.children[char][0].tag]
        #     self.edge.bounds = [0, 0]
        #
        # trace_edge.p1, trace_edge.p2 = self.edge.p1, self.edge.p2
        # trace_edge.t_k = self.edge.t_k
        # trace_edge.curve_amount = self.edge.curve_amount
        # trace_edge.bounds = [0, 1]
        # trace_edge.__calculate_p_mid__()
        #
        # trace_triangle.offsets_xy = self.triangle.offsets_xy
        # trace_triangle.offset_theta = self.triangle.offset_theta
        # trace_triangle.scale_x = self.triangle.scale_x
        # trace_triangle.scale_y = self.triangle.scale_y
        #
        # t = ease(t)
        #
        # trace_edge.undraw(1, ease, t)
        # trace_triangle.undraw(1, ease, t)
        # self.edge.draw_step(1, ease, t)
        pass


    def trace_curr_vertex_step(self, color, ease, t):
        v = self.current_vertex
        v.trace_circle.color = color
        v.trace_circle.draw_step(1, ease, t)

    def untrace_curr_vertex_step(self, ease, t):
        v = self.current_vertex
        assert v.trace_circle.bounds[1] > 0

        v.trace_circle.undraw(1, ease, t)

        if t >= 1:
            v.trace_circle.bounds[0] = 0
            v.trace_circle.bounds[1] = 0

    def fill_curr_vertex_step(self, ease, t):
        v = self.current_vertex
        v.circle.fade_fill_step(0.15, ease, t)

        if v.recognizing:
            v.circle2.fade_fill_step(0.05, ease, t)

    def unfill_curr_vertex_step(self, ease, t):
        v = self.current_vertex
        v.circle.fade_fill_step(0, ease, t)

        if v.recognizing:
            v.circle2.fade_fill_step(0, ease, t)

    def change_curr_vertex_color_step(self, color, ease, t):
        v = self.current_vertex
        v.circle.fade_color_step(color, ease, t)

        if v.recognizing:
            v.circle2.fade_color_step(color, ease, t)


# Special animation procedures
def make_curr_vertex_active(animator, dfa, parent_anim, delay):
    final_anim = None

    if dfa.current_vertex.recognizing:
        animator.add_animation(dfa.trace_curr_vertex_step, [apple_colors['lightyellow'], smooth],
                               duration=30, parent_animation=parent_anim, delay=delay)
        animator.add_animation(dfa.untrace_curr_vertex_step, [smooth], duration=20,
                               parent_animation=parent_anim, delay=delay+15)

        animator.add_animation(dfa.expand_current_vertex_step, [smooth], duration=30, parent_animation=parent_anim,
                               delay=delay+5)

        animator.add_animation(dfa.fill_curr_vertex_step, [smooth], duration=30, parent_animation=parent_anim,
                               delay=delay+7)

        final_anim = animator.add_animation(dfa.change_curr_vertex_color_step, [apple_colors['lightorange'], smooth], duration=30,
                               parent_animation=parent_anim, delay=delay+7)

    else:
        final_anim = animator.add_animation(dfa.expand_current_vertex_step, [smooth], duration=20, parent_animation=parent_anim,
                               delay=delay)

    return final_anim

def make_curr_vertex_inactive(animator, dfa, parent_anim, delay):
    final_anim = None

    if dfa.current_vertex.recognizing:
        animator.add_animation(dfa.contract_current_vertex_step, [smooth], duration=30, parent_animation=parent_anim,
                               delay=delay+5)

        animator.add_animation(dfa.unfill_curr_vertex_step, [smooth], duration=30, parent_animation=parent_anim,
                               delay=delay+7)

        final_anim = animator.add_animation(dfa.change_curr_vertex_color_step, [dfa.current_vertex.base_color, smooth], duration=30,
                               parent_animation=parent_anim, delay=delay+7)

    else:
        final_anim = animator.add_animation(dfa.contract_current_vertex_step, [smooth], duration=20,
                                            parent_animation=parent_anim, delay=delay)

    return final_anim

# Time Step
dt = 0.001


def scene():
    global dt
    # Object Construction
    stringy = 'ababbbaabb'
    dfa = Adjacent1sDFA(read_string = stringy)
    trace_edge = CurvedLine(0, 0, 0, 0, apple_colors['lightyellow'], -1, stop=0)
    trace_triangle = Triangle(0, 0, apple_colors['lightyellow'], -1, stop=0)

    box = Rectangle(-92, -260, -46, -340, color=apple_colors['lightgreen'], stop=0)

    objects = [dfa, box]

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()

    draw = animator.add_animation(dfa.draw_step, [1, smooth], duration=100, parent_animation=empty_anim, delay=5)
    latest_anims = [make_curr_vertex_active(animator, dfa, draw, delay=10)]

    animator.add_animation(box.draw_step, [1, smooth], duration=30, parent_animation=empty_anim, delay=10)

    for i, char in enumerate(stringy):
        latest_anims.append(make_curr_vertex_inactive(animator, dfa, latest_anims[-1], delay=10))
        dfa.current_vertex = dfa.current_vertex.children[char][0]
        animator.add_animation(box.translate_step, [-69 + ((i + 1) * 47), -300, smooth], duration=20,
                               parent_animation=latest_anims[-1], delay=0)
        latest_anims.append(make_curr_vertex_active(animator, dfa, latest_anims[-1], delay=10))

    dfa.current_vertex = dfa.original_curr_vert


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