from ghetto_manim import *
import time
from PIL import Image
import os
from svgpathtools import svg2paths
import random

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Special Shape Classes
class DVDLogoObject(ParamShape):
    def __init__(self, pseudo_size_d, init_x, init_y, init_vel_x, init_vel_y, color, fill_p=-1.,
                 drawing_point_delta=0.01, start=0, stop=1):

        self.paths, _ = svg2paths('/Users/adityaabhyankar/Desktop/dvd.svg')
        super().__init__(self.curve, color, fill_p=fill_p, start=start, stop=stop, offset_x=init_x, offset_y=init_y,
                         offset_theta=0, anchor_point_x=0, anchor_point_y=0, offset_sx=1, offset_sy=-1,
                         drawing_point_delta=drawing_point_delta)

        self.vx = init_vel_x
        self.vy = init_vel_y
        self.d = pseudo_size_d

        self.current_part = 0

        self.__calculate_extra_stuff__()

        # Method specific vars
        self.morphed = False
        self.raw_color = None

    def __calculate_extra_stuff__(self):
        p1, p2 = self.paths[0], self.paths[1]
        _, xmax, ymin, _ = p1.bbox()
        xmin, _, _, ymax = p2.bbox()

        self.extra_offset_x = (xmax + xmin) / 2
        self.extra_offset_y = (ymax + ymin) / 2

        self.xmax, self.xmin = (xmax - self.extra_offset_x) / self.d, (xmin - self.extra_offset_x) / self.d
        self.ymax, self.ymin = (ymax - self.extra_offset_y) / self.d, (ymin - self.extra_offset_y) / self.d

        self.length, self.height = self.xmax - self.xmin, self.ymax - self.ymin

    def curve(self, t):
        m1, m2 = 0.4, 0.8  # m1 is t at which first path will stop drawing,
                           # m2 is t at which subpath of 2nd path will start drawing
        p = 0
        if 0 <= t < m1:
            t = self.__remapped_t__(t, 0, m1)
            self.current_part = 0
            p = self.paths[0].point(t)
        elif m1 <= t < m2:
            t = self.__remapped_t__(t, m1, m2)
            self.current_part = 1
            p = self.paths[1].continuous_subpaths()[0].point(t)
        else:
            t = self.__remapped_t__(t, m2, 1)
            self.current_part = 2
            p = self.paths[1].continuous_subpaths()[1].point(t)

        return np.array([(p.real - self.extra_offset_x) / self.d, (p.imag - self.extra_offset_y) / self.d])

    @staticmethod
    def __remapped_t__(t, a, b):
        slope = (1 - 0) / (b - a)
        yint = -a * slope

        return (slope*t) + yint

    def get_drawing_points(self):
        dt = self.drawing_point_delta
        point_set = []

        prev_part = 0
        for t in np.arange(self.bounds[0], self.bounds[1]+dt, dt):
            v = self.__get_drawing_point__(min(t, self.bounds[1]))

            if prev_part != self.current_part or t >= 1:
                point_set.extend(point_set[0:2])
                yield point_set
                point_set = [*v]
                prev_part = self.current_part
            else:
                point_set.extend(v)

        yield point_set

    def bounce_around(self, rectangle, t):
        '''rectangle contrains logo inside it'''
        r_xmin, r_xmax, r_ymin, r_ymax = rectangle.get_bbox()

        if self.xmin + self.vx <= r_xmin:
            a = r_xmin - self.xmin
            self.offsets_xy[0] += a
            self.xmax += a
            self.xmin += a
            self.vx *= -1
            self.randomly_change_color()
        elif r_xmax <= self.xmax + self.vx:
            a = r_xmax - self.xmax
            self.offsets_xy[0] += a
            self.xmin += a
            self.xmax += a
            self.vx *= -1
            self.randomly_change_color()
        else:
            self.offsets_xy[0] += self.vx
            self.xmin += self.vx
            self.xmax += self.vx

        if self.ymin + 2*self.vy <= r_ymin:
            a = r_ymin - self.ymin
            self.offsets_xy[1] += a
            self.ymin += a
            self.ymax += a
            self.vy *= -1
            self.randomly_change_color()
        elif r_ymax <= self.ymax + self.vy:
            a = r_ymax - self.ymax
            self.offsets_xy[1] += a
            self.ymin += a
            self.ymax += a
            self.vy *= -1
            self.randomly_change_color()
        else:
            self.offsets_xy[1] += self.vy
            self.ymin += self.vy
            self.ymax += self.vy

    def randomly_change_color(self):
        colors = list(apple_colors.values())

        if self.raw_color is not None:
            self.raw_color = colors[random.randint(0, len(colors)-1)]
        else:
            self.color = colors[random.randint(0, len(colors)-1)]

    def morph_step(self, target_shape, ease, t):
        if t > 0.7:
            self.morphed = True

        super().morph_step(target_shape, ease, t)

    def fade_fill_step(self, target_p, ease, t):
        if t == 0:
            self.raw_color = self.color

        target_color = change_color_intensity(self.raw_color, target_p)
        super().fade_color_step(target_color, ease, t)


class DVDLogoMidpoint(Rectangle):
    def __init__(self, dvd_obj, side_len, color, fill_p=-1, start=0, stop=1, rot_theta=0, anchor_x=0, anchor_y=0,
                 drawing_point_delta=0.01):

        self.dvd = dvd_obj

        topleft_x = dvd_obj.offsets_xy[0] - (side_len/2)
        topleft_y = dvd_obj.offsets_xy[1] + (side_len/2)
        botright_x = dvd_obj.offsets_xy[0] + (side_len/2)
        botright_y = dvd_obj.offsets_xy[1] - (side_len/2)
        super().__init__(topleft_x, topleft_y, botright_x, botright_y, color, fill_p, start, stop, rot_theta,
                         anchor_x, anchor_y, drawing_point_delta)

    def follow_logo(self, t):
        self.offsets_xy[0] = self.dvd.offsets_xy[0]
        self.offsets_xy[1] = self.dvd.offsets_xy[1]

class DVDLogoBoundingBox(Rectangle):
    def __init__(self, dvd_obj, color, fill_p=-1, start=0, stop=1, rot_theta=0, anchor_x=0, anchor_y=0,
                 drawing_point_delta=0.01):

        self.dvd = dvd_obj
        length = dvd_obj.xmax - dvd_obj.xmin
        height = dvd_obj.ymax - dvd_obj.ymin

        topleft_x = dvd_obj.offsets_xy[0] - (length/2)
        topleft_y = dvd_obj.offsets_xy[1] + (height/2)
        botright_x = dvd_obj.offsets_xy[0] + (length/2)
        botright_y = dvd_obj.offsets_xy[1] - (height/2)

        super().__init__(topleft_x, topleft_y, botright_x, botright_y, color, fill_p, start, stop, rot_theta,
                         anchor_x, anchor_y, drawing_point_delta)

    def follow_logo(self, t):
        self.offsets_xy[0] = self.dvd.offsets_xy[0]
        self.offsets_xy[1] = self.dvd.offsets_xy[1]


class Arrow(ParamShapeGroup):
    def __init__(self, dvd_obj, x0, y0, x1, y1, color, fill_p=0., curve_place=0.5, curve_amount=0, start=0, stop=1):
        curve = CurvedLine(x0, y0, x1, y1, color, -1, curve_place, curve_amount, stop=1)
        triangle = Triangle(x1, y1, color, fill_p=fill_p, stop=1, rot_theta=curve.curve_angle(), scale_x=10, scale_y=10)

        self.dvd = dvd_obj

        super().__init__([curve, triangle], start=start, stop=stop)

    def follow_logo(self, t):
        self.offsets_xy[0] = self.dvd.offsets_xy[0]
        self.offsets_xy[1] = self.dvd.offsets_xy[1]

        if self.offset_sxy[0] * self.dvd.vx < 0:
            self.offset_sxy[0] *= -1
            self.scale_matrix = self.__scale_matrix__(*self.offset_sxy)

        if self.offset_sxy[1] * self.dvd.vy < 0:
            self.offset_sxy[1] *= -1
            self.scale_matrix = self.__scale_matrix__(*self.offset_sxy)

    def stretch_shrink(self, target_factor_x, target_factor_y, ease, t):
        assert target_factor_x > 0 and target_factor_y > 0

        if t == 0:
            self.prev_factor_x = abs(self.offset_sxy[0])
            self.prev_factor_y = abs(self.offset_sxy[1])

        t = ease(t)

        new_factors = np.array([((1-t)*self.prev_factor_x) + (t*target_factor_x),
                                ((1-t)*self.prev_factor_y) + (t*target_factor_y)])

        self.offset_sxy = np.sign(self.offset_sxy) * new_factors


class SmallBouncer(Rectangle):
    def __init__(self, x, y, r, vx, vy, color, fill_p=-1, start=0, stop=1, rot_theta=0, anchor_x=0, anchor_y=0,
                 drawing_point_delta=0.01):
        '''
            x, y are initial positions of the center of the square
            r is apothem of square,
            v_i is the speed of each component
        '''

        topleft_x, topleft_y = x - r, y + r
        botright_x, botright_y = x + r, y - r

        self.vx, self.vy = vx, vy
        self.hit = False
        super().__init__(topleft_x, topleft_y, botright_x, botright_y, color, fill_p, start, stop, rot_theta,
                         anchor_x, anchor_y, drawing_point_delta)


    def bounce_around(self, screen, t):
        r_xmin, r_xmax, r_ymin, r_ymax = screen.get_bbox()

        if self.offsets_xy[0] + self.vx <= r_xmin:
            a = r_xmin - self.offsets_xy[0]
            self.offsets_xy[0] += a
            self.vx *= -1
            self.hit = True
            # self.randomly_change_color()
        elif r_xmax <= self.offsets_xy[0] + self.vx:
            a = r_xmax - self.offsets_xy[0]
            self.offsets_xy[0] += a
            self.vx *= -1
            self.hit = True
            # self.randomly_change_color()
        else:
            self.hit = False
            self.offsets_xy[0] += self.vx

        if self.offsets_xy[1] + 2*self.vy <= r_ymin:
            a = r_ymin - self.offsets_xy[1]
            self.offsets_xy[1] += a
            self.vy *= -1
            self.hit = True
            # self.randomly_change_color()
        elif r_ymax <= self.offsets_xy[1] + self.vy:
            a = r_ymax - self.offsets_xy[1]
            self.offsets_xy[1] += a
            self.vy *= -1
            self.hit = True
            # self.randomly_change_color()
        else:
            if not self.hit:
                self.hit = False

            self.offsets_xy[1] += self.vy


class BounceTrajectory(ParamShape):
    def __init__(self, bouncer, screen, color, fill_p=-1, start=0, stop=1, drawing_point_delta=0.01):
        # Note: Set stop=1 if you want the curve to draw.

        self.bouncer = bouncer
        self.screen = screen

        self.state = 0  # 0 - seeking incoming bounce, 1 - drawing trajectory, 2 - trajectory drawn, not seeking

        self.p1 = None
        self.p_mid = None  # remember to reset this when you undraw the trajectory
        self.p2 = None

        super().__init__(self.curve, color, fill_p=-1, start=0, stop=1, offset_x=0, offset_y=0, offset_theta=0,
                 anchor_point_x=0, anchor_point_y=0, offset_sx=1, offset_sy=1, drawing_point_delta=drawing_point_delta)

    def curve(self, t):
        if self.state > 0:
            if self.bouncer.hit and self.state == 1:
                self.p_mid = copy.copy(self.bouncer.offsets_xy)

            if self.p_mid is not None:
                t_prime = self.__remapped_t__(t)
                if 0 <= t < 0.5:
                    return ((1 - t_prime) * self.p1) + (t_prime * self.p_mid)
                else:
                    if self.state != 2 and np.linalg.norm(self.bouncer.offsets_xy - self.p_mid) <= np.linalg.norm(self.p_mid - self.p1):
                        self.p2 = copy.copy(self.bouncer.offsets_xy)
                    else:
                        self.state = 2

                    return ((1 - t_prime) * self.p_mid) + (t_prime * self.p2)

            else:
                pk = self.bouncer.offsets_xy
                return ((1-t) * self.p1) + (t * pk)

        else:
            mult = 13
            r_xmin, r_xmax, r_ymin, r_ymax = self.screen.get_bbox()

            incoming_bounce = False
            if self.bouncer.offsets_xy[0] + (mult * abs(self.bouncer.vx)) >= r_xmax:
                incoming_bounce = True
            if self.bouncer.offsets_xy[0] - (mult * abs(self.bouncer.vx)) <= r_xmin:
                incoming_bounce= True
            if self.bouncer.offsets_xy[1] + (mult * abs(self.bouncer.vy)) >= r_ymax:
                incoming_bounce = True
            if self.bouncer.offsets_xy[1] - (mult * abs(self.bouncer.vy)) <= r_ymin:
                incoming_bounce = True

            if incoming_bounce:
                self.state = 1
                self.p1 = copy.copy(self.bouncer.offsets_xy)

                return self.p1

            return np.array([0, 0])

    def __remapped_t__(self, t):
        if 0 <= t < 0.5:
            return (1 / 0.5) * t

        else:
            m = 1 / (1 - 0.5)
            b = -0.5 * m
            return (m * t) + b

    def undraw(self, target_lbound, ease, t):
        super().undraw(target_lbound, ease, t)

        if t == 1:
            self.state = 0
            self.p1 = None
            self.p_mid = None
            self.p2 = None

            self.bounds[0] = 0
            self.bounds[1] = 1


class BounceAngleArc(Circle):
    def __init__(self, radius, color, fill_p=-1, start=0, stop=1, drawing_point_delta=0.01):
        super().__init__(0, 0, radius, color, fill_p=fill_p, start=start, stop=stop, rot_theta=0,
                 anchor_x=0, anchor_y=0, drawing_point_delta=drawing_point_delta)

    def draw_angle_arc(self, trajectory, half, ease, t):
        '''
            half decides on which side to draw arc. 0 - left half, 1 - right half
        '''
        assert isinstance(trajectory, BounceTrajectory) and trajectory.state == 2

        # We draw the arc FROM the screen TO the trajectory.
        if half == 0:
            self.offset_sxy[1] = -1
            self.scale_matrix = self.__scale_matrix__(*self.offset_sxy)

        self.offsets_xy = copy.copy(trajectory.p_mid)

        if trajectory.p_mid[0] > trajectory.p1[0] and trajectory.p_mid[1] > trajectory.p1[1]:
            self.offset_theta = -np.pi/2
            self.rot_matrix = self.__rotation_matrix__(self.offset_theta)

        super().draw_step(1./8., ease, t)

class LogoFollowingGlyph(Glyph):
    def __init__(self, character, dvd, pseudo_font_size, color, start=0, stop=1, left_x=0, bottom_y=0,
                 angle=0, anchor_point_x=0, anchor_point_y=0, offset_sx=1, offset_sy=1, drawing_points_delta=0.01):

        self.dvd = dvd
        super().__init__(character, pseudo_font_size, color, start, stop, left_x, bottom_y, angle, anchor_point_x,
                         anchor_point_y, offset_sx, offset_sy, drawing_points_delta=drawing_points_delta)

    def follow_logo(self, t):
        self.offsets_xy[0] = self.dvd.offsets_xy[0]
        self.offsets_xy[1] = self.dvd.offsets_xy[1]

    def morph_step(self, target_shape, ease, t):
        assert isinstance(target_shape, ParamShape)

        # Figure out a way to fix method to allow for >1 morphs in succession. Currently throws exception.
        if t == 0:
            self.prior_curve = self.curve

        t = ease(t)
        self.curve = lambda x: ((1 - t) * self.prior_curve(x)) + (t * target_shape.curve(x))

        self.rotate_step(target_shape.offset_theta, ease, t)
        self.scale_step(*target_shape.offset_sxy, ease, t)
        self.fade_color_step(target_shape.color, ease, t)

        if t > 0.45:
            self.morphed = True

# Time step
dt = 0.01

# Global vars
save_anim = False
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

def basic_logo_bounce_anim():
    global stop, main_dir, ps_files_dir, png_files_dir, img_list

    # Object Creation
    r = 40
    screen = Rectangle(-window_w / 2 + r, window_h / 2 - 150, window_w / 2 - r, -window_h / 2 + 40,
                       apple_colors['lightred'],
                       stop=0, drawing_point_delta=0.001)

    v = 10
    dvd = DVDLogoObject(0.5, 0, 0, v, v, apple_colors['lightteal'], drawing_point_delta=0.003, stop=0, fill_p=0.1)  # fill_p = 0.5
    title = Text('DVD Bouncing Animation', 10, (255, 255, 255), stop=0, left_x=-window_w/2 + 50,
                 bottom_y=200, drawing_delta_per_glyph=0.015, spacing=10)

    point = DVDLogoMidpoint(dvd, 10, (255,0,0), fill_p=1, stop=0, drawing_point_delta=0.005)
    inner_screen = Rectangle(-screen.length/2 + dvd.length/2, screen.height/2 + screen.offsets_xy[1] - dvd.height/2,
                             screen.length/2 - dvd.length/2, -screen.height/2 + screen.offsets_xy[1] + dvd.height/2,
                             apple_colors['lightblue'], stop=0, drawing_point_delta=0.001)

    objects = [dvd, title, screen, point, inner_screen]

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()
    draw = animator.add_animation(title.draw_step, [1, smooth], duration=40, parent_animation=empty_anim, delay=0)
    draw = animator.add_animation(screen.draw_step, [1, smooth], duration=40, parent_animation=empty_anim, delay=0)
    draw = animator.add_animation(dvd.draw_step, [1, smooth], duration=40, parent_animation=draw, delay=0)
    evolve = animator.add_animation(dvd.bounce_around, [screen], duration=2000, parent_animation=draw, delay=0)

    # follow = animator.add_animation(point.follow_logo, [], duration=2000, parent_animation=draw, delay=0)
    # wait = 40
    # draw_bouncer = animator.add_animation(point.draw_step, [1, smooth], duration=30, parent_animation=draw, delay=wait)
    # fade_out = animator.add_animation(dvd.fade_fill_step, [0.2, smooth], duration=30, parent_animation=draw, delay=wait)
    # draw_inner = animator.add_animation(inner_screen.draw_step, [1, smooth], duration=50, parent_animation=fade_out, delay=20)
    #
    # fadeout_logo = animator.add_animation(dvd.fade_fill_step, [0, smooth], duration=50, parent_animation=draw_inner, delay=40)
    # morph = animator.add_animation(screen.morph_step, [inner_screen, smooth], duration=30, parent_animation=fadeout_logo, delay=0)


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

def bounding_rectangle():
    global stop, main_dir, ps_files_dir, png_files_dir, img_list

    # Object Creation
    r = 40
    screen = Rectangle(-window_w / 2 + r, window_h / 2 - 150, window_w / 2 - r, -window_h / 2 + 40,
                       apple_colors['lightred'],
                       stop=1, drawing_point_delta=0.001)

    v = 10
    dvd = DVDLogoObject(0.5, 0, 0, v, v, apple_colors['lightteal'], drawing_point_delta=0.003, stop=1,
                        fill_p=0.1)  # fill_p = 0.5
    title = Text('DVD Bouncing Animation', 10, (255, 255, 255), stop=0, left_x=-window_w / 2 + 50,
                 bottom_y=200, drawing_delta_per_glyph=0.015, spacing=10)

    bounding_box = DVDLogoBoundingBox(dvd, apple_colors['lightgreen'], fill_p=-1, stop=0, drawing_point_delta=0.005)

    objects = [dvd, title, screen, bounding_box]

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()
    evolve = animator.add_animation(dvd.bounce_around, [screen], duration=2000, parent_animation=empty_anim, delay=0)
    follow = animator.add_animation(bounding_box.follow_logo, [], duration=2000, parent_animation=empty_anim, delay=0)

    wait = 80
    draw_box = animator.add_animation(bounding_box.draw_step, [1, smooth], duration=30, parent_animation=empty_anim, delay=wait)
    fade_logo = animator.add_animation(dvd.fade_fill_step, [0.22, smooth], duration=30, parent_animation=empty_anim, delay=wait+10)




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

def balanced_velocity():
    global stop, main_dir, ps_files_dir, png_files_dir, img_list

    # Object Creation
    r = 40
    screen = Rectangle(-window_w / 2 + r, window_h / 2 - 150, window_w / 2 - r, -window_h / 2 + 40,
                       apple_colors['lightred'], stop=1, drawing_point_delta=0.001)

    v = -10
    dvd = DVDLogoObject(0.5, 0, 0, v, v, apple_colors['lightteal'], drawing_point_delta=0.003, stop=1,
                        fill_p=0.1)  # fill_p = 0.5

    horiz_arrow = Arrow(dvd, dvd.offsets_xy[0], dvd.offsets_xy[1], dvd.offsets_xy[0] + 100, dvd.offsets_xy[1],
                        color=(255,255,255), stop=0, fill_p=0.4)
    horiz_avg = (horiz_arrow.shapes[0].p2 + horiz_arrow.shapes[0].p1) / 2.
    # horiz_vel_num = LogoFollowingGlyph('3', dvd, 20, apple_colors['lightblue'], stop=0,
    #                                    left_x=horiz_avg[0], bottom_y=horiz_avg[1], drawing_points_delta=0.01)

    vert_arrow = Arrow(dvd, dvd.offsets_xy[0], dvd.offsets_xy[1], dvd.offsets_xy[0], dvd.offsets_xy[1] + 100,
                        color=(255,255,255), stop=0, fill_p=0.4)
    vert_avg = (vert_arrow.shapes[0].p2 + vert_arrow.shapes[0].p1) / 2.
    # vert_vel_num = LogoFollowingGlyph('5', dvd, 20, apple_colors['lightblue'], stop=0,
    #                                    left_x=vert_avg[0], bottom_y=vert_avg[1], drawing_points_delta=0.01)

    objects = [dvd, screen, horiz_arrow, vert_arrow]

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()

    # Main logo moving
    evolve = animator.add_animation(dvd.bounce_around, [screen], duration=2000, parent_animation=empty_anim, delay=0)

    follow_logo1 = animator.add_animation(horiz_arrow.follow_logo, [], duration=2000, parent_animation=empty_anim, delay=0)
    # follow_logo1 = animator.add_animation(horiz_vel_num.follow_logo, [], duration=2000, parent_animation=empty_anim, delay=0)
    draw_arrow1 = animator.add_animation(horiz_arrow.draw_step, [1, smooth], duration=40, parent_animation=empty_anim, delay=80)
    # draw_velnum1 = animator.add_animation(horiz_vel_num.draw_step, [1, smooth], duration=40, parent_animation=empty_anim, delay=80)

    follow_logo2 = animator.add_animation(vert_arrow.follow_logo, [], duration=2000, parent_animation=empty_anim, delay=0)
    # follow_logo2 = animator.add_animation(vert_vel_num.follow_logo, [], duration=2000, parent_animation=empty_anim, delay=0)
    draw_arrow2 = animator.add_animation(vert_arrow.draw_step, [1, smooth], duration=40, parent_animation=empty_anim, delay=80)

    fade_logo = animator.add_animation(dvd.fade_fill_step, [0.22, smooth], duration=40, parent_animation=empty_anim, delay=80)
    # draw_velnum2 = animator.add_animation(vert_vel_num.draw_step, [1, smooth], duration=40, parent_animation=empty_anim, delay=80)

    # morph_num = animator.add_animation(horiz_vel_num.morph_step, [vert_vel_num, smooth], duration=40, parent_animation=fade_logo, delay=10)
    # stretch = animator.add_animation(horiz_arrow.stretch_shrink, [2, 1, smooth], duration=40,
    #                                    parent_animation=fade_logo, delay=10)

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

def center_point():
    global stop, main_dir, ps_files_dir, png_files_dir, img_list

    # Object Creation
    r = 40
    screen = Rectangle(-window_w / 2 + r, window_h / 2 - 150, window_w / 2 - r, -window_h / 2 + 40,
                       apple_colors['lightred'],
                       stop=1, drawing_point_delta=0.001)

    v = -10
    dvd = DVDLogoObject(0.5, 0, 0, v, -v, apple_colors['lightteal'], drawing_point_delta=0.003, stop=1, fill_p=0.1)  # fill_p = 0.5

    point = DVDLogoMidpoint(dvd, 10, (255,0,0), fill_p=0.5, stop=0, drawing_point_delta=0.005)
    inner_screen = Rectangle(-screen.length/2 + dvd.length/2, screen.height/2 + screen.offsets_xy[1] - dvd.height/2,
                             screen.length/2 - dvd.length/2, -screen.height/2 + screen.offsets_xy[1] + dvd.height/2,
                             apple_colors['lightblue'], stop=0, drawing_point_delta=0.001)

    objects = [dvd, screen, point, inner_screen]

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()
    evolve = animator.add_animation(dvd.bounce_around, [screen], duration=2000, parent_animation=empty_anim, delay=0)

    follow = animator.add_animation(point.follow_logo, [], duration=2000, parent_animation=empty_anim, delay=0)
    wait = 300
    draw_bouncer = animator.add_animation(point.draw_step, [1, smooth], duration=30, parent_animation=empty_anim, delay=wait)
    fade_out = animator.add_animation(dvd.fade_fill_step, [0.2, smooth], duration=30, parent_animation=empty_anim, delay=wait)
    draw_inner = animator.add_animation(inner_screen.draw_step, [1, smooth], duration=50, parent_animation=fade_out, delay=50)

    fadeout_logo = animator.add_animation(dvd.fade_fill_step, [0, smooth], duration=50, parent_animation=draw_inner, delay=220)
    morph = animator.add_animation(screen.morph_step, [inner_screen, smooth], duration=30, parent_animation=fadeout_logo, delay=0)


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

def elastic_collisions():
    global stop, main_dir, ps_files_dir, png_files_dir, img_list

    # Object Creation
    r = 40
    screen = Rectangle(-window_w / 2 + r, window_h / 2 - 150, window_w / 2 - r, -window_h / 2 + 40,
                       apple_colors['lightred'], stop=1, drawing_point_delta=0.001)

    v = 10
    bouncer = SmallBouncer(550, -300, 5, v, v, color=apple_colors['lightteal'], fill_p=0.4, stop=0, drawing_point_delta=0.02)
    trajectory = BounceTrajectory(bouncer, screen, (255, 255, 255), fill_p=-1, stop=1, drawing_point_delta=0.005)

    angle1 = BounceAngleArc(60, color=apple_colors['lightyellow'], stop=0, drawing_point_delta=0.001)

    objects = [bouncer, screen, trajectory, angle1]

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()
    draw_bouncer = animator.add_animation(bouncer.draw_step, [1, smooth], duration=30, parent_animation=empty_anim, delay=0)
    bounce_around = animator.add_animation(bouncer.bounce_around, [screen], duration=2000, parent_animation=draw_bouncer, delay=0)

    draw_angle = animator.add_animation(angle1.draw_angle_arc, [trajectory, 1, smooth], duration=15, parent_animation=draw_bouncer, delay=80)
    erase = animator.add_animation(trajectory.undraw, [1, smooth], duration=20, parent_animation=draw_angle, delay=50)
    # erase = animator.add_animation(trajectory.undraw, [1, smooth], duration=20, parent_animation=erase, delay=110)

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

def ok_zoomer():
    global stop, main_dir, ps_files_dir, png_files_dir, img_list

    # Object Creation
    text = Text('Ok Zoomer', 8, apple_colors['lightyellow'], spacing=10, stop=0, left_x=-400, bottom_y=0, drawing_delta_per_glyph=0.01)
    box = Rectangle(-430, 150, 410, -65, apple_colors['lightindigo'], stop=0, drawing_point_delta=0.001)

    objects = [text, box]

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()
    draw_text = animator.add_animation(text.draw_step, [1, smooth], duration=50, parent_animation=empty_anim, delay=0)
    embellish = animator.add_animation(box.draw_step, [1, smooth], duration=30, parent_animation=draw_text, delay=20)
    embellish2 = animator.add_animation(box.undraw, [1, smooth], duration=30, parent_animation=draw_text, delay=30)

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

def sandbox():
    global stop, main_dir, ps_files_dir, png_files_dir, img_list

    # Object Creation
    three = Glyph('3', 8, apple_colors['lightyellow'], stop=0, left_x=0, bottom_y=0, drawing_points_delta=0.01)
    four = Glyph('5', 8, apple_colors['lightyellow'], stop=0, left_x=0, bottom_y=0, drawing_points_delta=0.01)

    objects = [three]

    # Animation Tree Construction
    animator = Animator()
    empty_anim = animator.get_root()
    draw_num = animator.add_animation(three.draw_step, [1, smooth], duration=50, parent_animation=empty_anim, delay=0)
    change_num = animator.add_animation(three.morph_step, [four, smooth], duration=50, parent_animation=draw_num, delay=0)

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
    # basic_logo_bounce_anim()
    # bounding_rectangle()
    # balanced_velocity()
    # center_point()
    elastic_collisions()

    # Fun extra anims
    # ok_zoomer()

    # Sandbox
    # sandbox()



# Necessary line for Tkinter
mainloop()