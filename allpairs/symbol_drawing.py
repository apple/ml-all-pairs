"""a doc string"""

from __future__ import print_function
import numpy as np
import math
from skimage.draw import line_aa, circle_perimeter_aa


def draw_symbol(image, symbol_index):
    """draw the symbol specified into the image provided"""
    assert image.shape[0] == image.shape[1]
    sym_size = image.shape[0]

    def r(min_val, max_val):
        """a random int in [min_val, max_val]"""
        return np.random.randint(min_val, max_val + 1)

    def inset(delta):
        """random inset into image"""
        return r(delta, sym_size - 1 - delta)

    def inner_circle(im):
        """returns centerX, centerY, radius"""
        c_x = inset(5)
        c_y = inset(5)
        max_r = min(c_x, c_y)
        max_r = min(max_r, sym_size - 1 - c_x)
        max_r = min(max_r, sym_size - 1 - c_y)
        radius = max_r
        while radius > 4:
            if r(0, 1) == 1:
                radius -= 1
            else:
                break

        rr, cc, vals = circle_perimeter_aa(c_y, c_x, radius, shape=im.shape)
        im[rr, cc] += vals
        return c_x, c_y, radius

    # def within(minX, minY, w, h):
    #     """doc string"""
    #     x = r(minX, minX + w)
    #     y = r(minY, minY + h)
    #     return x, y

    def add_line(im, x1, y1, x2, y2):
        rr, cc, vals = line_aa(y1, x1, y2, x2)
        im[rr, cc] += vals

    def r_coord(min_val, max_val):
        """get a random coordinate"""
        return r(max(0, min_val), min(sym_size - 1, max_val))

    def circle(im, center_border=4, min_radius=4):
        """a circle, may be clipped"""
        rr, cc, vals = circle_perimeter_aa(
                            inset(center_border), inset(center_border),
                            r(min_radius, sym_size // 2 - 1), shape=im.shape
                        )
        im[rr, cc] = vals

    def line(im):
        """a line crossing the patch"""
        if r(0, 1):
            rr, cc, vals = line_aa(0, inset(1), sym_size - 1, inset(1))
        else:
            rr, cc, vals = line_aa(inset(1), 0, inset(1), sym_size - 1)

        im[rr, cc] = vals

    def cross(im):
        """two lines crossing the patch"""
        rr, cc, vals = line_aa(0, inset(3), sym_size - 1, inset(3))
        im[rr, cc] = vals
        rr, cc, vals = line_aa(inset(3), 0, inset(3), sym_size - 1)
        im[rr, cc] = vals

    def rays(num_rays, im, top, left, w, h, inset_amt=3):
        """a number of rays, 0 to 4, radiating from a point"""
        c_x = r(left, left + w)
        c_y = r(top, top + h)
        dest = [
            [inset(inset_amt), 0],
            [inset(inset_amt), sym_size - 1],
            [0, inset(inset_amt)],
            [sym_size - 1, inset(inset_amt)],
        ]
        np.random.shuffle(dest)
        for i in range(0, num_rays):
            rr, cc, vals = line_aa(c_y, c_x, dest[i][0], dest[i][1])
            im[rr, cc] += vals

    def angle(im, mid_border=4):
        """two lines forming an angle in the patch"""
        mid_x = inset(mid_border)
        mid_y = inset(mid_border)
        if r(0, 1):
            rr, cc, vals = line_aa(0, inset(4), mid_x, mid_y)
            im[rr, cc] = vals
        else:
            rr, cc, vals = line_aa(sym_size - 1, inset(4), mid_x, mid_y)
            im[rr, cc] = vals
        if r(0, 1):
            rr, cc, vals = line_aa(inset(4), 0, mid_x, mid_y)
            im[rr, cc] = vals
        else:
            rr, cc, vals = line_aa(inset(4), sym_size - 1, mid_x, mid_y)
            im[rr, cc] = vals

    def phi(im):
        """a circle with a semi-vertical line"""
        c_x, c_y, radius = inner_circle(im)
        x1 = r_coord(c_x - radius//2, c_x + radius//2)
        x3 = r_coord(c_x - radius//2, c_x + radius//2)
        x2 = c_x + r(-1, 1)
        top_y = max(0, c_y - radius - r(0, 3))
        bottom_y = min(sym_size - 1, c_y + radius + r(0, 3))
        c_y += r(-1, 1)
        add_line(im, x2, c_y, x1, top_y)
        add_line(im, x2, c_y, x3, bottom_y)

    def theta(im):
        """a circle with a semi-horizontal line"""
        c_x, c_y, radius = inner_circle(im)
        y1 = r_coord(c_y - radius//2, c_y + radius//2)
        y3 = r_coord(c_y - radius//2, c_y + radius//2)
        y2 = c_y + r(-1, 1)
        left_x = max(0, c_x - radius - r(0, 3))
        right_x = min(sym_size - 1, c_x + radius + r(0, 3))
        c_x += r(-1, 1)
        add_line(im, c_x, y2, left_x, y1)
        add_line(im, c_x, y2, right_x, y3)

    def three_star(im):
        """three radiating lines"""
        mid_x = inset(4)
        mid_y = inset(4)
        dest = [(0, inset(3)), (sym_size - 1, inset(3)), (inset(3), 0), (inset(3), sym_size - 1)]
        np.random.shuffle(dest)
        for i in range(3):
            rr, cc, vals = line_aa(dest[i][0], dest[i][1], mid_x, mid_y)
            im[rr, cc] += vals

    def double_circle(im):
        """two circles, less clipped that the single circle"""
        c_x, c_y, radius = inner_circle(im)
        radians = 2*np.pi * np.random.uniform()
        x = c_x + radius * math.cos(radians)
        y = c_y + radius * math.sin(radians)
        r2 = r(3, radius)
        rr, cc, vals = circle_perimeter_aa(int(round(y)), int(round(x)), r2, shape=im.shape)
        im[rr, cc] += vals

    def circle_3star(im):
        """overlap circle and 3-star"""
        c_x, c_y, radius = inner_circle(im)
        rays(3, im, c_y - 2, c_x - 2, 5, 5)

    def box(im, ordering=((0, 1), (1, 2), (2, 3), (3, 0))):
        """put a rectangle in the patch, im"""
        rad = r(sym_size // 4, sym_size // 3)
        c_x = sym_size / 2.0
        c_y = sym_size / 2.0
        ang = 2*np.pi * np.random.uniform()
        a = np.pi/8.0 + np.pi/8.0 * np.random.uniform()
        p = (
            (int(round(c_y + rad * math.sin(ang + a))), int(round(c_x + rad * math.cos(ang + a)))),
            (int(round(c_y + rad * math.sin(ang - a))), int(round(c_x + rad * math.cos(ang - a)))),
            (int(round(c_y + rad * math.sin(ang + a + np.pi))), int(round(c_x + rad * math.cos(ang + a + np.pi)))),
            (int(round(c_y + rad * math.sin(ang - a + np.pi))), int(round(c_x + rad * math.cos(ang - a + np.pi)))),
        )
        for i, j in ordering:
            add_line(im, p[i][0], p[i][1], p[j][0], p[j][1])

    def triangle(im):
        """put a triangle in the patch, im"""
        radius = r(sym_size // 4, sym_size // 3)
        c_x = sym_size / 2.0
        c_y = sym_size / 2.0
        ang = 2*np.pi * np.random.uniform()
        a = np.pi/4.0 + np.pi/2.0 * np.random.uniform()
        b = np.pi/4.0 + np.pi/2.0 * np.random.uniform()
        p = (
            (int(round(c_y + radius*math.sin(ang))),     int(round(c_x + radius*math.cos(ang)))),
            (int(round(c_y + radius*math.sin(ang + a))), int(round(c_x + radius*math.cos(ang + a)))),
            (int(round(c_y + radius*math.sin(ang - b))), int(round(c_x + radius*math.cos(ang - b)))),
        )
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            add_line(im, p[i][0], p[i][1], p[j][0], p[j][1])

    def box_diag(im):
        """put a rectangle with a diagonal in the patch, im"""
        box(im, ordering=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])

    def hourglass(im):
        """put a hourglass like shape in the patch, im"""
        box(im, ordering=[(0, 1), (1, 3), (3, 2), (2, 0)])

    def triangle_lid(im):
        """put a box with an open lid shape in the patch, im"""
        box(im, ordering=[(0, 1), (1, 2), (2, 3), (3, 1)])

    def zshape(im):
        """put a z-shape in the patch, im"""
        box(im, ordering=[(0, 1), (1, 3), (3, 2)])

    def add_dot(im, x, y):
        """add a dot at the location in patch, im"""
        for dy in range(-1, 2):
            add_line(im, x - 1, y + dy, x + 1, y + dy)

    def barbell(im, end_mid_end=(True, False, True)):
        """a line with dots at each end"""
        radius = r(sym_size // 4, sym_size // 3)
        c_x = sym_size / 2.0
        c_y = sym_size / 2.0
        ang = 2*np.pi * np.random.uniform()
        p1 = (int(round(c_y + radius*math.sin(ang))),         int(round(c_x + radius*math.cos(ang))))
        p2 = (int(round(c_y + radius*math.sin(ang + np.pi))), int(round(c_x + radius*math.cos(ang + np.pi))))
        add_line(im, p1[0], p1[1], p2[0], p2[1])
        if end_mid_end[0]:
            add_dot(im, p1[0], p1[1])
        if end_mid_end[1]:
            add_dot(im, int(round((p1[0] + p2[0])/2.0)), int(round((p1[1] + p2[1])/2.0)))
        if end_mid_end[2]:
            add_dot(im, p2[0], p2[1])

    def dot_line(im):
        """line with a dot at one end"""
        barbell(im, end_mid_end=(True, False, False))

    def dot_on_line(im):
        """a line with a dot in the middle"""
        barbell(im, end_mid_end=(False, True, False))

    draw = [
        circle,         # 0
        line,           # 1
        cross,          # 2
        angle,          # 3
        three_star,     # 4
        theta,          # 5
        phi,            # 6
        double_circle,  # 7
        circle_3star,   # 8
        box,            # 9
        box_diag,       # 10
        barbell,        # 11
        dot_line,       # 12
        zshape,         # 13
        triangle_lid,   # 14
        dot_on_line,    # 15
        hourglass,      # 16
        triangle,       # 17
        #  num_symbols = 18
    ]
    assert symbol_index < len(draw), "exceeded max symbols with index {}".format(symbol_index)
    draw[symbol_index](image)
