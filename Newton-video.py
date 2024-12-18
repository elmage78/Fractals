#Importamos os módulos e os comandos de Python necesarios:
import cxroots.root_approximation
import numpy as np
import sympy as sp
from sympy import Symbol,Derivative,simplify,lambdify
import matplotlib.pyplot as plt
from manim import *
import cxroots

ROOT_COLORS_BRIGHT = [RED, GREEN, BLUE]
ROOT_COLORS_DEEP = ["#440154", "#3b528b", "#21908c", "#5dc963", "#29abca"]
ROOT_COLORS_DEEP = [ManimColor.from_hex(Molor) for Molor in ROOT_COLORS_DEEP]

class TrueFractal(Mobject):
    data = {
        "shader_folder": "newton_fractal",
        "data_dtype": [
            ('point', np.float32, (3,)),
        ],
        "colors": ROOT_COLORS_DEEP,
        "coefs": [1.0, -1.0, 1.0, 0.0, 0.0, 1.0],
        "scale_factor": 1.0,
        "offset": ORIGIN,
        "n_steps": 30,
        "julia_highlight": 0.0,
        "max_degree": 5,
        "saturation_factor": 0.0,
        "opacity": 1.0,
        "black_for_cycles": False,
        "is_parameter_space": False,
    }

    def __init__(self, plane, **kwargs):
        super().__init__(
            scale_factor=plane.get_x_unit_size(),
            offset=plane.n2p(0),
            **kwargs,
        )
        self.replace(plane, stretch=True)

    def init_data(self):
        self.set_points([UL, DL, UR, DR])

    def init_uniforms(self):
        super().init_uniforms()
        self.set_colors(self.colors)
        self.set_julia_highlight(self.julia_highlight)
        self.set_coefs(self.coefs)
        self.set_scale(self.scale_factor)
        self.set_offset(self.offset)
        self.set_n_steps(self.n_steps)
        self.set_saturation_factor(self.saturation_factor)
        self.set_opacity(self.opacity)
        self.uniforms["black_for_cycles"] = float(self.black_for_cycles)
        self.uniforms["is_parameter_space"] = float(self.is_parameter_space)

    def set_colors(self, colors):
        self.uniforms.update({
            f"color{n}": np.array(color_to_rgba(color))
            for n, color in enumerate(colors)
        })
        return self

    def set_julia_highlight(self, value):
        self.uniforms["julia_highlight"] = value

    def set_coefs(self, coefs, reset_roots=True):
        full_coefs = [*coefs] + [0] * (self.max_degree - len(coefs) + 1)
        self.uniforms.update({
            f"coef{n}": np.array([coef.real, coef.imag], dtype=np.float64)
            for n, coef in enumerate(map(complex, full_coefs))
        })
        if reset_roots:
            self.set_roots(coefficients_to_roots(coefs), False)
        self.coefs = coefs
        return self

    def set_roots(self, roots, reset_coefs=True):
        self.uniforms["n_roots"] = float(len(roots))
        full_roots = [*roots] + [0] * (self.max_degree - len(roots))
        self.uniforms.update({
            f"root{n}": np.array([root.real, root.imag], dtype=np.float64)
            for n, root in enumerate(map(complex, full_roots))
        })
        if reset_coefs:
            self.set_coefs(roots_to_coefficients(roots), False)
        self.roots = roots
        return self

    def set_scale(self, scale_factor):
        self.uniforms["scale_factor"] = scale_factor
        return self

    def set_offset(self, offset):
        self.uniforms["offset"] = np.array(offset)
        return self

    def set_n_steps(self, n_steps):
        self.uniforms["n_steps"] = float(n_steps)
        return self

    def set_saturation_factor(self, saturation_factor):
        self.uniforms["saturation_factor"] = float(saturation_factor)
        return self

    def set_opacities(self, *opacities):
        for n, opacity in enumerate(opacities):
            self.uniforms[f"color{n}"][3] = opacity
        return self

    def set_opacity(self, opacity, recurse=True):
        self.set_opacities(*len(self.roots) * [opacity])
        return self
    
class NewtonFractal(Mobject):

    def __init__(self, plane,coefs=[1.0, -1.0, 1.0, 0.0, 0.0, 1.0],n_steps=30,colors=ROOT_COLORS_DEEP, **kwargs):
        super().__init__(
            **kwargs
        )
        self.data_dtype= [
            ('point', np.float32, (3,))
        ]
        self.colors= colors
        self.coefs= coefs
        self.n_steps= n_steps
        self.julia_highlight= 0.0
        self.max_degree= 5
        self.saturation_factor= 0.0
        self.opacity= 1.0
        self.black_for_cycles= False
        self.is_parameter_space= False
        self.init_uniforms()
        self.replace(plane, stretch=True)

    def init_data(self):
        self.set_points([UL, DL, UR, DR])

    def Plot_start(self,**kwargs):
        self.plane.plot_surface(**kwargs)


    def set_coefs(self, coefs, reset_roots=True):
        self.coefs
        if reset_roots:
            self.set_roots(np.roots(coefs), False)
        self.coefs = coefs
        return self

    def set_roots(self, roots, reset_coefs=True):
        self.n_roots = float(len(roots))
        self.full_roots = [*roots] + [0] * (self.max_degree - len(roots))
        if reset_coefs:
            self.set_coefs(np.roots(roots), False)
        self.roots = roots
        return self

    def set_opacities(self, *opacities):
        for colorm, opacity in enumerate(self.colors,opacities):
             colorm[3] = opacity
        return self

    def set_opacity(self, opacity, recurse=True):
        self.set_opacities(*len(self.roots) * [opacity])
        return self

class NewtonRhapsonFractal(Scene):
    def Pfunc(self,val):
            return val/(val**2+val)**(1/2)
        
    def Pdfunc(self,val):
        return ((-2 * val**2 + 2 *(val**2 + val) - val))/((2*(val**2)*(val**2 + val)**(1/2) + 2*val * (val**2 + val)**(1/2)))


    def func(self,val):
        if (val >= -1.1 and val <= 0.02):
            return 10
        return val/(val**2+val)**(1/2)
    
    def Difffunc(self,val):
        if (val >= -1.1 and val <= 0.02):
            return 10
        return ((-2 * val**2 + 2 *(val**2 + val) - val))/((2*(val**2)*(val**2 + val)**(1/2) + 2*val * (val**2 + val)**(1/2)))
    
    def Newton(self,val):
        return val-(self.Pfunc(val)/self.Pdfunc(val))
    
    def CalcPointN(self, Num):
            if Num.real == 0 and Num.imag == 0:
                return 30
            tol=1.0e-6
            z = Num
            n=0
            while (n<30 and abs(self.Pfunc(z))>tol):
                if abs(self.Newton(z))<1/tol: # si el numero se va al infinito sale con x iteraciones (cuanto tarda en ir al infinito)
                    z=self.Newton(z) # si llega a 30 iteraciones podemos aproximar su convergencia
                    n=n+1
                else:
                    break
            return n
    
    def construct(self,val):
        #O fractal representarase no rectángulo [a,b]x[c,d]: 
        maxiter=30
        npuntos=50
        a=-1.1
        b=0.6
        c=-0.5
        d=0.5
        #Plane construction
        plane = ComplexPlane(
        ).add_coordinates().set_z_index(0).scale_to_fit_width(config.frame_width)

class ComplexNewtonsMethod(NewtonRhapsonFractal):
    RootPtsAprox = [complex(0.12320000000000003,3.984723764762459e-33)] # other root is inf
    poly_tex = "\\frac{z}{\sqrt{z^2+z}}"
    a=-5
    b=5
    c=-2.5
    d=2.5
    plane_config = {
        "x_range": (a,b),
        "y_range": (c, d),
        "x_length": config.frame_width,
        "y_length": config.frame_height
    }
    seed = complex(-0.5, 0.5)
    seed_tex = "-0.5 + 0.5i"
    guess_color = YELLOW
    pz_color = MAROON_B
    step_arrow_width = 0.5
    step_arrow_opacity = 1.0
    step_arrow_len = None
    n_search_steps = 9

    def construct(self):
        self.add_plane()
        self.add_title()
        self.add_z0_def()
        self.add_pz_dot()
        self.add_rule()
        self.find_root()

    def add_plane(self):
        plane = ComplexPlane(**self.plane_config)
        plane.add_coordinates(font_size=24)
        plane.to_edge(RIGHT, buff=0)
        self.plane = plane
        self.add(plane)


    def add_titleSub(self, axes, opacity=0): #super().add_title
        title = Tex("Newton's method", font_size=40)
        title.move_to(midpoint(axes.get_left(), LEFT))
        title.to_edge(UP)
        title.set_opacity(opacity)

        poly = Tex(r"P(z) = $"+ self.poly_tex+ "$ = 0 ")
        poly.match_width(title)
        poly.next_to(title, DOWN, buff=MED_LARGE_BUFF)
        poly.set_fill(GREY_A)
        title.add(poly)

        self.title = title
        self.poly = poly
        self.play(Write(self.title))

    def add_title(self, opacity=1):
        self.add_titleSub(self.plane, opacity)
    

    def add_z0_def(self):
        seed_text = Tex("(Arbitrary seed)")
        z0_def = Tex(
            r"$z_0$ = {"+str(self.seed_tex)+"}",
            font_size=self.rule_font_size
        ).set_color_by_tex_to_color_map({"z_0": self.guess_color})

        guess_dot = Dot(self.plane.n2p(self.seed), color=self.guess_color)

        guess = DecimalNumber(self.seed, num_decimal_places=3, font_size=30)
        guess.add_updater(
            lambda m: m.set_value(self.plane.p2n(
                guess_dot.get_center()
            )).set_fill(self.guess_color).add_background_rectangle()
        )
        guess.add_updater(lambda m: m.next_to(guess_dot, UP, buff=0.15))

        self.play(
            Write(seed_text, run_time=1),
            FadeIn(z0_def),
        )
        self.play(
            FadeTransform(z0_def[0].copy(), guess_dot),
            FadeIn(guess),
        )
        self.wait()

        self.z0_def = z0_def
        self.guess_dot = guess_dot
        self.guess = guess

    def add_pz_dot(self):
        plane = self.plane
        guess_dot = self.guess_dot

        def get_pz():
            z = plane.p2n(guess_dot.get_center())
            return self.Pfunc(z)

        pz_dot = Dot(color=self.pz_color)
        pz_dot.add_updater(lambda m: m.move_to(plane.n2p(get_pz())))
        pz_label = Text("P(z)", font_size=24)
        pz_label.set_color(self.pz_color)
        pz_label.add_background_rectangle()
        pz_label.add_updater(lambda m: m.next_to(pz_dot, UL, buff=0))

        self.play(
            FadeTransform(self.poly[0], pz_label),
            FadeIn(pz_dot),
        )
        self.wait()

    def get_update_rule(self, char="x", fsize=42):#also is on super()
        rule = Tex(
            r"""
                $z_1$ =
                $z_0$ - $\frac{z_0}{z_1}$
            """.replace("z", char),
            font_size=fsize,
        ).set_color_by_tex_to_color_map({"{"+char+"}_1": self.guess_color,"{"+char+"}_0": self.guess_color})

        rule.n = 0
        rule.zns = rule.get_parts_by_tex(f"{char}_0")
        rule.znp1 = rule.get_parts_by_tex(f"{char}_1")
        self.rule_font_size = fsize
        return rule

    def add_rule(self):
        self.rule = rule = self.get_update_rule("z")
        rule.next_to(self.z0_group, DOWN, buff=LARGE_BUFF)
#
        #self.play(
        #    FadeTransformPieces(self.z0_def[0].copy(), rule.zns),
        #    FadeIn(rule),
        #)
        self.wait()

    def find_root(self):
        for x in range(self.n_search_steps):
            self.root_search_step()

    def root_search_step(self):
        dot = self.guess_dot
        dot_step_anims = self.get_dot_step_anims(VGroup(dot))
        diff_rect = SurroundingRectangle(
            self.rule.slice_by_tex("-"),
            buff=0.1,
            stroke_color=GREY_A,
            stroke_width=1,
        )

        self.play(
            Create(diff_rect),
            dot_step_anims[0],
        )
        self.play(
            dot_step_anims[1],
            FadeOut(diff_rect),
            *self.cycle_rule_entries_anims(),
            run_time=2
        )
        self.wait()

    def cycle_rule_entries_anims(self):
        rule = self.rule
        rule.n += 1
        char = rule.get_tex_string().strip()[1]
        zns = VGroup()
        for old_zn in rule.zns:
            zn = Tex(r"${"+f"{char}"+"}_{"+f"{str(rule.n)}"+"}$", font_size=self.rule_font_size)
            zn[0][1:].set_max_width(0.2)
            zn.move_to(old_zn)
            zn.match_color(old_zn)
            zns.add(zn)
        znp1 = Tex(r"${"+str(char)+"}_{"+str(rule.n + 1)+"}$", font_size=self.rule_font_size)
        znp1.move_to(rule.znp1)
        znp1.match_color(rule.znp1[0])

        result = (
            FadeOut(rule.zns),
            FadeTransformPieces(rule.znp1, zns),
            FadeIn(znp1,SHIFT_VALUE=0.5 * RIGHT)
        )
        rule.zns = zns
        rule.znp1 = znp1
        return result
    
    def get_dot_step_anims(self, dots):
        plane = self.plane
        arrows = VGroup()
        dots.generate_target()
        for dot, dot_target in zip(dots, dots.target):
            try:
                z0 = plane.p2n(dot.get_center())
                if (z0 == 0):
                    pz = 0
                    dpz = 0
                else:
                    pz = self.Pfunc(z0)
                    dpz = self.Pdfunc(z0)
                if (abs(z0 - complex(100,100)) >= 1e-4):
                    if abs(pz) < 1e-3:
                        z1 = z0
                    else:
                        if dpz == 0:
                            dpz = 0.1  # ???
                        z1 = z0 - pz / dpz #Substitute by Newton, Halley OR chebysev method accordingly

                    if np.isnan(z1):
                        z1 = z0

                    if (abs(z1) >= 1/(1e-4) or abs(z0) == 1/(1e-4)):
                        z0 = complex(100,100)
                        z1 = complex(100,100)
                elif 'z1' not in locals():
                    z1 = z0

                arrow = Arrow(
                    #the arrow pointing to X place
                    plane.n2p(z0), plane.n2p(z1),
                    buff=0.1,
                    stroke_width=self.step_arrow_width,
                    stroke_opacity=self.step_arrow_opacity,
                    max_stroke_width_to_length_ratio=0.1,
                    max_tip_length_to_length_ratio=0.1,
                )
                if self.step_arrow_len is not None:
                    if arrow.get_length() > self.step_arrow_len:
                        arrow.set_length(self.step_arrow_len)
                        

                if arrow.get_tip().get_stroke_width() > 0.05:
                        arrow.get_tip().set_stroke_width(0.05)

                if not hasattr(dot, "history"):
                    dot.history = [dot.get_center().copy()]
                dot.history.append(plane.n2p(z1))

                arrows.add(arrow)
                dot_target.move_to(plane.n2p(z1))
            except ValueError:
                pass
        return [
            Create(arrows, lag_ratio=0),
            AnimationGroup(
                MoveToTarget(dots),
                FadeOut(arrows),
            )
        ]

class ComplexNewtonsMethodManySeeds(ComplexNewtonsMethod):
    dot_radius = 0.035
    dot_color = WHITE
    dot_opacity = 0.8
    step_arrow_width = 0.5
    step_arrow_opacity = 0.1
    step_arrow_len = 0.1
    step = 0.2
    n_search_steps = 20
    colors = ROOT_COLORS_BRIGHT
    rule_font_size = 24

    def construct(self):
        self.add_plane()
        self.add_title()
        self.add_z0_def()
        self.add_rule()
        self.add_true_root_circles()
        self.find_root()
        self.add_color()
        self.wait()

    def add_z0_def(self):
        title = self.title
        poly = self.poly
        z0_def = Tex(r"$z_0$",
            font_size=self.rule_font_size
        ).set_color_by_tex_to_color_map({"z_0": self.guess_color})
        z0_group = VGroup(title, z0_def,poly)
        z0_group.arrange(DOWN)
        z0_group.next_to(self.poly, DL, buff=LARGE_BUFF)

        x_range = self.plane_config["x_range"]
        y_range = self.plane_config["y_range"]
        step = self.step
        x_vals = np.arange(x_range[0], x_range[1] + step, step)
        y_vals = np.arange(y_range[0], y_range[1] + step, step)
        guess_dots = VGroup(*(
            Dot(
                self.plane.c2p(x, y),
                radius=self.dot_radius,
                fill_opacity=self.dot_opacity,
            )
            for i, x in enumerate(x_vals)
            for y in (y_vals if i % 2 == 0 else reversed(y_vals))
        ))
        guess_dots.set_submobject_colors_by_gradient(WHITE, GREY_B)
        guess_dots.set_fill(opacity=self.dot_opacity)
        guess_dots.set_stroke(BLACK, 2, background=True)

        self.play(
            FadeIn(z0_def),
        )
        print("start puttin dots")
        self.play(
            LaggedStart(*(
                FadeTransform(z0_def[0].copy(), guess_dot)
                for guess_dot in guess_dots
            ), lag_ratio=0.1 / len(guess_dots)),
            FadeOut(z0_def),
            run_time=3
        )
        print("finish puttin dots")
        self.add(guess_dots)
        self.wait()

        self.z0_group = z0_group
        self.z0_def = z0_def
        self.poly = poly
        self.title = poly
        self.guess_dots = guess_dots

    def add_true_root_circles(self):
        roots = [complex(i.real,i.imag) for i in self.RootPtsAprox]
        root_points = list(map(self.plane.n2p, roots))
        colors = self.colors

        root_circles = VGroup(*(
            Dot(radius=0.1).set_fill(color, opacity=0.75).move_to(rp)
            for rp, color in zip(root_points, colors)
        ))

        self.play(
            LaggedStart(*(
                FadeIn(rc, scale=0.5)
                for rc in root_circles
            ), lag_ratio=0.7, run_time=1),
        )
        self.wait()

        self.root_circles = root_circles

    def root_search_step(self):
        dots = self.guess_dots
        dot_step_anims = self.get_dot_step_anims(dots)

        self.play(dot_step_anims[0], run_time=0.25)
        self.play(
            dot_step_anims[1],
            #*self.cycle_rule_entries_anims(),
            run_time=1
        )

    def add_color(self):
        
        plane = self.plane
        root_points = [circ.get_center() for circ in self.root_circles]
        colors = [circ.get_fill_color() for circ in self.root_circles]

        dots = self.guess_dots
        dots.generate_target()
        for dot, dot_target in zip(dots, dots.target):
            Multiplier = self.CalcPointN(plane.p2n(dot.history[0]))
            dot_target.set_color(ManimColor.from_rgb((((Multiplier/30)*255,((30-Multiplier)/30)*255,0))))

        rect = SurroundingRectangle(self.rule)
        rect.set_fill(BLACK, 1)
        rect.set_stroke(width=0)

        self.play(
            FadeIn(rect),
            MoveToTarget(dots)
        )
        self.wait()

        len_history = max([len(dot.history) for dot in dots if hasattr(dot, "history")], default=0)
        for n in range(len_history):
            dots.generate_target()
            for dot, dot_target in zip(dots, dots.target):
                try:
                    dot_target.move_to(dot.history[len_history - n - 1])
                except Exception:
                    pass
            self.play(MoveToTarget(dots, run_time=0.5))

class ComplexNewtonsMethodManyManySeeds(ComplexNewtonsMethodManySeeds):
    step = 0.1
    dot_radius = 0.020
    n_search_steps = 15
    def construct(self):
        self.add_plane()
        self.add_title()
        self.wait()
        self.play(Unwrite(self.title),run_time = 2)
        self.add_z0_def()
        self.add_rule()
        self.add_true_root_circles()
        self.find_root()
        self.add_color()
        self.wait()

class IntroNewtonFractal(MovingCameraScene):
    coefs = [1.0, -1.0, 1.0, 0.0, 0.0, 1.0]
    plane_config = {
        "x_range": (-4, 4),
        "y_range": (-4, 4),
        "x_length": 16,
        "y_length": 16,
        "background_line_style": {
            "stroke_color": GREY_A,
            "stroke_width": 1.0,
        },
        "axis_config": {
            "stroke_width": 1.0,
        }
    }
    n_steps = 30

    def construct(self):
        self.init_fractal(root_colors=ROOT_COLORS_BRIGHT)
        fractal, plane, root_dots = self.group

        # Transition from last scene
        frame = self.camera.frame
        frame.shift(plane.n2p(2) - RIGHT)

        blocker = BackgroundRectangle(plane, fill_opacity=1)
        blocker.move_to(plane.n2p(-2), RIGHT)
        self.add(blocker)

        self.play(
            frame.animate.center(),
            FadeOut(blocker),
            run_time=2,
        )
        self.wait()
        self.play(
            fractal.animate.set_colors(ROOT_COLORS_DEEP),
            *(
                dot.animate.set_fill(interpolate_color(color, WHITE, 0.2))
                for dot, color in zip(root_dots, ROOT_COLORS_DEEP)
            )
        )
        self.wait()

        # Zoom in
        fractal.set_n_steps(40)
        zoom_points = [
            [-3.12334879, 1.61196545, 0.],
            [1.21514006, 0.01415811, 0.],
        ]
        for point in zoom_points:
            self.play(
                frame.animate.set_height(2e-3).move_to(point),
                run_time=25,
                rate_func=bezier(2 * [0] + 6 * [1])
            )
            self.wait()
            self.play(
                frame.animate.center().set_height(8),
                run_time=10,
                rate_func=bezier(6 * [0] + 2 * [1])
            )

        # Allow for play
        self.tie_fractal_to_root_dots(fractal)
        fractal.set_n_steps(12)

    def init_fractal(self, root_colors=ROOT_COLORS_DEEP):
        plane = self.get_plane()
        fractal = self.get_fractal(
            plane,
            colors=root_colors,
            n_steps=self.n_steps,
        )
        root_dots = self.get_root_dots(plane, fractal)
        self.tie_fractal_to_root_dots(fractal)

        self.plane = plane
        self.fractal = fractal
        self.group = Group(fractal, plane, root_dots)
        self.add(*self.group)

    def get_plane(self):
        plane = ComplexPlane(**self.plane_config)
        plane.add_coordinates(font_size=24)
        self.plane = plane
        return plane

    def get_fractal(self, plane, colors=ROOT_COLORS_DEEP, n_steps=30):
        return NewtonFractal(
            plane,
            colors=colors,
            coefs=self.coefs,
            n_steps=n_steps,
        )

    def get_root_dots(self, plane, fractal):
        self.root_dots = VGroup(*(
            Dot(plane.n2p(root), color=color)
            for root, color in zip(
                np.roots(fractal.coefs),
                fractal.colors
            )
        ))
        self.root_dots.set_stroke(BLACK, 5, background=True)
        return self.root_dots

    def tie_fractal_to_root_dots(self, fractal):
        fractal.add_updater(lambda f: f.set_roots([
            self.plane.p2n(dot.get_center())
            for dot in self.root_dots
        ]))

    def on_mouse_press(self, point, button, mods):
        super().on_mouse_press(point, button, mods)
        mob = self.point_to_mobject(point, search_set=self.root_dots)
        if mob is None:
            return
        self.mouse_drag_point.move_to(point)
        mob.add_updater(lambda m: m.move_to(self.mouse_drag_point))
        self.unlock_mobject_data()
        self.lock_static_mobject_data()

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        self.root_dots.clear_updaters()

class Movie(Scene):
    def construct(self):
        ComplexNewtonsMethod.construct()
        ComplexNewtonsMethodManySeeds.construct()
        ComplexNewtonsMethodManyManySeeds.construct()
        IntroNewtonFractal.construct()