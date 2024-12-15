from manimlib import *
from manimlib.constants import *
# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'E:\\Programacion\\Python\\Manim\\Fractals\\custom')
from shaders import ShaderMobject

def coefficients_to_roots(coefs):
    return np.roots(coefs)
def roots_to_coefficients(roots):
    return np.poly(roots)

def poly(x, coefs):
    return sum(coefs[k] * x**k for k in range(len(coefs)))

def dpoly(x, coefs):
    return sum(k * coefs[k] * x**(k - 1) for k in range(1, len(coefs)))

manim_config.key_bindings.select = 's'
ROOT_COLORS_BRIGHT = [RED, GREEN, BLUE, YELLOW, MAROON_B]
ROOT_COLORS_DEEP = ["#440154", "#3b528b", "#21908c", "#5dc963", "#29abca"]
CUBIC_COLORS = [RED_E, TEAL_E, BLUE_E]


def glow_dot(point, r_min=0.05, r_max=0.15, color=YELLOW, n=20, opacity_mult=1.0):
    result = VGroup(*(
        Dot(point, radius=interpolate(r_min, r_max, a))
        for a in np.linspace(0, 1, n)
    ))
    result.set_fill(color, opacity=opacity_mult / n)
    return result

class NewtonFractal(ShaderMobject):
    CONFIG = {
        "shader_folder": "E:\\Programacion\\Python\\Manim\\Fractals\\Newton_fractal",
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
        "saturation_factor": 1.0,
        "opacity": 1.0,
        "black_for_cycles": False,
        "is_parameter_space": False,
    }

    def __init__(self, plane,colors = ROOT_COLORS_DEEP, coefs = [1.0, -1.0, 1.0, 0.0, 0.0, 1.0], n_steps = 30,
                  **kwargs):
        self.colors=colors
        self.coefs=coefs
        self.n_steps=n_steps
        self.offset=plane.n2p(0)
        self.scale_factor=((abs(plane.get_all_ranges()[0][0])+abs(plane.get_all_ranges()[0][0])))
        super().__init__(
            "Newton_fractal",
            **kwargs
        )
        self.replace(plane, stretch=True)

    def init_data(self):
        super().init_data()
        self.set_points([UL, DL, UR, DR])

    def init_uniforms(self):
        super().init_uniforms()
        self.set_colors(self.colors)
        self.set_julia_highlight(self.CONFIG["julia_highlight"])
        self.set_coefs(self.coefs)
        self.set_scale(self.scale_factor)
        self.set_offset(self.offset)
        self.set_n_steps(self.n_steps)
        self.set_saturation_factor(self.CONFIG["saturation_factor"])
        self.set_opacity(self.opacity)
        self.uniforms["black_for_cycles"] = float(self.CONFIG["black_for_cycles"])
        self.uniforms["is_parameter_space"] = float(self.CONFIG["is_parameter_space"])

    def set_colors(self, colors):
        self.uniforms.update({
            f"color{n}": np.array(color_to_rgba(color))
            for n, color in enumerate(colors)
        })
        return self

    def set_julia_highlight(self, value):
        self.uniforms["julia_highlight"] = value

    def set_coefs(self, coefs, reset_roots=True):
        full_coefs = [*coefs] + [0] * (self.CONFIG["max_degree"] - len(coefs) + 1)
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
        full_roots = [*roots] + [0] * (self.CONFIG["max_degree"] - len(roots))
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


class HalleyFractal(ShaderMobject):
    CONFIG = {
        "shader_folder": "E:\\Programacion\\Python\\Manim\\Fractals\\Halley_fractal",
        "data_dtype": [
            ('point', np.float32, (3,)),
        ],
        "colors": ROOT_COLORS_DEEP,
        "coefs": [1.0, -1.0, 1.0, 0.0, 0.0, 1.0],
        "scale_factor": 1.0,
        "offset": ORIGIN,
        "n_steps": 30,
        "julia_highlight": 0.0,
        "max_degree": 3,
        "saturation_factor": 1.0,
        "opacity": 1.0,
        "black_for_cycles": False,
        "is_parameter_space": False,
    }

    def __init__(self, plane,colors = ROOT_COLORS_DEEP, coefs = [1.0, -1.0, 1.0, 0.0, 0.0, 1.0], n_steps = 30,
                  **kwargs):
        self.colors=colors
        self.coefs=coefs
        self.n_steps=n_steps
        self.offset=plane.n2p(0)
        self.scale_factor=((abs(plane.get_all_ranges()[0][0])+abs(plane.get_all_ranges()[0][0])))
        super().__init__(
            self.CONFIG["shader_folder"],
            **kwargs
        )
        self.replace(plane, stretch=True)

    def init_data(self):
        super().init_data()
        self.set_points([UL, DL, UR, DR])

    def init_uniforms(self):
        super().init_uniforms()
        self.set_colors(self.colors)
        self.set_julia_highlight(self.CONFIG["julia_highlight"])
        self.set_coefs(self.coefs)
        self.set_scale(self.scale_factor)
        self.set_offset(self.offset)
        self.set_n_steps(self.n_steps)
        self.set_saturation_factor(self.CONFIG["saturation_factor"])
        self.set_opacity(self.opacity)
        self.uniforms["black_for_cycles"] = float(self.CONFIG["black_for_cycles"])
        self.uniforms["is_parameter_space"] = float(self.CONFIG["is_parameter_space"])

    def set_colors(self, colors):
        self.uniforms.update({
            f"color{n}": np.array(color_to_rgba(color))
            for n, color in enumerate(colors)
        })
        return self

    def set_julia_highlight(self, value):
        self.uniforms["julia_highlight"] = value

    def set_coefs(self, coefs, reset_roots=True):
        full_coefs = [*coefs] + [0] * (self.CONFIG["max_degree"] - len(coefs) + 1)
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
        full_roots = [*roots] + [0] * (self.CONFIG["max_degree"] - len(roots))
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


class ChebysevFractal(ShaderMobject):
    CONFIG = {
        "shader_folder": "E:\\Programacion\\Python\\Manim\\Fractals\\Chebysev_fractal",
        "data_dtype": [
            ('point', np.float32, (3,)),
        ],
        "colors": ROOT_COLORS_DEEP,
        "coefs": [1.0, -1.0, 1.0, 0.0, 0.0, 1.0],
        "scale_factor": 1.0,
        "offset": ORIGIN,
        "n_steps": 50,
        "julia_highlight": 0.0,
        "max_degree": 5,
        "saturation_factor": 1.0,
        "opacity": 1.0,
        "black_for_cycles": False,
        "is_parameter_space": False,
    }

    def __init__(self, plane,colors = ROOT_COLORS_DEEP, coefs = [1.0, -1.0, 1.0, 0.0, 0.0, 1.0], n_steps = 30,
                  **kwargs):
        self.colors=colors
        self.coefs=coefs
        self.n_steps=n_steps
        self.offset=plane.n2p(0)
        print(plane.get_all_ranges())
        self.scale_factor=((abs(plane.get_all_ranges()[0][0])+abs(plane.get_all_ranges()[0][0])))
        super().__init__(
            self.CONFIG["shader_folder"],
            **kwargs
        )
        self.replace(plane, stretch=True)

    def init_data(self):
        super().init_data()
        self.set_points([UL, DL, UR, DR])

    def init_uniforms(self):
        super().init_uniforms()
        self.set_colors(self.colors)
        self.set_julia_highlight(self.CONFIG["julia_highlight"])
        self.set_coefs(self.coefs)
        self.set_scale(self.scale_factor)
        self.set_offset(self.offset)
        self.set_n_steps(self.n_steps)
        self.set_saturation_factor(self.CONFIG["saturation_factor"])
        self.set_opacity(self.opacity)
        self.uniforms["black_for_cycles"] = float(self.CONFIG["black_for_cycles"])
        self.uniforms["is_parameter_space"] = float(self.CONFIG["is_parameter_space"])

    def set_colors(self, colors):
        self.uniforms.update({
            f"color{n}": np.array(color_to_rgba(color))
            for n, color in enumerate(colors)
        })
        return self

    def set_julia_highlight(self, value):
        self.uniforms["julia_highlight"] = value

    def set_coefs(self, coefs, reset_roots=True):
        full_coefs = [*coefs] + [0] * (self.CONFIG["max_degree"] - len(coefs) + 1)
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
        full_roots = [*roots] + [0] * (self.CONFIG["max_degree"] - len(roots))
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

# Scenes

class NewtonFract(InteractiveScene):
    coefs = [1.0,0.0,0.0, -1.0]
    plane_config = {
        "x_range": (-10, 10),
        "y_range": (-10, 10),
        "height": 24,
        "width": 24,
        "background_line_style": {
            "stroke_color": GREY_A,
            "stroke_width": 1.0,
        },
        "axis_config": {
            "stroke_width": 1.0,
        }
    }
    n_steps = 100

    def construct(self):
        self.drag_to_pan = False
        self.init_fractal(root_colors=ROOT_COLORS_BRIGHT)
        fractal, plane, root_dots = self.group

        # Transition from last scene
        frame = self.camera.frame
        frame.shift(plane.n2p(2) - RIGHT_SIDE)

        self.play(
            frame.animate.center(),
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
        # zoom_points = [
        #     [-3.12334879, 1.61196545, 0.],
        #     [1.21514006, 0.01415811, 0.],
        # ]
        # for point in zoom_points:
        #     self.play(
        #         frame.animate.set_height(2e-3).move_to(point),
        #         run_time=10,
        #         rate_func=bezier(2 * [0] + 6 * [1])
        #     )
        #     self.wait()
        #     self.play(
        #         frame.animate.center().set_height(8),
        #         run_time=10,
        #         rate_func=bezier(6 * [0] + 2 * [1])
        #     )

        # Allow for play
        self.tie_fractal_to_root_dots(fractal)
        fractal.set_n_steps(100)

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
        self.disable_interaction(fractal,plane)
        self.add(fractal)
        self.add(plane)
        self.add(*root_dots)

    def get_plane(self):
        plane = ComplexPlane(**self.plane_config)
        plane.add_coordinate_labels(font_size=24)
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
            Dot(plane.n2p(root), color=color,fill_color=color)
            for root, color in zip(
                coefficients_to_roots(fractal.coefs),
                fractal.colors
            )
        ))
        self.root_dots.set_stroke(BLACK, 5, behind=True)
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

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        self.root_dots.clear_updaters()


class HalleyFract(InteractiveScene):
    coefs = [1.0, 0.0, 0.0, -1.0]
    plane_config = {
        "x_range": (-4, 4),
        "y_range": (-4, 4),
        "height": 16,
        "width": 16,
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
        self.drag_to_pan = False
        self.init_fractal(root_colors=ROOT_COLORS_BRIGHT)
        fractal, plane, root_dots = self.group

        # Transition from last scene
        frame = self.camera.frame
        frame.shift(plane.n2p(2) - RIGHT_SIDE)

        self.play(
            frame.animate.center(),
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
                run_time=10,
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
        self.disable_interaction(fractal,plane)
        self.add(fractal)
        self.add(plane)
        self.add(*root_dots)

    def get_plane(self):
        plane = ComplexPlane(**self.plane_config)
        plane.add_coordinate_labels(font_size=24)
        self.plane = plane
        return plane

    def get_fractal(self, plane, colors=ROOT_COLORS_DEEP, n_steps=30):
        return HalleyFractal(
            plane,
            colors=colors,
            coefs=self.coefs,
            n_steps=n_steps,
        )

    def get_root_dots(self, plane, fractal):
        self.root_dots = VGroup(*(
            Dot(plane.n2p(root), color=color,fill_color=color)
            for root, color in zip(
                coefficients_to_roots(fractal.coefs),
                fractal.colors
            )
        ))
        self.root_dots.set_stroke(BLACK, 5, behind=True)
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

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        self.root_dots.clear_updaters()


class ChebysevFract(InteractiveScene):
    coefs = [3.0,5.0,12.0,-18.0]
    plane_config = {
        "x_range": (-6, 6),
        "y_range": (-6, 6),
        "height": 24,
        "width": 24,
        "background_line_style": {
            "stroke_color": GREY_A,
            "stroke_width": 1.0,
        },
        "axis_config": {
            "stroke_width": 1.0,
        }
    }
    n_steps = 100

    def construct(self):
        self.drag_to_pan = False
        self.init_fractal(root_colors=ROOT_COLORS_BRIGHT)
        fractal, plane, root_dots = self.group

        # Transition from last scene
        frame = self.camera.frame
        frame.shift(plane.n2p(2) - RIGHT_SIDE)

        self.play(
            frame.animate.center(),
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
        zoom_points = [
            [0.0,0.0, 0.]
        ]
        for point in zoom_points:
            self.play(
                frame.animate.set_height(2e-3).move_to(point),
                run_time=10,
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
        self.disable_interaction(fractal,plane)
        self.add(fractal)
        self.add(plane)
        self.add(*root_dots)

    def get_plane(self):
        plane = ComplexPlane(**self.plane_config)
        plane.add_coordinate_labels(font_size=24)
        self.plane = plane
        return plane

    def get_fractal(self, plane, colors=ROOT_COLORS_DEEP, n_steps=30):
        return ChebysevFractal(
            plane,
            colors=colors,
            coefs=self.coefs,
            n_steps=n_steps,
        )

    def get_root_dots(self, plane, fractal):
        self.root_dots = VGroup(*(
            Dot(plane.n2p(root), color=color,fill_color=color)
            for root, color in zip(
                coefficients_to_roots(fractal.coefs),
                fractal.colors
            )
        ))
        self.root_dots.set_stroke(BLACK, 5, behind=True)
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

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        self.root_dots.clear_updaters()
