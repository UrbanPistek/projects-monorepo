import math
from manim import *

class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))  # show the circle on screen

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.rotate(PI / 4)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation

class RotatingSurface(ThreeDScene):
    def construct(self):
        # Define the axes
        axes = ThreeDAxes()
        
        # Define the surface
        surface = Surface(
            lambda u, v: axes.c2p(u, v, u*v),
            u_range=(-2, 2),
            v_range=(-2, 2),
        )
        
        # Add the axes and the surface to the scene
        self.add(axes, surface)

        # Animate rotation around the y-axis
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(6)
        self.stop_ambient_camera_rotation()

