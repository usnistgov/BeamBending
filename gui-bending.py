#GUI FOR BENDING
#Benjamin Schreyer
#stephan.schlamminger@nist.gov
import sys
import numpy as np
import mpmath as mp
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QDoubleSpinBox, QProgressBar, QCheckBox, QSlider,
    QPushButton, QLabel
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import make_interp_spline
from functions_bending_schreyer_adaptive89 import bend_samples, integrate_xz

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
class DraggableSplineCanvas(FigureCanvas):
    def __init__(self):
        self.L = 1
        self.w = 2
        self.fig = Figure(figsize=(5, 3), dpi=100)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.set_dark_style()

        self.xs = np.linspace(0, 1, 10)
        self.ys = np.ones(10)
        self.drag_points, = self.ax.plot(self.xs, self.ys, 'o', color='cyan', picker=5)
        self.spline_line, = self.ax.plot([], [], '-', color='magenta')
        self._dragging_point = None

        self.update_spline()
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        

    def set_dark_style(self):
        self.ax.set_facecolor("black")
        self.fig.patch.set_facecolor('black')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 2)
        self.ax.set_title("Flexure profile half width editor", color='white')
        self.ax.set_xlabel("Distance along flexure (m)", color='white')
        self.ax.set_ylabel("Flexure profile half width (m)", color='white')
    def update_spline_axes(self):
        self.ax.set_xlim(0, self.L)
        self.ax.set_ylim(0,self.w)
        self.xs = np.linspace(0,self.L, 10)
        self.ys = np.ones(self.xs.shape) * 0.5 * ( self.w)
    def update_spline(self):

        spline_x = np.linspace(0, self.L, 200)
        spline_y = make_interp_spline(self.xs, self.ys, k=3, bc_type = "natural")(spline_x)

        self.spline_line.set_data(spline_x, spline_y)
        self.drag_points.set_data(self.xs, self.ys)
        self.draw()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        for i, (x, y) in enumerate(zip(self.xs, self.ys)):
            if abs(event.xdata - x) < 0.05 * self.L and abs(event.ydata - y) < 0.1 * self.w:
                self._dragging_point = i
                break

    def on_motion(self, event):
        if self._dragging_point is None or event.inaxes != self.ax:
            return
        index = self._dragging_point
        self.ys[index] = np.clip(event.ydata, 0, float(self.w))
        self.update_spline()

    def on_release(self, event):
        self._dragging_point = None

    def get_spline_data(self):
        return self.xs.copy(), self.ys.copy()


class SimulationPlot(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 3), dpi=100)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.set_dark_style()
        self.last_geometry = None  # Store previous geometry
        self.cbar = None

    def set_dark_style(self):
        self.ax.set_facecolor("black")
        self.fig.patch.set_facecolor("black")
        self.ax.tick_params(colors="white")
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.set_title("Flexure shape", color='white')
        self.ax.set_xlabel("x (m)", color='white')
        self.ax.set_ylabel("y (m)", color='white')

    def plot_geometry(self, s, theta, L, energy_density = None):
        x, z = integrate_xz(theta, s)
        x = -x
        z = -z
        print(x, z)

        self.ax.clear()
        self.set_dark_style()
        self.ax.set_xlim([0, L])
        self.ax.set_ylim([-L * 0.03, L * 0.2007])

        # Plot last geometry as ghost line
        if self.last_geometry is not None:
            last_z, last_x = self.last_geometry
            self.ax.plot(last_z, last_x, ':', color='red', label="Previous", zorder = 100)

        # Plot current geometry
        if energy_density is not None:
            points = np.array([z, x]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = Normalize(vmin=0, vmax=np.max(energy_density))
            lc = LineCollection(segments, cmap='cool', norm=norm)
            lc.set_array(energy_density)
            lc.set_linewidth(2)
            self.ax.add_collection(lc)
            # Add colorbar
            if self.cbar is None:
                self.cbar = self.fig.colorbar(lc, ax=self.ax, pad=0.02)
                self.cbar.set_label("Relative energy density", color='white')
                self.cbar.ax.yaxis.set_tick_params(color='white')
                self.cbar.outline.set_edgecolor('white')
                plt.setp(self.cbar.ax.yaxis.get_ticklabels(), color='white')
        else:
            self.ax.plot(z, x, '-', color='magenta', label="Current", zorder=0)

        self.last_geometry = (z, x)  # Save current geometry as last
        self.ax.legend(loc='upper right')
        for line in self.ax.lines:
            print("Line:", line.get_linestyle(), line.get_color())
        self.draw()


class Screen01(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive bending simulator")
        self.setGeometry(100, 100, 1400, 1000)
        self.setStyleSheet("background-color: black; color: white;")

        layout = QVBoxLayout(self)

        # -------- Input Form --------
        form_group = QGroupBox("Input Parameters")
        form_layout = QFormLayout()

        self.bend_angle = QDoubleSpinBox(suffix=" rad", minimum=0.01, maximum=100.0)
        self.mass = QDoubleSpinBox(suffix=" kg", minimum=1.0, maximum=1000.0)
        self.gravity = QDoubleSpinBox(suffix=" m/s²", minimum=1.0, maximum=100.0, value=9.81)
        self.youngs_modulus = QDoubleSpinBox(decimals=2, minimum = 10,maximum=1000, suffix = " GPa")
        self.length = QDoubleSpinBox(suffix=" mm", minimum=1.0, maximum=100.0)
        self.width = QDoubleSpinBox(suffix=" mm", minimum=1.0, maximum=100.0)
        self.thickness = QDoubleSpinBox(suffix=" um", minimum=1.0, maximum=100.0)


        form_layout.addRow("Final Bend Angle:", self.bend_angle)
        form_layout.addRow("Mass:", self.mass)
        form_layout.addRow("Gravity:", self.gravity)
        form_layout.addRow("Young's Modulus:", self.youngs_modulus)
        form_layout.addRow("Length:", self.length)
        form_layout.addRow("Width:", self.width)
        form_layout.addRow("Thickness:", self.thickness)
        self.fcos_checkbox = QCheckBox("Include Fcos (cos(θ))")
        self.fcos_checkbox.setChecked(True)  # default value to match current behavior
        form_layout.addRow("Sideforce (rather than initial moment):", self.fcos_checkbox)

        self.rkf89_checkbox = QCheckBox("Use higher order solver")
        self.rkf89_checkbox.setChecked(True)  # default value to match current behavior
        form_layout.addRow("RKF89?", self.rkf89_checkbox)


        form_group.setLayout(form_layout)

        # -------- Tolerance Slider --------
        self.slider_label = QLabel("Error Tolerance: 1e-3")
        self.slider_label.setStyleSheet("color: white")
        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setMinimum(1)
        self.tol_slider.setMaximum(10)
        self.tol_slider.setValue(3)
        self.tol_slider.valueChanged.connect(self.update_slider_label)

        # -------- Run Button --------
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)

        # -------- Plots --------
        self.spline_canvas = DraggableSplineCanvas()
        self.sim_canvas = SimulationPlot()

        def spline_plot_update():
            self.spline_canvas.L = self.length.value() / 10**3
            self.spline_canvas.w = self.width.value() / 10**3
            self.spline_canvas.update_spline_axes()
        self.length.valueChanged.connect(spline_plot_update)
        self.width.valueChanged.connect(spline_plot_update)
        # -------- Assemble Layout --------
        layout.addWidget(form_group)
        layout.addWidget(self.slider_label)
        layout.addWidget(self.tol_slider)
        layout.addWidget(self.spline_canvas)
        layout.addWidget(self.run_button)
        layout.addWidget(self.sim_canvas)

    def update_slider_label(self):
        val = self.tol_slider.value()
        tol_exp = -val
        self.slider_label.setText(f"Error Tolerance: 1e{tol_exp}")

    def run_simulation(self):
        xs, ys = self.spline_canvas.get_spline_data()

        bend_angle = self.bend_angle.value()
        mass = self.mass.value()
        gravity = self.gravity.value()
        E = self.youngs_modulus.value() * 10**9
        L = self.length.value() / 10**3
        T = self.thickness.value() / 10**6

        tol = 10**(-self.tol_slider.value())

        
        s_eval = mp.matrix(np.linspace(0,L,int(200),endpoint = True))

        hs = ((make_interp_spline(xs, ys, bc_type = "natural" )))

        Fw = mass * gravity


        S, F, Es = bend_samples(
            grid=s_eval,
            hspline = hs,
            E=E,
            Fsin=mp.mpf(Fw),
            Fcos=self.fcos_checkbox.isChecked(),
            theta0=bend_angle,
            tol=mp.mpf(tol),
            T = T,
            use89 = self.rkf89_checkbox.isChecked()
        )
        theta = [-1*float(f[1]) for f in F]

        Ms = np.array([float(f[0]) for f in F])
        ws = np.array([float(hs(float(s))) for s in S])
        Is = (1/12) * (2*ws)**3 * T
        energy_density = Ms**2 / Is
        normalized_energy = energy_density / np.max(energy_density)

        self.sim_canvas.plot_geometry(S, theta, L, energy_density)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Screen01()
    window.show()
    sys.exit(app.exec())