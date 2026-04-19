import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QFileDialog, QLabel, QSplitter,
    QGroupBox, QTextEdit, QMessageBox, QSlider, QSpinBox, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from processing.loader    import BrukerLoader
from processing.processor import extract_trajectory, fit_t1
from utils.export         import export_results


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMR T1 Processor")
        self.setGeometry(100, 100, 1200, 750)

        self._loader   = None
        self._data     = None
        self._delays   = None
        self._n_real   = 0
        self._ppm      = None
        self._x_axis   = None
        self._x_label  = "Point index"

        self._int_lo   = None
        self._int_hi   = None

        self._span_selector = None
        self._row_updating  = False
        self._win_updating  = False

        # Stored after each successful fit so Export can access them
        self._last_fit     = None
        self._last_traj    = None
        self._last_delays  = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # ---- Left panel ----
        left_panel = QWidget()
        left_panel.setFixedWidth(290)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_layout.setSpacing(6)

        # --- Load Data ---
        load_group = QGroupBox("Load Data")
        load_vbox  = QVBoxLayout(load_group)
        load_vbox.setSpacing(4)
        self.btn_load_exp = QPushButton("Open Experiment Folder...")
        self.btn_load_exp.setToolTip(
            "Point directly at a numbered Bruker experiment folder\n"
            "e.g. .../1282026 T1 .../3"
        )
        self.btn_load_exp.clicked.connect(self.open_experiment_folder)
        load_vbox.addWidget(self.btn_load_exp)
        self.btn_load_dataset = QPushButton("Open Dataset Root (auto-detect)")
        self.btn_load_dataset.setToolTip(
            "Point at the dataset root folder.\n"
            "The program will find the T1 experiment inside automatically."
        )
        self.btn_load_dataset.clicked.connect(self.open_dataset_root)
        load_vbox.addWidget(self.btn_load_dataset)
        left_layout.addWidget(load_group)

        # --- Experiment Info (compact, read-only) ---
        meta_group = QGroupBox("Experiment Info")
        meta_vbox  = QVBoxLayout(meta_group)
        meta_vbox.setContentsMargins(4, 4, 4, 4)
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setFixedHeight(148)   # fits ~9 lines, no scroll needed
        self.meta_text.setStyleSheet("font-size: 11px;")
        meta_vbox.addWidget(self.meta_text)
        left_layout.addWidget(meta_group)

        # --- Spectrum Row Viewer ---
        row_group = QGroupBox("Spectrum Row Viewer")
        row_vbox  = QVBoxLayout(row_group)
        row_vbox.setSpacing(4)

        row_top = QHBoxLayout()
        row_top.addWidget(QLabel("Row:"))
        self.row_spinbox = QSpinBox()
        self.row_spinbox.setMinimum(0)
        self.row_spinbox.setMaximum(0)
        self.row_spinbox.setEnabled(False)
        self.row_spinbox.setFixedWidth(55)
        self.row_spinbox.setToolTip("Type a row number or use arrows")
        row_top.addWidget(self.row_spinbox)
        row_top.addStretch()
        self.delay_label = QLabel("tau = --")
        row_top.addWidget(self.delay_label)
        row_vbox.addLayout(row_top)

        self.row_slider = QSlider(Qt.Orientation.Horizontal)
        self.row_slider.setMinimum(0)
        self.row_slider.setMaximum(0)
        self.row_slider.setEnabled(False)
        self.row_slider.setToolTip("Drag to scrub through spectra")
        row_vbox.addWidget(self.row_slider)

        self.row_slider.valueChanged.connect(self._on_row_slider_changed)
        self.row_spinbox.valueChanged.connect(self._on_row_spinbox_changed)
        left_layout.addWidget(row_group)

        # --- Integration Window ---
        # Contains: drag hint, Left/Right exact inputs, width info, Fit button
        # The fit button lives here so it's always visible next to the controls
        # that feed it.
        int_group = QGroupBox("Integration Window")
        int_vbox  = QVBoxLayout(int_group)
        int_vbox.setSpacing(4)

        int_vbox.addWidget(QLabel("Drag on spectrum, or type exact values:"))

        left_row = QHBoxLayout()
        left_row.addWidget(QLabel("Left :"))
        self.win_left = QDoubleSpinBox()
        self.win_left.setDecimals(4)
        self.win_left.setSingleStep(0.001)
        self.win_left.setRange(-10000.0, 10000.0)
        self.win_left.setEnabled(False)
        self.win_left.setToolTip("Left edge of integration window")
        left_row.addWidget(self.win_left)
        self.win_left_unit = QLabel("ppm")
        left_row.addWidget(self.win_left_unit)
        int_vbox.addLayout(left_row)

        right_row = QHBoxLayout()
        right_row.addWidget(QLabel("Right:"))
        self.win_right = QDoubleSpinBox()
        self.win_right.setDecimals(4)
        self.win_right.setSingleStep(0.001)
        self.win_right.setRange(-10000.0, 10000.0)
        self.win_right.setEnabled(False)
        self.win_right.setToolTip("Right edge of integration window")
        right_row.addWidget(self.win_right)
        self.win_right_unit = QLabel("ppm")
        right_row.addWidget(self.win_right_unit)
        int_vbox.addLayout(right_row)

        self.win_width_label = QLabel("Width: --")
        self.win_width_label.setStyleSheet("color: gray; font-size: 11px;")
        int_vbox.addWidget(self.win_width_label)

        self.win_left.editingFinished.connect(self._on_win_left_changed)
        self.win_right.editingFinished.connect(self._on_win_right_changed)

        # Fit button lives inside the Integration Window group —
        # it is the action that consumes the window, so it belongs here.
        btn_row = QHBoxLayout()
        self.btn_fit = QPushButton("Fit T1 Curve")
        self.btn_fit.setEnabled(False)
        self.btn_fit.clicked.connect(self.run_fit)
        btn_row.addWidget(self.btn_fit)
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip("Save fit results and raw data to a CSV file")
        self.btn_export.clicked.connect(self.export_results)
        btn_row.addWidget(self.btn_export)
        int_vbox.addLayout(btn_row)

        left_layout.addWidget(int_group)

        # --- Results (fixed-height, always visible) ---
        # Only the most important numbers go here: T1, R², zero-crossing,
        # polarity flag, and any data-quality warnings.
        results_group = QGroupBox("Fit Results")
        results_vbox  = QVBoxLayout(results_group)
        results_vbox.setContentsMargins(4, 4, 4, 4)
        self.fit_text = QTextEdit()
        self.fit_text.setReadOnly(True)
        self.fit_text.setFixedHeight(160)
        self.fit_text.setStyleSheet("font-size: 11px;")
        self.fit_text.setPlaceholderText(
            "T1, R\u00b2, and fit quality will appear here\n"
            "after clicking 'Fit T1 Curve'."
        )
        results_vbox.addWidget(self.fit_text)
        left_layout.addWidget(results_group)

        root_layout.addWidget(left_panel)

        # ---- Right panel: spectrum (top) + recovery/residuals (bottom) ----
        plot_splitter = QSplitter(Qt.Orientation.Vertical)

        self.fig_spec    = Figure(figsize=(7, 3), tight_layout=True)
        self.canvas_spec = FigureCanvas(self.fig_spec)
        self.ax_spec     = self.fig_spec.add_subplot(111)
        self.ax_spec.set_title("Spectrum (no data loaded)")
        plot_splitter.addWidget(self.canvas_spec)

        self.fig_traj    = Figure(figsize=(7, 4), tight_layout=True)
        self.canvas_traj = FigureCanvas(self.fig_traj)
        self.ax_traj     = self.fig_traj.add_subplot(211)
        self.ax_resid    = self.fig_traj.add_subplot(212)
        self.ax_traj.set_title("T1 Recovery Trajectory (no data loaded)")
        self.ax_resid.set_title("Residuals")
        plot_splitter.addWidget(self.canvas_traj)

        root_layout.addWidget(plot_splitter)

    # ------------------------------------------------------------------
    # Folder selection
    # ------------------------------------------------------------------

    def open_experiment_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Bruker Experiment Folder (e.g. .../3)"
        )
        if folder:
            self._load(folder, auto_find=False)

    def open_dataset_root(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Bruker Dataset Root Folder"
        )
        if folder:
            self._load(folder, auto_find=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load(self, path, auto_find):
        self.meta_text.clear()
        self.fit_text.clear()
        self.btn_fit.setEnabled(False)
        self._int_lo = self._int_hi = None

        try:
            loader = BrukerLoader(path, auto_find=auto_find)
        except FileNotFoundError as e:
            self._show_error("Could not find T1 experiment", str(e))
            return

        try:
            meta = loader.get_metadata()
        except Exception as e:
            self._show_error("Could not read experiment parameters", str(e))
            return

        self._display_metadata(meta)

        try:
            dic, data = loader.load_processed_data()
        except FileNotFoundError as e:
            reply = QMessageBox.question(
                self, "Processed data not found",
                str(e) + "\n\nLoad raw (unprocessed) FID instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    dic, data = loader.load_raw_data()
                except FileNotFoundError as e2:
                    self._show_error("Could not load raw data", str(e2))
                    return
            else:
                return
        except Exception as e:
            self._show_error("Error loading data", str(e))
            return

        try:
            delays = loader.get_delays()
        except FileNotFoundError:
            n_rows = data.shape[0] if data.ndim >= 1 else 1
            delays = np.arange(n_rows, dtype=float)
            QMessageBox.warning(
                self, "vdlist not found",
                "Could not find the variable delay list.\n"
                "Using row indices as placeholder delay values.\n"
                "T1 fitting results will not have physical units.",
            )

        ppm = loader.get_ppm_axis()

        self._loader = loader
        self._data   = data
        self._delays = delays
        self._n_real = len(delays)
        self._ppm    = ppm

        if ppm is not None:
            self._x_axis  = ppm
            self._x_label = "Chemical Shift (ppm)"
        else:
            n_pts = data.shape[1] if data.ndim == 2 else data.shape[0]
            self._x_axis  = np.arange(n_pts, dtype=float)
            self._x_label = "Point index"

        # Configure row controls (capped at n_real-1)
        if data.ndim == 2:
            last_real = self._n_real - 1
            self._row_updating = True
            self.row_slider.setMaximum(last_real)
            self.row_slider.setValue(last_real)
            self.row_spinbox.setMaximum(last_real)
            self.row_spinbox.setValue(last_real)
            self.row_slider.setEnabled(True)
            self.row_spinbox.setEnabled(True)
            self._row_updating = False
            self._update_delay_label(last_real)
        else:
            self.row_slider.setEnabled(False)
            self.row_spinbox.setEnabled(False)

        # Configure window spinboxes
        unit_str = "ppm" if ppm is not None else "pts"
        self.win_left.setEnabled(True)
        self.win_right.setEnabled(True)
        self.win_left_unit.setText(unit_str)
        self.win_right_unit.setText(unit_str)
        if ppm is not None and len(ppm) > 1:
            step = abs(float(ppm[0] - ppm[1]))
            self.win_left.setSingleStep(round(step, 5))
            self.win_right.setSingleStep(round(step, 5))

        self._auto_set_integration()
        self._plot_spectrum(self.row_spinbox.value())
        self._plot_trajectory()
        self.btn_fit.setEnabled(True)

    # ------------------------------------------------------------------
    # Auto-detect integration window
    # ------------------------------------------------------------------

    def _auto_set_integration(self):
        data   = self._data
        x_axis = self._x_axis
        n_real = self._n_real

        if data is None or data.ndim != 2 or n_real == 0:
            return

        reference = data[n_real - 1].astype(float)
        peak_idx  = int(np.argmax(np.abs(reference)))
        n_pts     = len(reference)
        padding   = 30

        lo_idx = max(0, peak_idx - padding)
        hi_idx = min(n_pts - 1, peak_idx + padding)

        self._int_lo = float(x_axis[lo_idx])
        self._int_hi = float(x_axis[hi_idx])
        self._push_window_to_spinboxes()
        self._update_int_label()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_spectrum(self, row_idx):
        data   = self._data
        x_axis = self._x_axis

        if data is None:
            return

        spectrum = data[row_idx].astype(float) if data.ndim == 2 else data.astype(float)

        self.ax_spec.clear()
        self.ax_spec.plot(x_axis, spectrum, linewidth=0.8, color='steelblue')

        if self._int_lo is not None and self._int_hi is not None:
            lo = min(self._int_lo, self._int_hi)
            hi = max(self._int_lo, self._int_hi)
            self.ax_spec.axvspan(
                lo, hi, alpha=0.25, color='orange',
                label="Window: {:.4f} - {:.4f}".format(lo, hi)
            )
            self.ax_spec.legend(fontsize=8, loc='upper right')

        delay_val = (
            self._delays[row_idx]
            if (self._delays is not None and row_idx < self._n_real)
            else None
        )
        title = (
            "Spectrum  Row {}   (tau = {:.4f} s)".format(row_idx, delay_val)
            if delay_val is not None
            else "Spectrum  Row {}   (zero-padded)".format(row_idx)
        )
        self.ax_spec.set_title(title)
        self.ax_spec.set_xlabel(self._x_label)
        self.ax_spec.set_ylabel("Intensity")

        if self._ppm is not None:
            self.ax_spec.invert_xaxis()

        self.canvas_spec.draw()

        self._span_selector = SpanSelector(
            self.ax_spec,
            self._on_span_selected,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="orange"),
            interactive=True,
            drag_from_anywhere=True,
        )

    def _plot_trajectory(self):
        data   = self._data
        delays = self._delays
        n_real = self._n_real

        if data is None or data.ndim != 2 or n_real == 0:
            return

        x0, x1 = self._window_to_indices()
        traj    = extract_trajectory(data, x0, x1)[:n_real]
        d_plot  = delays[:n_real]

        self.ax_traj.clear()
        self.ax_traj.plot(d_plot, traj, 'ro', markersize=6, label="Integrated intensity")
        self.ax_traj.set_xlabel("Delay time (s)")
        self.ax_traj.set_ylabel("Intensity (a.u.)")

        lo, hi = self._int_lo, self._int_hi
        if self._ppm is not None and lo is not None:
            left_ppm  = max(lo, hi)
            right_ppm = min(lo, hi)
            title_window = "{:.4f} to {:.4f} ppm".format(left_ppm, right_ppm)
        elif lo is not None:
            title_window = "cols {} to {}".format(x0, x1)
        else:
            title_window = "no window set"

        self.ax_traj.set_title("T1 Recovery  (window: {})".format(title_window))
        self.ax_traj.legend()

        self.ax_resid.clear()
        self.ax_resid.set_title("Residuals (run Fit T1 Curve to populate)")
        self.ax_resid.set_xlabel("Delay time (s)")
        self.fig_traj.tight_layout()
        self.canvas_traj.draw()

    # ------------------------------------------------------------------
    # Interaction callbacks
    # ------------------------------------------------------------------

    def _on_row_slider_changed(self, value):
        if self._row_updating:
            return
        self._row_updating = True
        self.row_spinbox.setValue(value)
        self._row_updating = False
        self._update_delay_label(value)
        if self._data is not None:
            self._plot_spectrum(value)

    def _on_row_spinbox_changed(self, value):
        if self._row_updating:
            return
        self._row_updating = True
        self.row_slider.setValue(value)
        self._row_updating = False
        self._update_delay_label(value)
        if self._data is not None:
            self._plot_spectrum(value)

    def _update_delay_label(self, row_idx):
        if self._delays is not None and row_idx < self._n_real:
            self.delay_label.setText("tau = {:.4f} s".format(self._delays[row_idx]))
        elif row_idx >= self._n_real:
            self.delay_label.setText("zero-padded row")
        else:
            self.delay_label.setText("tau = --")

    def _on_span_selected(self, lo, hi):
        if abs(hi - lo) < 1e-9:
            return
        self._int_lo = lo
        self._int_hi = hi
        self._push_window_to_spinboxes()
        self._update_int_label()
        self._plot_spectrum(self.row_spinbox.value())
        self._plot_trajectory()

    def _on_win_left_changed(self):
        if self._win_updating or self._x_axis is None:
            return
        val = self.win_left.value()
        if self._ppm is not None:
            self._int_lo = val
            if self._int_hi is None:
                self._int_hi = val - 0.1
        else:
            self._int_lo = val
            if self._int_hi is None:
                self._int_hi = val + 100
        self._update_int_label()
        self._plot_spectrum(self.row_spinbox.value())
        self._plot_trajectory()

    def _on_win_right_changed(self):
        if self._win_updating or self._x_axis is None:
            return
        val = self.win_right.value()
        if self._ppm is not None:
            self._int_hi = val
            if self._int_lo is None:
                self._int_lo = val + 0.1
        else:
            self._int_hi = val
            if self._int_lo is None:
                self._int_lo = max(0, val - 100)
        self._update_int_label()
        self._plot_spectrum(self.row_spinbox.value())
        self._plot_trajectory()

    # ------------------------------------------------------------------
    # T1 Fitting
    # ------------------------------------------------------------------

    def run_fit(self):
        if self._data is None or self._delays is None:
            return

        x0, x1 = self._window_to_indices()
        traj    = extract_trajectory(self._data, x0, x1)[:self._n_real]
        delays  = self._delays[:self._n_real]

        result  = fit_t1(delays, traj)

        self.ax_traj.clear()
        self.ax_resid.clear()
        self.ax_traj.plot(delays, traj, 'ro', markersize=7, zorder=5, label="Data")

        if result["error"] is None:
            A, T1, C   = result["amplitude"], result["t1"], result["offset"]
            polarity   = result["polarity"]
            pole_label = "standard" if polarity == 1 else "phase-flipped (auto-corrected)"

            t_max        = min(max(delays[-1], 5.0 * T1), delays[-1] * 10)
            t_dense      = np.linspace(0, t_max, 1000)
            fitted_dense = polarity * A * (1.0 - 2.0 * np.exp(-t_dense / T1)) + C

            self.ax_traj.plot(
                t_dense, fitted_dense, 'b-', linewidth=2,
                label="Fit:  T1 = {:.4f} s   (R\u00b2 = {:.5f})".format(T1, result['r_squared'])
            )

            t_zero = T1 * np.log(2)
            self.ax_traj.axvline(
                t_zero, color='gray', linestyle='--', linewidth=1,
                label="t₀ = {:.4f} s".format(t_zero)
            )

            equil = polarity * A + C
            self.ax_traj.axhline(
                equil, color='green', linestyle=':', linewidth=1,
                label="Equilibrium = {:.0f} a.u.".format(equil)
            )

            if t_max > delays[-1]:
                self.ax_traj.axvspan(
                    delays[-1], t_max, alpha=0.06, color='blue',
                    label="Extrapolated region"
                )

            self.ax_resid.stem(
                delays, result["residuals"],
                linefmt='r-', markerfmt='ro', basefmt='k-'
            )
            self.ax_resid.axhline(0, color='black', linewidth=0.8)
            self.ax_resid.set_xlabel("Delay time (s)")
            self.ax_resid.set_ylabel("Residual (a.u.)")
            self.ax_resid.set_title(
                "Residuals  (random scatter = good fit | pattern = poor fit)"
            )

            # --- Results box: only the critical numbers ---
            lines = [
                "T1     = {:.4f} s".format(T1),
                "R\u00b2     = {:.5f}".format(result['r_squared']),
                "t₀     = {:.4f} s  (T1 × ln2)".format(t_zero),
                "A      = {:.1f} a.u.".format(A),
                "C      = {:.1f} a.u.".format(result['offset']),
                "Data   : {}".format(pole_label),
            ]
            if result["warnings"]:
                lines.append("")
                lines.extend(result["warnings"])
            self.fit_text.setPlainText("\n".join(lines))
            # Store for export
            self._last_fit    = result
            self._last_traj   = traj
            self._last_delays = delays
            self.btn_export.setEnabled(True)

        else:
            self.fit_text.setPlainText("Fit failed:\n\n{}".format(result['error']))
            self._last_fit = result
            self.ax_resid.set_title("Residuals (fit failed)")

        self.ax_traj.set_xlabel("Delay time (s)")
        self.ax_traj.set_ylabel("Intensity (a.u.)")
        self.ax_traj.set_title("T1 Recovery - Fitted")
        self.ax_traj.legend(fontsize=7)
        self.fig_traj.tight_layout()
        self.canvas_traj.draw()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(self):
        if self._last_fit is None:
            return

        import os
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        if not os.path.isdir(desktop):
            desktop = os.path.expanduser("~")
        default_path = os.path.join(desktop, "t1_results.csv")

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Results", default_path,
            "CSV files (*.csv);;All files (*)"
        )
        if not filepath:
            return

        unit = "ppm" if self._ppm is not None else "pts"
        lo   = self._int_lo if self._int_lo is not None else 0
        hi   = self._int_hi if self._int_hi is not None else 0
        window = {
            "left":  max(lo, hi) if self._ppm is not None else min(lo, hi),
            "right": min(lo, hi) if self._ppm is not None else max(lo, hi),
            "width": abs(hi - lo),
            "unit":  unit,
        }

        try:
            export_results(
                filepath,
                self._last_fit,
                self._last_delays,
                self._last_traj,
                window,
                self._loader.get_experiment_path() if self._loader else "unknown",
            )
            QMessageBox.information(
                self, "Exported",
                "Results saved to:\n{}".format(filepath)
            )
        except Exception as e:
            self._show_error("Export failed", str(e))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _push_window_to_spinboxes(self):
        if self._int_lo is None or self._int_hi is None:
            return
        self._win_updating = True
        if self._ppm is not None:
            self.win_left.setValue(max(self._int_lo, self._int_hi))
            self.win_right.setValue(min(self._int_lo, self._int_hi))
        else:
            self.win_left.setValue(min(self._int_lo, self._int_hi))
            self.win_right.setValue(max(self._int_lo, self._int_hi))
        self._win_updating = False

    def _update_int_label(self):
        if self._int_lo is None:
            self.win_width_label.setText("Width: --")
            return
        unit  = "ppm" if self._ppm is not None else "pts"
        width = abs(self._int_hi - self._int_lo)
        self.win_width_label.setText("Width: {:.4f} {}".format(width, unit))
        self._push_window_to_spinboxes()

    def _window_to_indices(self):
        x_axis = self._x_axis
        data   = self._data

        if x_axis is None or data is None:
            n = data.shape[1] if (data is not None and data.ndim == 2) else 1
            return 0, n

        if self._int_lo is None or self._int_hi is None:
            ref      = data[self._n_real - 1].astype(float) if data.ndim == 2 else data.astype(float)
            peak_idx = int(np.argmax(np.abs(ref)))
            n_pts    = len(ref)
            return max(0, peak_idx - 30), min(n_pts, peak_idx + 30)

        lo = min(self._int_lo, self._int_hi)
        hi = max(self._int_lo, self._int_hi)

        if x_axis[0] > x_axis[-1]:
            i_hi = int(np.searchsorted(-x_axis, -hi, side='left'))
            i_lo = int(np.searchsorted(-x_axis, -lo, side='right'))
            x0, x1 = min(i_hi, i_lo), max(i_hi, i_lo)
        else:
            x0 = int(np.searchsorted(x_axis, lo, side='left'))
            x1 = int(np.searchsorted(x_axis, hi, side='right'))

        n_pts = x_axis.shape[0]
        return max(0, x0), min(n_pts, x1)

    def _display_metadata(self, meta):
        lines = [
            "Pulse program : {}".format(meta.get('pulprog', '?')),
            "Nucleus       : {}".format(meta.get('nuc1', '?')),
            "SFO1          : {} MHz".format(meta.get('sfo1_mhz', '?')),
            "SW            : {} Hz".format(meta.get('sw_hz', '?')),
            "NS (scans)    : {}".format(meta.get('ns', '?')),
        ]
        if meta.get("has_vdlist") and "vdlist" in meta:
            d = meta["vdlist"]
            lines += [
                "N delays      : {}".format(len(d)),
                "Delay range   : {:.4f} - {:.4f} s".format(d[0], d[-1]),
            ]
        lines += [
            "Has processed : {}".format('Yes' if meta.get('has_2rr') else 'No'),
        ]
        self.meta_text.setPlainText("\n".join(lines))

    def _show_error(self, title, message):
        QMessageBox.critical(self, title, message)