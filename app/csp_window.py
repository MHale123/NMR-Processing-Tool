"""
CSP Binding Analysis Window
===========================
Layout mirrors the T1 module: controls on the left, waterfall fills the
entire right panel.  The Δδ / Kd analysis plots open in a separate window
(CSPResultsWindow) only when the user clicks "Run CSP Analysis".

Workflow
--------
1. Load spectra — one at a time, or batch-load all numbered subfolders.
2. Full spectrum shown immediately.  All detected peaks are marked with
   dotted vertical lines.
3. Click a peak (snaps to nearest detected) or drag to select a region.
   View zooms in automatically on click.
4. Smooth slider adjusts display trace in real time.
5. "Export Waterfall…" saves the current view as PNG / PDF / SVG.
6. "Run CSP Analysis" opens a separate results window with Δδ and Kd plots.
"""

import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QFileDialog, QLabel, QSplitter,
    QGroupBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QDoubleSpinBox, QComboBox, QCheckBox,
    QAbstractItemView, QSlider, QSizePolicy, QSpacerItem,
    QDialog, QDialogButtonBox, QComboBox as QCombo,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import SpanSelector

from processing.loader import load_1d_spectrum
from processing.csp_processor import (
    extract_peak_centres, compute_delta_delta, fit_kd
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _savgol(spec, window):
    if window < 3:
        return spec
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(spec, window_length=window,
                             polyorder=min(3, window - 1))
    except Exception:
        return spec


def _phase_correct(spec):
    return -spec if np.max(spec) < -np.min(spec) else spec.copy()


def _iterative_poly_baseline(ppm, spec, degree=5, n_iter=10, threshold=2.0):
    """
    Iterative polynomial baseline correction for NMR spectra.

    Algorithm (standard NMR practice, similar to MestReNova/NMRPipe POLY):
    1. Fit a polynomial of `degree` through ALL points.
    2. Mark any points that lie MORE than `threshold` × RMS above the
       current fit as peaks — exclude them.
    3. Refit using only the remaining (non-peak) points.
    4. Repeat for `n_iter` iterations or until the peak mask stabilises.
    5. Return spectrum − final polynomial.

    This works on noisy spectra because:
    - Iteration progressively eliminates real peaks from the fit.
    - Points at the noise floor (neither peaks nor baseline artifacts)
      constrain the polynomial from both above and below.
    - A degree-5 polynomial is flexible enough to follow broad probe
      background / field-gradient baseline but stiff enough not
      to fit into peaks.
    """
    x    = ppm.copy()
    y    = spec.copy()
    mask = np.ones(len(y), dtype=bool)   # True = include in fit

    for _ in range(n_iter):
        if mask.sum() < degree + 2:
            break
        coeffs   = np.polyfit(x[mask], y[mask], degree)
        baseline = np.polyval(coeffs, x)
        residual = y - baseline
        rms      = np.sqrt(np.mean(residual[mask] ** 2))
        # Exclude points significantly above the baseline (peaks)
        new_mask = residual < threshold * rms
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    coeffs   = np.polyfit(x[mask], y[mask], degree)
    baseline = np.polyval(coeffs, x)
    return spec - baseline


def _minmax_decimate(ppm, spec, target=6000):
    """
    Decimate to ~target display points using min-max per block.
    Higher target = thinner noise appearance, sharper peaks preserved.
    """
    n = len(spec)
    if n <= target:
        return ppm, spec
    block = max(1, n // (target // 2))
    n_blocks = n // block
    pd, sd = [], []
    for b in range(n_blocks):
        sl = slice(b * block, (b + 1) * block)
        pb, sb = ppm[sl], spec[sl]
        imin, imax = int(np.argmin(sb)), int(np.argmax(sb))
        if imin <= imax:
            pd += [pb[imin], pb[imax]]; sd += [sb[imin], sb[imax]]
        else:
            pd += [pb[imax], pb[imin]]; sd += [sb[imax], sb[imin]]
    return np.array(pd), np.array(sd)


def _find_peaks(ppm, spec, sensitivity=5.0):
    """
    Detect genuine peaks in a prepared spectrum.
    Returns ppm positions sorted by prominence (strongest first).

    sensitivity : float
        Multiplier on the MAD noise floor used as the height and prominence
        threshold.  Lower = more sensitive (picks up weaker peaks, more false
        positives).  Higher = stricter (only strong, clear peaks).
        Typical range: 2 (very sensitive) – 10 (strict).
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return np.array([])
    n = len(spec)
    if n < 10:
        return np.array([])
    noise = np.median(np.abs(spec - np.median(spec))) * 1.4826
    if noise < 1e-12:
        return np.array([])
    idx, props = find_peaks(
        spec,
        height=noise * sensitivity,
        distance=max(5, n // 200),
        prominence=noise * max(1.5, sensitivity * 0.75),
    )
    if len(idx) == 0:
        return np.array([])
    return ppm[idx[np.argsort(props['prominences'])[::-1]]]


class ExportDialog(QDialog):
    """Small dialog to pick format and DPI before saving the waterfall."""

    FORMATS = [
        ("PNG  — raster image  (publication-ready)", "png"),
        ("PDF  — vector, scalable                  ", "pdf"),
        ("SVG  — vector, editable in Illustrator   ", "svg"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Waterfall")
        self.setFixedSize(360, 160)
        layout = QVBoxLayout(self)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self.fmt_combo = QCombo()
        for label, _ in self.FORMATS:
            self.fmt_combo.addItem(label)
        fmt_row.addWidget(self.fmt_combo)
        layout.addLayout(fmt_row)

        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("Resolution (PNG only):"))
        self.dpi_combo = QCombo()
        for dpi in ["150 dpi  (screen)", "300 dpi  (print)", "600 dpi  (high-res)"]:
            self.dpi_combo.addItem(dpi)
        self.dpi_combo.setCurrentIndex(1)   # 300 dpi default
        dpi_row.addWidget(self.dpi_combo)
        layout.addLayout(dpi_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_format(self):
        return self.FORMATS[self.fmt_combo.currentIndex()][1]

    def get_dpi(self):
        return [150, 300, 600][self.dpi_combo.currentIndex()]


# ---------------------------------------------------------------------------
# Main CSP window
# ---------------------------------------------------------------------------

class CSPWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSP Binding Analysis")
        self.setGeometry(120, 120, 1350, 780)

        self._series      = []
        self._peak_ppm    = None
        self._auto_peaks  = []
        self._peak_cursor = -1

        self._view_lo = None
        self._view_hi = None

        # Cache: key = (id(raw_array), smooth_window) for display
        #             (id(raw_array), 'det')          for detection
        self._cache = {}

        self._span_sel  = None
        self._click_cid = None

        # Keep results window alive
        self._results_win = None

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ── Left panel ──────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(295)
        lv = QVBoxLayout(left)
        lv.setAlignment(Qt.AlignmentFlag.AlignTop)
        lv.setSpacing(6)

        # Load group
        load_grp = QGroupBox("Load Data")
        load_v   = QVBoxLayout(load_grp)
        load_v.setSpacing(4)
        b_add = QPushButton("Add Spectrum…")
        b_add.setToolTip("Select a single numbered Bruker experiment folder")
        b_add.clicked.connect(self.add_spectrum)
        load_v.addWidget(b_add)
        b_bat = QPushButton("Add Folder  (batch load subfolders)…")
        b_bat.setToolTip(
            "Select a parent folder — all numbered subfolders\n"
            "(1/, 2/, 3/ …) with processed 1D data are loaded at once.")
        b_bat.clicked.connect(self.add_folder_batch)
        load_v.addWidget(b_bat)
        lv.addWidget(load_grp)

        # Series table group
        series_grp = QGroupBox("Concentration Series")
        series_v   = QVBoxLayout(series_grp)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Label", "Conc", "Unit", "Ref"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col, w in [(1, 60), (2, 50), (3, 34)]:
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(col, w)
        self.table.setMaximumHeight(160)
        self.table.setMinimumHeight(28)
        self.table.setSizeAdjustPolicy(
            QTableWidget.SizeAdjustPolicy.AdjustToContents)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.table.itemChanged.connect(self._on_table_item_changed)
        series_v.addWidget(self.table)

        tbl_btns = QHBoxLayout()
        b_rem = QPushButton("Remove")
        b_rem.clicked.connect(self.remove_selected)
        tbl_btns.addWidget(b_rem)
        b_up = QPushButton("▲")
        b_up.setFixedWidth(30); b_up.clicked.connect(self.move_up)
        tbl_btns.addWidget(b_up)
        b_dn = QPushButton("▼")
        b_dn.setFixedWidth(30); b_dn.clicked.connect(self.move_down)
        tbl_btns.addWidget(b_dn)
        series_v.addLayout(tbl_btns)
        lv.addWidget(series_grp)

        # Peak selection group
        peak_grp = QGroupBox("Peak Selection")
        peak_v   = QVBoxLayout(peak_grp)
        peak_v.setSpacing(5)

        cycle_row = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(30)
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._prev_peak)
        cycle_row.addWidget(self.btn_prev)
        self.lbl_cycle = QLabel("Load spectra to detect peaks")
        self.lbl_cycle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_cycle.setStyleSheet("font-size: 10px; color: gray;")
        self.lbl_cycle.setWordWrap(True)
        cycle_row.addWidget(self.lbl_cycle)
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(30)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._next_peak)
        cycle_row.addWidget(self.btn_next)
        peak_v.addLayout(cycle_row)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Selected:"))
        self.spb_peak = QDoubleSpinBox()
        self.spb_peak.setDecimals(4); self.spb_peak.setRange(-10000, 10000)
        self.spb_peak.setEnabled(False); self.spb_peak.setFixedWidth(90)
        self.spb_peak.setToolTip(
            "Centre of the selected peak in ppm.\n"
            "Type a value directly if auto-detection missed it.")
        self.spb_peak.editingFinished.connect(self._on_peak_spinbox_changed)
        sel_row.addWidget(self.spb_peak)
        sel_row.addWidget(QLabel("ppm"))
        peak_v.addLayout(sel_row)

        # Sensitivity slider — controls peak detection threshold
        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Sensitivity:"))
        self.sld_sens = QSlider(Qt.Orientation.Horizontal)
        self.sld_sens.setMinimum(1)
        self.sld_sens.setMaximum(10)
        self.sld_sens.setValue(5)
        self.sld_sens.setTickInterval(1)
        self.sld_sens.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_sens.setToolTip(
            "Peak detection sensitivity.\n"
            "1 = most sensitive (picks up weak peaks, more false positives).\n"
            "10 = strictest (only strong, unambiguous peaks).\n"
            "If no peaks are detected at sensitivity 1, the signal is below\n"
            "the noise floor — click the spectrum or type the ppm directly.")
        self.sld_sens.valueChanged.connect(self._on_sensitivity_changed)
        sens_row.addWidget(self.sld_sens)
        self.lbl_sens = QLabel("5×")
        self.lbl_sens.setFixedWidth(26)
        sens_row.addWidget(self.lbl_sens)
        peak_v.addLayout(sens_row)

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Fit window ±"))
        self.spb_win = QDoubleSpinBox()
        self.spb_win.setDecimals(3); self.spb_win.setRange(0.01, 10.0)
        self.spb_win.setValue(0.15);  self.spb_win.setFixedWidth(68)
        self.spb_win.setToolTip("Gaussian fitting half-window (ppm)")
        win_row.addWidget(self.spb_win)
        win_row.addWidget(QLabel("ppm"))
        peak_v.addLayout(win_row)
        lv.addWidget(peak_grp)

        # View group
        view_grp = QGroupBox("View")
        view_v   = QVBoxLayout(view_grp)
        view_v.setSpacing(4)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom:"))
        self.spb_zlo = QDoubleSpinBox()
        self.spb_zlo.setDecimals(2); self.spb_zlo.setRange(-2000, 2000)
        self.spb_zlo.setValue(-60);  self.spb_zlo.setFixedWidth(68)
        zoom_row.addWidget(self.spb_zlo)
        zoom_row.addWidget(QLabel("to"))
        self.spb_zhi = QDoubleSpinBox()
        self.spb_zhi.setDecimals(2); self.spb_zhi.setRange(-2000, 2000)
        self.spb_zhi.setValue(-140); self.spb_zhi.setFixedWidth(68)
        zoom_row.addWidget(self.spb_zhi)
        zoom_row.addWidget(QLabel("ppm"))
        view_v.addLayout(zoom_row)

        zoom_btns = QHBoxLayout()
        b_z = QPushButton("Apply Zoom")
        b_z.clicked.connect(self._apply_zoom)
        zoom_btns.addWidget(b_z)
        b_r = QPushButton("Full View")
        b_r.clicked.connect(self._reset_zoom)
        zoom_btns.addWidget(b_r)
        view_v.addLayout(zoom_btns)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Smooth:"))
        self.sld_smooth = QSlider(Qt.Orientation.Horizontal)
        self.sld_smooth.setMinimum(0); self.sld_smooth.setMaximum(20)
        self.sld_smooth.setValue(0)
        self.sld_smooth.setTickInterval(5)
        self.sld_smooth.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_smooth.setToolTip(
            "Savitzky-Golay display smoothing (0 = off).\n"
            "Does not affect peak fitting.")
        self.sld_smooth.valueChanged.connect(self._on_smooth_changed)
        smooth_row.addWidget(self.sld_smooth)
        self.lbl_smooth = QLabel("Off")
        self.lbl_smooth.setFixedWidth(30)
        smooth_row.addWidget(self.lbl_smooth)
        view_v.addLayout(smooth_row)

        spacing_row = QHBoxLayout()
        spacing_row.addWidget(QLabel("Spacing:"))
        self.sld_spacing = QSlider(Qt.Orientation.Horizontal)
        self.sld_spacing.setMinimum(1); self.sld_spacing.setMaximum(10)
        self.sld_spacing.setValue(2)
        self.sld_spacing.setTickInterval(1)
        self.sld_spacing.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_spacing.setToolTip(
            "Vertical spacing between traces.\n"
            "All spectra share the same intensity scale — this just\n"
            "controls how far apart the rows are stacked.")
        self.sld_spacing.valueChanged.connect(lambda _: self._plot_stacked())
        spacing_row.addWidget(self.sld_spacing)
        self.lbl_spacing = QLabel("3")
        self.lbl_spacing.setFixedWidth(20)
        self.sld_spacing.valueChanged.connect(
            lambda v: self.lbl_spacing.setText(str(v)))
        spacing_row.addWidget(self.lbl_spacing)
        view_v.addLayout(spacing_row)

        self.chk_baseline = QCheckBox("Auto baseline correction")
        self.chk_baseline.setChecked(True)
        self.chk_baseline.setToolTip(
            "Iterative polynomial baseline correction (degree 5).\n"
            "Progressively excludes peaks from the fit so only the\n"
            "broad background is subtracted — works on noisy spectra\n"
            "and curved baselines without distorting peak positions.\n"
            "Cosmetic only; does not affect Gaussian peak fitting.")
        self.chk_baseline.stateChanged.connect(self._on_display_option_changed)
        view_v.addWidget(self.chk_baseline)

        self.chk_sort = QCheckBox("Sort waterfall by concentration")
        self.chk_sort.setChecked(True)
        self.chk_sort.stateChanged.connect(lambda _: self._plot_stacked())
        view_v.addWidget(self.chk_sort)

        lv.addWidget(view_grp)

        # Actions group
        act_grp = QGroupBox("Actions")
        act_v   = QVBoxLayout(act_grp)
        act_v.setSpacing(4)

        self.btn_export = QPushButton("Export Waterfall…")
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip(
            "Save the current waterfall view as PNG, PDF, or SVG")
        self.btn_export.clicked.connect(self.export_waterfall)
        act_v.addWidget(self.btn_export)

        self.btn_run = QPushButton("Run CSP Analysis")
        self.btn_run.setEnabled(False)
        self.btn_run.setToolTip(
            "Fit Gaussian peak centres across all spectra and show\n"
            "Δδ vs concentration and Kd determination plots.")
        self.btn_run.clicked.connect(self.run_analysis)
        act_v.addWidget(self.btn_run)

        lv.addWidget(act_grp)
        lv.addItem(QSpacerItem(
            0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        root.addWidget(left)

        # ── Right panel — waterfall only ─────────────────────────────────
        self.fig  = Figure(figsize=(9, 6))
        self.cvs  = FigureCanvas(self.fig)
        self.ax   = self.fig.add_subplot(111)
        self._empty_plot()
        root.addWidget(self.cvs)

    def _empty_plot(self):
        self.ax.clear()
        self.ax.set_title("Load spectra to begin", fontsize=11)
        self.ax.set_xlabel("Chemical Shift (ppm)")
        self.ax.set_ylabel("Concentration / ID")
        self.ax.text(0.5, 0.5,
            'Use "Add Spectrum…" or "Add Folder…" to load Bruker 1D data',
            transform=self.ax.transAxes,
            ha='center', va='center', color='gray', fontsize=10)
        self.cvs.draw()

    # -----------------------------------------------------------------------
    # Spectrum preparation
    # -----------------------------------------------------------------------

    def _smooth_win(self):
        v = self.sld_smooth.value()
        return 0 if v == 0 else v * 2 + 3

    def _prepare_display(self, data):
        """
        Returns (ppm, spec) ready for display.
        Pipeline: phase-correct → optional iterative poly baseline → user smoothing.
        Cached by (array id, smooth_window, baseline_on, phase_offset).
        """
        key = (id(data['spectrum']), self._smooth_win(),
               self.chk_baseline.isChecked())
        if key not in self._cache:
            ppm  = data['ppm']
            spec = _phase_correct(data['spectrum'].astype(float))
            if self.chk_baseline.isChecked():
                spec = _iterative_poly_baseline(ppm, spec)
            spec = _savgol(spec, self._smooth_win())
            self._cache[key] = (ppm, spec)
        return self._cache[key]

    def _prepare_detect(self, data):
        """
        Returns (ppm, spec) for peak detection.
        Pipeline: phase-correct → iterative poly baseline → fixed smoothing (w=11).
        Always uses baseline correction for reliable detection.
        Cached separately from display cache.
        """
        key = (id(data['spectrum']), 'det')
        if key not in self._cache:
            ppm  = data['ppm']
            spec = _phase_correct(data['spectrum'].astype(float))
            spec = _iterative_poly_baseline(ppm, spec)
            spec = _savgol(spec, 11)
            self._cache[key] = (ppm, spec)
        return self._cache[key]

    # -----------------------------------------------------------------------
    # Series management
    # -----------------------------------------------------------------------

    def _load_entry(self, folder):
        try:
            data = load_1d_spectrum(folder)
        except (FileNotFoundError, ValueError) as e:
            QMessageBox.critical(self, "Load failed",
                                 f"{os.path.basename(folder)}:\n{e}")
            return None
        return {'data': data, 'conc': 0.0, 'unit': 'uM',
                'label': os.path.basename(folder), 'is_ref': False}

    def add_spectrum(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Bruker 1D Experiment Folder")
        if not folder:
            return
        entry = self._load_entry(folder)
        if entry is None:
            return
        if not self._series:
            entry['is_ref'] = True
        self._series.append(entry)
        self._refresh()

    def add_folder_batch(self):
        parent = QFileDialog.getExistingDirectory(
            self, "Select Parent Dataset Folder")
        if not parent:
            return
        candidates = []
        try:
            names = os.listdir(parent)
        except OSError:
            return
        for name in names:
            sub = os.path.join(parent, name)
            if not os.path.isdir(sub) or not name.isdigit():
                continue
            for pdr in ('pdata', 'data'):
                if os.path.exists(os.path.join(sub, pdr, '1', '1r')):
                    candidates.append((int(name), sub))
                    break
        if not candidates:
            QMessageBox.warning(self, "No spectra found",
                f"No numbered subfolders with pdata/1/1r found in:\n{parent}")
            return
        candidates.sort(key=lambda x: x[0])
        loaded = 0
        first  = not bool(self._series)
        for _, folder in candidates:
            entry = self._load_entry(folder)
            if entry is None:
                continue
            if first and loaded == 0:
                entry['is_ref'] = True
            self._series.append(entry)
            loaded += 1
        if loaded:
            QMessageBox.information(self, "Batch load",
                f"Loaded {loaded} spectra from:\n{parent}")
            self._refresh()

    def _refresh(self):
        self._cache = {}
        self._rebuild_table()
        self._detect_peaks()
        self._plot_stacked()
        self._update_buttons()

    def remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedItems()},
                      reverse=True)
        for r in rows:
            if 0 <= r < len(self._series):
                self._series.pop(r)
        self._refresh()

    def move_up(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if not rows or rows[0] == 0:
            return
        r = rows[0]
        self._series[r-1], self._series[r] = self._series[r], self._series[r-1]
        self._rebuild_table(); self.table.selectRow(r - 1)
        self._plot_stacked()

    def move_down(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if not rows or rows[-1] >= len(self._series) - 1:
            return
        r = rows[-1]
        self._series[r], self._series[r+1] = self._series[r+1], self._series[r]
        self._rebuild_table(); self.table.selectRow(r + 1)
        self._plot_stacked()

    def _rebuild_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for i, e in enumerate(self._series):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(e['label']))
            ci = QTableWidgetItem(str(e['conc']))
            ci.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 1, ci)
            combo = QComboBox()
            combo.addItems(['uM', 'mM', 'nM', 'mg/mL'])
            combo.setCurrentText(e['unit'])
            combo.currentTextChanged.connect(
                lambda t, idx=i: self._on_unit_changed(idx, t))
            self.table.setCellWidget(i, 2, combo)
            chk = QCheckBox()
            chk.setChecked(e['is_ref'])
            chk.setStyleSheet("margin-left: 8px;")
            chk.stateChanged.connect(
                lambda s, idx=i: self._on_ref_changed(idx, s))
            self.table.setCellWidget(i, 3, chk)
            if e['is_ref']:
                for col in range(2):
                    item = self.table.item(i, col)
                    if item:
                        item.setForeground(QColor(220, 80, 60))
                        f = item.font(); f.setBold(True); item.setFont(f)
        self.table.blockSignals(False)

    def _on_table_item_changed(self, item):
        r, c = item.row(), item.column()
        if r >= len(self._series):
            return
        if c == 0:
            self._series[r]['label'] = item.text(); self._plot_stacked()
        elif c == 1:
            try: self._series[r]['conc'] = float(item.text())
            except ValueError: pass

    def _on_unit_changed(self, idx, text):
        if idx < len(self._series):
            self._series[idx]['unit'] = text

    def _on_ref_changed(self, idx, state):
        if idx >= len(self._series):
            return
        if state == Qt.CheckState.Checked.value:
            for i, e in enumerate(self._series):
                e['is_ref'] = (i == idx)
        else:
            self._series[idx]['is_ref'] = False
        self._cache = {}
        self._rebuild_table(); self._detect_peaks(); self._plot_stacked()

    def _get_ref_idx(self):
        for i, e in enumerate(self._series):
            if e['is_ref']:
                return i
        return 0

    def _to_uM(self, conc, unit):
        if unit == 'mM': return conc * 1e3
        if unit == 'nM': return conc * 1e-3
        return float(conc)

    def _update_buttons(self):
        has_data  = bool(self._series)
        has_peaks = len(self._auto_peaks) > 0
        self.btn_prev.setEnabled(has_peaks)
        self.btn_next.setEnabled(has_peaks)
        self.btn_export.setEnabled(has_data)
        self.btn_run.setEnabled(
            len(self._series) >= 2 and self._peak_ppm is not None)

    def _on_sensitivity_changed(self, value):
        self.lbl_sens.setText(f"{value}×")
        # Invalidate detection cache so next _detect_peaks is fresh
        self._cache = {k: v for k, v in self._cache.items()
                       if not (len(k) > 1 and k[1] == 'det')}
        self._detect_peaks()
        self._plot_stacked()

    # -----------------------------------------------------------------------
    # Peak detection
    # -----------------------------------------------------------------------

    def _detect_peaks(self):
        if not self._series:
            self._auto_peaks  = []
            self._peak_cursor = -1
            self.lbl_cycle.setText("Load spectra to detect peaks")
            self._update_buttons()
            return

        ppm, spec     = self._prepare_detect(
            self._series[self._get_ref_idx()]['data'])
        ppm_d, spec_d = _minmax_decimate(ppm, spec, target=16384)
        sensitivity   = float(self.sld_sens.value())
        peaks         = _find_peaks(ppm_d, spec_d, sensitivity=sensitivity)

        self._auto_peaks  = list(peaks)
        self._peak_cursor = -1
        n = len(self._auto_peaks)
        if n == 0:
            self.lbl_cycle.setText(
                "No peaks detected\n"
                "Lower sensitivity, click the spectrum,\n"
                "or type a ppm value in 'Selected'")
        else:
            self.lbl_cycle.setText(
                f"{n} peak{'s' if n != 1 else ''} detected — "
                "click one or use ◀ ▶")
        self._update_buttons()

    # -----------------------------------------------------------------------
    # Peak cycling / selection
    # -----------------------------------------------------------------------

    def _prev_peak(self):
        if not self._auto_peaks: return
        self._peak_cursor = (self._peak_cursor - 1) % len(self._auto_peaks)
        self._select_peak(self._auto_peaks[self._peak_cursor], zoom=True)

    def _next_peak(self):
        if not self._auto_peaks: return
        self._peak_cursor = (self._peak_cursor + 1) % len(self._auto_peaks)
        self._select_peak(self._auto_peaks[self._peak_cursor], zoom=True)

    def _select_peak(self, ppm_val, zoom=False):
        self._peak_ppm = ppm_val
        self.spb_peak.setEnabled(True)
        self.spb_peak.setValue(ppm_val)
        if zoom:
            if self._view_lo is not None and self._view_hi is not None:
                hw = min(abs(self._view_lo - self._view_hi) / 2.0, 2.0)
            else:
                hw = 1.5
            self._view_lo = ppm_val + hw
            self._view_hi = ppm_val - hw
            self.spb_zlo.blockSignals(True); self.spb_zhi.blockSignals(True)
            self.spb_zlo.setValue(round(self._view_lo, 2))
            self.spb_zhi.setValue(round(self._view_hi, 2))
            self.spb_zlo.blockSignals(False); self.spb_zhi.blockSignals(False)
        n = len(self._auto_peaks)
        if 0 <= self._peak_cursor < n:
            self.lbl_cycle.setText(
                f"Peak {self._peak_cursor + 1} of {n}  "
                f"({self._auto_peaks[self._peak_cursor]:.3f} ppm)")
        self._plot_stacked()
        self._update_buttons()

    def _on_peak_spinbox_changed(self):
        self._peak_ppm    = self.spb_peak.value()
        self._peak_cursor = -1
        self._plot_stacked(); self._update_buttons()

    # -----------------------------------------------------------------------
    # Zoom / smooth / display options
    # -----------------------------------------------------------------------

    def _apply_zoom(self):
        self._view_lo = self.spb_zlo.value()
        self._view_hi = self.spb_zhi.value()
        self._plot_stacked()

    def _reset_zoom(self):
        self._view_lo = None; self._view_hi = None
        self._plot_stacked()

    def _on_smooth_changed(self, value):
        w = self._smooth_win()
        self.lbl_smooth.setText("Off" if w == 0 else str(w))
        # Invalidate display cache only; keep 'det' keys
        self._cache = {k: v for k, v in self._cache.items()
                       if len(k) > 1 and k[1] == 'det'}
        self._plot_stacked()

    def _on_display_option_changed(self):
        self._cache = {}   # baseline toggle affects everything
        self._detect_peaks()
        self._plot_stacked()

    # -----------------------------------------------------------------------
    # Waterfall plot
    # -----------------------------------------------------------------------

    def _plot_stacked(self):
        self.ax.clear()

        if not self._series:
            self._empty_plot()
            return

        indices = list(range(len(self._series)))
        if self.chk_sort.isChecked():
            indices.sort(key=lambda k: self._to_uM(
                self._series[k]['conc'], self._series[k]['unit']))

        has_zoom = self._view_lo is not None and self._view_hi is not None
        if has_zoom:
            lo_v = min(self._view_lo, self._view_hi)
            hi_v = max(self._view_lo, self._view_hi)

        # Simple per-spectrum normalization: each trace normalized to its 99th
        # percentile so a single spike doesn't squash the whole spectrum.
        # Spacing controlled by the slider.
        spacing_factor = self.sld_spacing.value()
        STEP = spacing_factor * 1.2

        prepared = []
        for k in indices:
            e         = self._series[k]
            ppm, spec = self._prepare_display(e['data'])
            nv = float(np.percentile(np.abs(spec), 99))
            if nv < 1e-12:
                nv = float(np.max(np.abs(spec))) or 1.0
            prepared.append((k, e, ppm, spec / nv))

        ytick_pos, ytick_lbl = [], []

        for row_i, (k, e, ppm, spec) in enumerate(prepared):
            ppm_d, spec_d = _minmax_decimate(ppm, spec)
            offset = row_i * STEP
            color  = '#C0392B' if e['is_ref'] else '#2C3E50'
            lw     = 1.1       if e['is_ref'] else 0.85
            self.ax.plot(ppm_d, spec_d + offset,
                         color=color, linewidth=lw, alpha=0.9)
            self.ax.axhline(offset, color='#C8C8C8', linewidth=0.3, zorder=0)

            c = e['conc']
            if c > 0:
                lbl = f"{int(c) if c == int(c) else c} {e['unit']}"
            else:
                # Build a meaningful label from the path:
                # show "parent_folder/exp_number" so mixed datasets are distinguishable
                raw = e['data'].get('path', '')
                if raw:
                    parts = raw.replace('\\', '/').rstrip('/').split('/')
                    # Take last two path components: e.g. "AW/1" or "dataset/3"
                    lbl = '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                else:
                    lbl = e['label']
                if len(lbl) > 24:
                    lbl = '…' + lbl[-22:]
            if e['is_ref']:
                lbl += '  ◀'
            ytick_pos.append(offset)
            ytick_lbl.append(lbl)

        self.ax.set_yticks(ytick_pos)
        self.ax.set_yticklabels(ytick_lbl, fontsize=8)

        # Detected peak tick marks
        for pk in self._auto_peaks:
            is_sel = (self._peak_ppm is not None
                      and abs(pk - self._peak_ppm) < 0.005)
            self.ax.axvline(
                pk,
                color='#E67E22' if is_sel else '#7F8C8D',
                linewidth=1.3 if is_sel else 0.6,
                linestyle='--' if is_sel else ':',
                alpha=0.9 if is_sel else 0.45,
                zorder=4)

        # Selected peak highlight
        if self._peak_ppm is not None:
            hw = self.spb_win.value()
            self.ax.axvspan(self._peak_ppm - hw, self._peak_ppm + hw,
                            alpha=0.10, color='#F39C12', zorder=1)
            self.ax.legend(
                handles=[Line2D([], [], color='#E67E22', linestyle='--',
                                linewidth=1.3,
                                label=f'selected  {self._peak_ppm:.4f} ppm')],
                fontsize=8, loc='upper right',
                framealpha=0.75, edgecolor='none')

        if has_zoom:
            self.ax.set_xlim(max(self._view_lo, self._view_hi),
                             min(self._view_lo, self._view_hi))

        self.ax.invert_xaxis()
        self.ax.set_xlabel("Chemical Shift (ppm)", fontsize=9)
        self.ax.set_ylabel("Concentration / ID", fontsize=9)

        n_pk   = len(self._auto_peaks)
        pk_n   = f"  ·  {n_pk} peaks" if n_pk else "  ·  no peaks detected"
        sort_n = "  ·  sorted by conc." if self.chk_sort.isChecked() else ""
        self.ax.set_title(
            f"NMR Waterfall  "
            f"({len(self._series)} spectra{sort_n}{pk_n})",
            fontsize=10)

        total = (len(prepared) - 1) * STEP + 1.0
        self.ax.set_ylim(-STEP * 0.5, total + STEP * 0.5)

        self._attach_interactions()
        self.fig.tight_layout()
        self.cvs.draw()

    # -----------------------------------------------------------------------
    # Mouse / span interaction
    # -----------------------------------------------------------------------

    def _attach_interactions(self):
        if self._click_cid is not None:
            try: self.cvs.mpl_disconnect(self._click_cid)
            except Exception: pass
        self._click_cid = self.cvs.mpl_connect(
            'button_press_event', self._on_click)
        self._span_sel = SpanSelector(
            self.ax, self._on_span,
            direction='horizontal', useblit=True,
            props=dict(alpha=0.25, facecolor='#F39C12'),
            interactive=True, drag_from_anywhere=False)

    def _on_click(self, event):
        if event.inaxes is not self.ax: return
        if event.button != 1 or event.xdata is None: return
        x = float(event.xdata)
        if self._auto_peaks:
            dists = np.abs(np.array(self._auto_peaks) - x)
            best  = int(np.argmin(dists))
            if float(dists[best]) < 0.5:
                self._peak_cursor = best
                self._select_peak(self._auto_peaks[best], zoom=True)
                return
        self._peak_cursor = -1
        self._select_peak(x, zoom=True)

    def _on_span(self, lo, hi):
        if abs(hi - lo) < 1e-6: return
        self._peak_cursor = -1
        self._select_peak((lo + hi) / 2.0, zoom=False)

    # -----------------------------------------------------------------------
    # Export waterfall
    # -----------------------------------------------------------------------

    def export_waterfall(self):
        if not self._series:
            return

        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        fmt = dlg.get_format()
        dpi = dlg.get_dpi()

        filter_map = {
            'png': "PNG image (*.png)",
            'pdf': "PDF file (*.pdf)",
            'svg': "SVG file (*.svg)",
        }
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Waterfall",
            f"waterfall.{fmt}",
            f"{filter_map[fmt]};;All files (*)"
        )
        if not filepath:
            return

        try:
            save_kwargs = {'bbox_inches': 'tight', 'facecolor': 'white'}
            if fmt == 'png':
                save_kwargs['dpi'] = dpi
            self.fig.savefig(filepath, format=fmt, **save_kwargs)
            QMessageBox.information(
                self, "Exported",
                f"Waterfall saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    # -----------------------------------------------------------------------
    # CSP Analysis — opens separate window
    # -----------------------------------------------------------------------

    def run_analysis(self):
        if len(self._series) < 2 or self._peak_ppm is None:
            return

        from app.csp_results_window import CSPResultsWindow

        ref_idx  = self._get_ref_idx()
        spectra  = [e['data'] for e in self._series]
        centres  = extract_peak_centres(spectra, self._peak_ppm,
                                        self.spb_win.value())
        dd = compute_delta_delta(centres, ref_idx)
        concs_uM = np.array([self._to_uM(e['conc'], e['unit'])
                              for e in self._series])
        labels   = [e['label'] for e in self._series]
        result   = fit_kd(concs_uM, dd, ref_idx)

        # Overlay fitted centres on waterfall
        self._plot_stacked()
        for c in centres:
            self.ax.axvline(c, color='#7F8C8D', linewidth=0.7,
                            linestyle=':', alpha=0.7, zorder=5)
        self.cvs.draw()

        # Open / refresh results window
        self._results_win = CSPResultsWindow(
            concs_uM=concs_uM,
            delta_delta=dd,
            kd_result=result,
            centres=centres,
            labels=labels,
            ref_idx=ref_idx,
            peak_ppm=self._peak_ppm,
            parent_series=self._series,
        )
        self._results_win.show()
        self._results_win.raise_()