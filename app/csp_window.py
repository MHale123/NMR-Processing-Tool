"""
CSP Binding Analysis Window
===========================
Mouse interactions on the waterfall
-------------------------------------
Left-click          : place peak marker at cursor ppm
Left-drag           : rubber-band zoom into selected region
Double-click (left) : reset to full view
Right-drag          : live phase correction on the selected spectrum
                      (horizontal = P0, vertical = P1)
                      — spectrum must be selected in the table first
"""

import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QFileDialog, QLabel,
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
from matplotlib.patches import Rectangle

from processing.loader import load_1d_spectrum
from processing.csp_processor import (
    extract_peak_centres, compute_delta_delta, fit_kd
)
from utils.export import export_waterfall as _export_waterfall_bw


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


def _apply_phase(spec, p0_deg, p1_deg):
    """Whole-spectrum phase rotation (used by preview widget)."""
    from scipy.signal import hilbert
    analytic  = hilbert(spec.astype(float))
    ramp      = np.linspace(0.0, 1.0, len(spec))
    angle_rad = np.deg2rad(p0_deg + p1_deg * ramp)
    return np.real(analytic * np.exp(1j * angle_rad))


def _apply_phase_region(spec, ppm, p0_deg, p1_deg, region_lo, region_hi):
    """
    Apply P0/P1 correction only within [region_lo, region_hi],
    blending edges with a cosine taper to avoid discontinuities.
    """
    from scipy.signal import hilbert
    spec = spec.astype(float).copy()
    ppm  = np.asarray(ppm, dtype=float)
    lo, hi = min(region_lo, region_hi), max(region_lo, region_hi)
    idx = np.where((ppm >= lo) & (ppm <= hi))[0]
    if len(idx) < 4:
        return spec
    analytic = hilbert(spec)
    n_win    = len(idx)
    frac     = np.linspace(0.0, 1.0, n_win)
    angle    = np.deg2rad(p0_deg + p1_deg * frac)
    rotated  = np.real(analytic[idx] * np.exp(1j * angle))
    taper_n  = max(5, n_win // 10)
    taper    = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, taper_n))
    result       = spec.copy()
    result[idx]  = rotated
    result[idx[:taper_n]]  = spec[idx[:taper_n]]  * (1 - taper) + rotated[:taper_n]  * taper
    result[idx[-taper_n:]] = rotated[-taper_n:] * (1 - taper[::-1]) + spec[idx[-taper_n:]] * taper[::-1]
    return result


def _phase_correct(spec):
    return -spec if np.max(spec) < -np.min(spec) else spec.copy()


def _iterative_poly_baseline(ppm, spec, degree=5, n_iter=10, threshold=2.0):
    x, y  = ppm.copy(), spec.copy()
    mask  = np.ones(len(y), dtype=bool)
    for _ in range(n_iter):
        if mask.sum() < degree + 2:
            break
        coeffs   = np.polyfit(x[mask], y[mask], degree)
        baseline = np.polyval(coeffs, x)
        residual = y - baseline
        rms      = np.sqrt(np.mean(residual[mask] ** 2))
        new_mask = residual < threshold * rms
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    coeffs   = np.polyfit(x[mask], y[mask], degree)
    return spec - np.polyval(coeffs, x)


def _minmax_decimate(ppm, spec, target=6000):
    n = len(spec)
    if n <= target:
        return ppm, spec
    block    = max(1, n // (target // 2))
    n_blocks = n // block
    pd, sd   = [], []
    for b in range(n_blocks):
        sl = slice(b * block, (b + 1) * block)
        pb, sb = ppm[sl], spec[sl]
        imin, imax = int(np.argmin(sb)), int(np.argmax(sb))
        if imin <= imax:
            pd += [pb[imin], pb[imax]]; sd += [sb[imin], sb[imax]]
        else:
            pd += [pb[imax], pb[imin]]; sd += [sb[imax], sb[imin]]
    return np.array(pd), np.array(sd)


def _integrate_region(ppm, spec, lo, hi):
    lo, hi = min(lo, hi), max(lo, hi)
    mask   = (ppm >= lo) & (ppm <= hi)
    if mask.sum() < 2:
        return 0.0
    ppm_w, spec_w = ppm[mask], spec[mask]
    order = np.argsort(ppm_w)
    return float(np.trapz(spec_w[order], ppm_w[order]))


# ---------------------------------------------------------------------------
# Phase-preview widget (used inside PhaseDialog)
# ---------------------------------------------------------------------------

class _SpectrumPhaseWidget(QWidget):
    def __init__(self, label, ppm, spec_raw, region_lo, region_hi,
                 init_p0=0.0, init_p1=0.0, parent=None):
        super().__init__(parent)
        self._ppm      = ppm
        self._spec_raw = spec_raw
        self._lo = min(region_lo, region_hi)
        self._hi = max(region_lo, region_hi)
        pad = max((self._hi - self._lo) * 0.3, 0.05)
        self._disp_lo = self._lo - pad
        self._disp_hi = self._hi + pad
        self.p0 = init_p0
        self.p1 = init_p1

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self.fig = Figure(figsize=(5, 2.8), tight_layout=True)
        self.cvs = FigureCanvas(self.fig)
        self.cvs.setMinimumHeight(200)
        self.ax  = self.fig.add_subplot(111)
        layout.addWidget(self.cvs)

        for name, attr, mn, mx, tick, init in [
            ("P0  —  Zero-order phase  (°)",   'sld_p0', -18000, 18000, 4500, init_p0),
            ("P1  —  First-order phase  (°)",  'sld_p1', -36000, 36000, 9000, init_p1),
        ]:
            grp = QGroupBox(name)
            gv  = QVBoxLayout(grp)
            row = QHBoxLayout()
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setMinimum(mn); sld.setMaximum(mx)
            sld.setValue(int(round(init * 100)))
            sld.setTickInterval(tick)
            sld.setTickPosition(QSlider.TickPosition.TicksBelow)
            setattr(self, attr, sld)
            lbl = QLabel(f"{init:.1f}°")
            lbl.setFixedWidth(56)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            setattr(self, attr.replace('sld_', 'lbl_'), lbl)
            row.addWidget(sld); row.addWidget(lbl)
            gv.addLayout(row)
            btn_r = QPushButton(f"Reset {name[:2]}"); btn_r.setFixedWidth(80)
            btn_r.clicked.connect(lambda _, s=sld: s.setValue(0))
            gv.addWidget(btn_r, alignment=Qt.AlignmentFlag.AlignRight)
            layout.addWidget(grp)

        btn_all = QPushButton("Reset Both to 0°")
        btn_all.clicked.connect(lambda: (self.sld_p0.setValue(0), self.sld_p1.setValue(0)))
        layout.addWidget(btn_all)

        self.sld_p0.valueChanged.connect(self._on_p0)
        self.sld_p1.valueChanged.connect(self._on_p1)
        self._draw()

    def _on_p0(self, v):
        self.p0 = v / 100.0; self.lbl_p0.setText(f"{self.p0:.1f}°"); self._draw()

    def _on_p1(self, v):
        self.p1 = v / 100.0; self.lbl_p1.setText(f"{self.p1:.1f}°"); self._draw()

    def set_phase(self, p0, p1):
        """External setter — used by programmatic phase updates."""
        self.sld_p0.setValue(int(round(p0 * 100)))
        self.sld_p1.setValue(int(round(p1 * 100)))

    def _draw(self):
        ax, ppm = self.ax, self._ppm
        ax.clear()
        lo = min(self._disp_lo, self._disp_hi)
        hi = max(self._disp_lo, self._disp_hi)
        mask   = (ppm >= lo) & (ppm <= hi)
        ppm_w  = ppm[mask]
        spec_w = _apply_phase(self._spec_raw, self.p0, self.p1)[mask]
        if len(ppm_w) > 2:
            ppm_d, spec_d = _minmax_decimate(ppm_w, spec_w, target=4000)
            ax.plot(ppm_d, spec_d, color='#2C3E50', linewidth=0.9)
        ax.axvspan(min(self._lo, self._hi), max(self._lo, self._hi),
                   alpha=0.12, color='#F39C12')
        ax.axhline(0, color='#BDC3C7', linewidth=0.6)
        ax.invert_xaxis()
        ax.set_xlabel("Chemical Shift (ppm)", fontsize=8)
        ax.set_ylabel("Intensity", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"P0 = {self.p0:.1f}°   P1 = {self.p1:.1f}°", fontsize=8)
        self.fig.tight_layout()
        self.cvs.draw()


# ---------------------------------------------------------------------------
# Phase dialog
# ---------------------------------------------------------------------------

class PhaseDialog(QDialog):
    def __init__(self, label, ppm, spec_raw, region_lo, region_hi,
                 init_p0=0.0, init_p1=0.0, parent=None):
        super().__init__(parent)
        lo, hi = min(region_lo, region_hi), max(region_lo, region_hi)
        self.setWindowTitle(f"Phase Correction  ·  {label}  ·  {lo:.3f}–{hi:.3f} ppm")
        self.setMinimumSize(620, 560); self.resize(700, 600)
        layout = QVBoxLayout(self)
        info = QLabel(
            f"Phasing: <b>{label}</b>  ({lo:.3f}–{hi:.3f} ppm).  "
            "Adjust P0/P1 — preview updates live.  OK to apply, Cancel to discard.")
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 10px; color: #555;")
        layout.addWidget(info)
        self._widget = _SpectrumPhaseWidget(
            label=label, ppm=ppm, spec_raw=spec_raw,
            region_lo=region_lo, region_hi=region_hi,
            init_p0=init_p0, init_p1=init_p1, parent=self)
        layout.addWidget(self._widget)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    @property
    def p0(self): return self._widget.p0
    @property
    def p1(self): return self._widget.p1


# ---------------------------------------------------------------------------
# Main CSP window
# ---------------------------------------------------------------------------


# Distinct per-spectrum colours for on-screen display (cycles if >12 spectra)
_SPECTRUM_COLORS = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
    '#9467bd', '#8c564b', '#e377c2', '#17becf',
    '#bcbd22', '#7f7f7f', '#aec7e8', '#ffbb78',
]

class CSPWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSP Binding Analysis")
        self.setGeometry(120, 120, 1350, 780)

        self._series    = []
        self._peak_ppm  = None
        self._view_lo   = None
        self._view_hi   = None
        self._phases: dict[int, tuple] = {}   # idx → (p0, p1, rlo, rhi)
        self._offsets: dict[int, float] = {}  # idx → ppm shift offset
        self._cache     = {}

        # Interaction state
        self._cids      = []        # matplotlib callback ids
        self._drag_mode = None      # 'zoom' | None
        self._drag_x0   = None      # press x (data coords or pixel)
        self._drag_y0   = None
        self._zoom_rect = None      # Rectangle patch for rubber-band
        self._row_map     = []      # row_i → series index (updated each plot)
        self._row_offsets = []      # y-data coord of each row

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QWidget()
        left.setFixedWidth(300)
        lv = QVBoxLayout(left)
        lv.setAlignment(Qt.AlignmentFlag.AlignTop)
        lv.setSpacing(5)

        # ── Load ────────────────────────────────────────────────────────────
        load_grp = QGroupBox("Load Data")
        load_v   = QVBoxLayout(load_grp)
        b_add = QPushButton("Add Spectrum…")
        b_add.clicked.connect(self.add_spectrum)
        load_v.addWidget(b_add)
        lv.addWidget(load_grp)

        # ── Concentration Series (no Ref column) ────────────────────────────
        series_grp = QGroupBox("Concentration Series")
        series_v   = QVBoxLayout(series_grp)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Label", "Conc", "Unit"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 65)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 55)
        self.table.setMinimumHeight(80)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        series_v.addWidget(self.table)
        tbl_btns = QHBoxLayout()
        b_rem = QPushButton("Remove"); b_rem.clicked.connect(self.remove_selected)
        tbl_btns.addWidget(b_rem)
        b_up = QPushButton("▲"); b_up.setFixedWidth(30); b_up.clicked.connect(self.move_up)
        tbl_btns.addWidget(b_up)
        b_dn = QPushButton("▼"); b_dn.setFixedWidth(30); b_dn.clicked.connect(self.move_down)
        tbl_btns.addWidget(b_dn)
        series_v.addLayout(tbl_btns)
        lv.addWidget(series_grp)

        # ── Peak / Region ────────────────────────────────────────────────────
        peak_grp = QGroupBox("Peak / Region")
        peak_v   = QVBoxLayout(peak_grp)
        peak_v.setSpacing(4)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Centre:"))
        self.spb_peak = QDoubleSpinBox()
        self.spb_peak.setDecimals(4); self.spb_peak.setRange(-10000, 10000)
        self.spb_peak.setEnabled(False); self.spb_peak.setFixedWidth(88)
        self.spb_peak.editingFinished.connect(self._on_peak_spinbox_changed)
        row1.addWidget(self.spb_peak); row1.addWidget(QLabel("ppm"))
        peak_v.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Window ±"))
        self.spb_win = QDoubleSpinBox()
        self.spb_win.setDecimals(3); self.spb_win.setRange(0.01, 50.0)
        self.spb_win.setValue(0.15); self.spb_win.setFixedWidth(72)
        self.spb_win.setToolTip("Half-width of integration / phasing region (ppm).")
        row2.addWidget(self.spb_win); row2.addWidget(QLabel("ppm"))
        peak_v.addLayout(row2)
        lv.addWidget(peak_grp)

        # ── Phase correction ─────────────────────────────────────────────────
        phase_grp = QGroupBox("Phase Correction")
        phase_v   = QVBoxLayout(phase_grp)
        phase_v.setSpacing(4)

        self.lbl_phase_target = QLabel("No spectrum selected")
        self.lbl_phase_target.setStyleSheet("font-size: 10px; color: #888;")
        self.lbl_phase_target.setWordWrap(True)
        phase_v.addWidget(self.lbl_phase_target)

        hint = QLabel("Select a row, set a peak, then open\nthe dialog to phase that spectrum.")
        hint.setStyleSheet("font-size: 10px; color: gray;")
        phase_v.addWidget(hint)

        self.btn_phase = QPushButton("Open Phase Dialog…")
        self.btn_phase.setEnabled(False)
        self.btn_phase.clicked.connect(self.open_phase_dialog)
        phase_v.addWidget(self.btn_phase)

        self.btn_clear_phases = QPushButton("Clear All Phase Corrections")
        self.btn_clear_phases.setEnabled(False)
        self.btn_clear_phases.clicked.connect(self._clear_phases)
        phase_v.addWidget(self.btn_clear_phases)

        # ppm offset for the selected spectrum
        off_row = QHBoxLayout()
        off_row.addWidget(QLabel("ppm offset:"))
        self.spb_offset = QDoubleSpinBox()
        self.spb_offset.setDecimals(4); self.spb_offset.setRange(-100.0, 100.0)
        self.spb_offset.setValue(0.0); self.spb_offset.setFixedWidth(80)
        self.spb_offset.setToolTip(
            "Shift the selected spectrum left/right by this many ppm.\n"
            "Also: Shift+scroll on the waterfall to nudge interactively.")
        self.spb_offset.editingFinished.connect(self._on_offset_spinbox_changed)
        off_row.addWidget(self.spb_offset)
        btn_off_reset = QPushButton("Reset")
        btn_off_reset.setFixedWidth(46)
        btn_off_reset.clicked.connect(self._reset_offset)
        off_row.addWidget(btn_off_reset)
        phase_v.addLayout(off_row)
        lv.addWidget(phase_grp)

        # ── View ─────────────────────────────────────────────────────────────
        view_grp = QGroupBox("View")
        view_v   = QVBoxLayout(view_grp)
        view_v.setSpacing(4)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom:"))
        self.spb_zlo = QDoubleSpinBox()
        self.spb_zlo.setDecimals(2); self.spb_zlo.setRange(-2000, 2000)
        self.spb_zlo.setValue(-60); self.spb_zlo.setFixedWidth(65)
        zoom_row.addWidget(self.spb_zlo)
        zoom_row.addWidget(QLabel("to"))
        self.spb_zhi = QDoubleSpinBox()
        self.spb_zhi.setDecimals(2); self.spb_zhi.setRange(-2000, 2000)
        self.spb_zhi.setValue(-140); self.spb_zhi.setFixedWidth(65)
        zoom_row.addWidget(self.spb_zhi); zoom_row.addWidget(QLabel("ppm"))
        view_v.addLayout(zoom_row)

        zoom_btns = QHBoxLayout()
        b_z = QPushButton("Apply Zoom"); b_z.clicked.connect(self._apply_zoom)
        zoom_btns.addWidget(b_z)
        b_r = QPushButton("Full View");  b_r.clicked.connect(self._reset_zoom)
        zoom_btns.addWidget(b_r)
        view_v.addLayout(zoom_btns)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Smooth:"))
        self.sld_smooth = QSlider(Qt.Orientation.Horizontal)
        self.sld_smooth.setMinimum(0); self.sld_smooth.setMaximum(20); self.sld_smooth.setValue(0)
        self.sld_smooth.setTickInterval(5)
        self.sld_smooth.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_smooth.valueChanged.connect(self._on_smooth_changed)
        smooth_row.addWidget(self.sld_smooth)
        self.lbl_smooth = QLabel("Off"); self.lbl_smooth.setFixedWidth(30)
        smooth_row.addWidget(self.lbl_smooth)
        view_v.addLayout(smooth_row)

        spacing_row = QHBoxLayout()
        spacing_row.addWidget(QLabel("Spacing:"))
        self.sld_spacing = QSlider(Qt.Orientation.Horizontal)
        self.sld_spacing.setMinimum(1); self.sld_spacing.setMaximum(10); self.sld_spacing.setValue(2)
        self.sld_spacing.setTickInterval(1)
        self.sld_spacing.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_spacing.valueChanged.connect(lambda _: self._plot_stacked())
        spacing_row.addWidget(self.sld_spacing)
        self.lbl_spacing = QLabel("2"); self.lbl_spacing.setFixedWidth(20)
        self.sld_spacing.valueChanged.connect(lambda v: self.lbl_spacing.setText(str(v)))
        spacing_row.addWidget(self.lbl_spacing)
        view_v.addLayout(spacing_row)

        self.chk_baseline = QCheckBox("Auto baseline correction")
        self.chk_baseline.setChecked(True)
        self.chk_baseline.stateChanged.connect(self._on_display_option_changed)
        view_v.addWidget(self.chk_baseline)

        self.chk_sort = QCheckBox("Sort by concentration")
        self.chk_sort.setChecked(True)
        self.chk_sort.stateChanged.connect(lambda _: self._plot_stacked())
        view_v.addWidget(self.chk_sort)

        self.chk_region_only = QCheckBox("Show region only")
        self.chk_region_only.setToolTip(
            "Crop every trace to peak ± window, normalise each to fit,\n"
            "and annotate with trapezoidal integral.")
        self.chk_region_only.stateChanged.connect(lambda _: self._plot_stacked())
        view_v.addWidget(self.chk_region_only)

        lv.addWidget(view_grp)

        # ── Export ───────────────────────────────────────────────────────────
        export_grp = QGroupBox("Export")
        export_v   = QVBoxLayout(export_grp)
        self.btn_export = QPushButton("Export Waterfall…")
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip("Save current waterfall as PNG, PDF, or SVG.")
        self.btn_export.clicked.connect(self.export_waterfall)
        export_v.addWidget(self.btn_export)
        lv.addWidget(export_grp)

        lv.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        root.addWidget(left)

        # ── Right — waterfall canvas ─────────────────────────────────────────
        self.fig = Figure(figsize=(9, 6))
        self.cvs = FigureCanvas(self.fig)
        self.ax  = self.fig.add_subplot(111)
        self._empty_plot()
        root.addWidget(self.cvs)

    def _empty_plot(self):
        self.ax.clear()
        self.ax.set_title("Load spectra to begin", fontsize=11)
        self.ax.set_xlabel("Chemical Shift (ppm)")
        self.ax.set_ylabel("Concentration / ID")
        self.ax.text(0.5, 0.5, 'Use "Add Spectrum…" to load Bruker 1D data',
                     transform=self.ax.transAxes,
                     ha='center', va='center', color='gray', fontsize=10)
        self.cvs.draw()

    # -----------------------------------------------------------------------
    # Spectrum preparation
    # -----------------------------------------------------------------------

    def _smooth_win(self):
        v = self.sld_smooth.value()
        return 0 if v == 0 else v * 2 + 3

    def _prepare_display(self, data, series_idx):
        p0, p1, rlo, rhi = self._phases.get(series_idx, (0.0, 0.0, 0.0, 0.0))
        offset_ppm = self._offsets.get(series_idx, 0.0)
        key = (id(data['spectrum']), self._smooth_win(),
               self.chk_baseline.isChecked(), p0, p1, rlo, rhi, offset_ppm)
        if key not in self._cache:
            ppm  = data['ppm'] + offset_ppm   # apply ppm shift
            spec = _phase_correct(data['spectrum'].astype(float))
            if self.chk_baseline.isChecked():
                spec = _iterative_poly_baseline(ppm, spec)
            if (p0 != 0.0 or p1 != 0.0) and rlo != rhi:
                spec = _apply_phase_region(spec, ppm, p0, p1, rlo, rhi)
            spec = _savgol(spec, self._smooth_win())
            self._cache[key] = (ppm, spec)
        return self._cache[key]

    # -----------------------------------------------------------------------
    # Series management
    # -----------------------------------------------------------------------

    def _load_entry(self, folder):
        try:
            data = load_1d_spectrum(folder)
        except (FileNotFoundError, ValueError) as e:
            QMessageBox.critical(self, "Load failed", f"{os.path.basename(folder)}:\n{e}")
            return None
        return {'data': data, 'conc': 0.0, 'unit': 'uM',
                'label': os.path.basename(folder), 'is_ref': False}

    def add_spectrum(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Bruker 1D Experiment Folder")
        if not folder: return
        entry = self._load_entry(folder)
        if entry is None: return
        self._series.append(entry)
        self._refresh()

    def _refresh(self):
        self._cache = {}
        for i in range(len(self._series)):
            if i not in self._phases:
                self._phases[i] = (0.0, 0.0, 0.0, 0.0)
            if i not in self._offsets:
                self._offsets[i] = 0.0
        self._rebuild_table()
        self._plot_stacked()
        self._update_buttons()

    def remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self._series):
                self._series.pop(r)
        self._phases  = {i: self._phases.get(i,  (0.0, 0.0, 0.0, 0.0))
                         for i in range(len(self._series))}
        self._offsets = {i: self._offsets.get(i, 0.0)
                         for i in range(len(self._series))}
        self._refresh()

    def move_up(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if not rows or rows[0] == 0: return
        r = rows[0]
        self._series[r-1], self._series[r] = self._series[r], self._series[r-1]
        self._phases[r-1], self._phases[r] = (
            self._phases.get(r,   (0.0, 0.0, 0.0, 0.0)),
            self._phases.get(r-1, (0.0, 0.0, 0.0, 0.0)))
        self._offsets[r-1], self._offsets[r] = (
            self._offsets.get(r,   0.0),
            self._offsets.get(r-1, 0.0))
        self._rebuild_table(); self.table.selectRow(r - 1)

    def move_down(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if not rows or rows[-1] >= len(self._series) - 1: return
        r = rows[-1]
        self._series[r], self._series[r+1] = self._series[r+1], self._series[r]
        self._phases[r], self._phases[r+1] = (
            self._phases.get(r+1, (0.0, 0.0, 0.0, 0.0)),
            self._phases.get(r,   (0.0, 0.0, 0.0, 0.0)))
        self._offsets[r], self._offsets[r+1] = (
            self._offsets.get(r+1, 0.0),
            self._offsets.get(r,   0.0))
        self._rebuild_table(); self.table.selectRow(r + 1)

    def _rebuild_table(self):
        sel_idx = self._selected_series_idx()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for i, e in enumerate(self._series):
            self.table.insertRow(i)
            lbl_item = QTableWidgetItem(e['label'])
            # Persistent highlight: selected row gets a warm tint
            if i == sel_idx:
                bg = QColor(255, 200, 100, 60)   # amber tint
                lbl_item.setBackground(bg)
            self.table.setItem(i, 0, lbl_item)
            ci = QTableWidgetItem(str(e['conc']))
            ci.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if i == sel_idx:
                ci.setBackground(QColor(255, 200, 100, 60))
            self.table.setItem(i, 1, ci)
            combo = QComboBox()
            combo.addItems(['uM', 'mM', 'nM', 'mg/mL'])
            combo.setCurrentText(e['unit'])
            combo.currentTextChanged.connect(lambda t, idx=i: self._on_unit_changed(idx, t))
            self.table.setCellWidget(i, 2, combo)
        # Re-apply Qt selection highlight while signals are still blocked
        if sel_idx is not None:
            self.table.selectRow(sel_idx)
        self.table.blockSignals(False)

    def _on_table_item_changed(self, item):
        r, c = item.row(), item.column()
        if r >= len(self._series): return
        if c == 0:
            self._series[r]['label'] = item.text(); self._plot_stacked()
        elif c == 1:
            try: self._series[r]['conc'] = float(item.text())
            except ValueError: pass

    def _on_unit_changed(self, idx, text):
        if idx < len(self._series):
            self._series[idx]['unit'] = text

    def _get_ref_idx(self):
        for i, e in enumerate(self._series):
            if e['is_ref']: return i
        return 0

    def _to_uM(self, conc, unit):
        if unit == 'mM': return conc * 1e3
        if unit == 'nM': return conc * 1e-3
        return float(conc)

    def _selected_series_idx(self):
        rows = list({i.row() for i in self.table.selectedItems()})
        if len(rows) == 1 and 0 <= rows[0] < len(self._series):
            return rows[0]
        return None

    def _update_buttons(self):
        has_data   = bool(self._series)
        has_peak   = self._peak_ppm is not None
        has_phases = any(p[0] != 0.0 or p[1] != 0.0 for p in self._phases.values())
        has_sel    = self._selected_series_idx() is not None
        self.btn_phase.setEnabled(has_data and has_peak and has_sel)
        self.btn_clear_phases.setEnabled(has_phases)
        self.btn_export.setEnabled(has_data)

    def _on_table_selection_changed(self):
        idx = self._selected_series_idx()
        if idx is not None and idx < len(self._series):
            lbl = self._series[idx]['label']
            ph  = self._phases.get(idx, (0.0, 0.0, 0.0, 0.0))
            p0, p1 = ph[0], ph[1]
            suffix = f"  φ({p0:.0f}°, {p1:.0f}°)" if (p0 or p1) else ""
            self.lbl_phase_target.setText(f"Active: {lbl}{suffix}")
        else:
            self.lbl_phase_target.setText("No spectrum selected")
        # Update amber tint on rows without rebuilding the whole table
        self.table.blockSignals(True)
        for i in range(self.table.rowCount()):
            bg = QColor(255, 200, 100, 60) if i == idx else QColor(0, 0, 0, 0)
            for col in range(self.table.columnCount()):
                item = self.table.item(i, col)
                if item:
                    item.setBackground(bg)
        self.table.blockSignals(False)
        # Sync offset spinbox to selected spectrum
        if idx is not None and idx < len(self._series):
            self.spb_offset.blockSignals(True)
            self.spb_offset.setValue(self._offsets.get(idx, 0.0))
            self.spb_offset.blockSignals(False)
        self._update_buttons()
        self._plot_stacked()

    # -----------------------------------------------------------------------
    # Phase correction
    # -----------------------------------------------------------------------

    def open_phase_dialog(self):
        if not self._series or self._peak_ppm is None: return
        idx = self._selected_series_idx()
        if idx is None: idx = 0; self.table.selectRow(0)
        entry     = self._series[idx]
        hw        = self.spb_win.value()
        region_lo = self._peak_ppm - hw
        region_hi = self._peak_ppm + hw
        ppm       = entry['data']['ppm']
        spec_raw  = _phase_correct(entry['data']['spectrum'].astype(float))
        if self.chk_baseline.isChecked():
            spec_raw = _iterative_poly_baseline(ppm, spec_raw)
        _ph = self._phases.get(idx, (0.0, 0.0, 0.0, 0.0))
        dlg = PhaseDialog(
            label=entry['label'], ppm=ppm, spec_raw=spec_raw,
            region_lo=region_lo, region_hi=region_hi,
            init_p0=_ph[0], init_p1=_ph[1], parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._phases[idx] = (dlg.p0, dlg.p1, region_lo, region_hi)
            self._cache = {}
            self._on_table_selection_changed()  # updates label, tint, buttons, and plot

    def _clear_phases(self):
        for i in range(len(self._series)):
            self._phases[i] = (0.0, 0.0, 0.0, 0.0)
        self._cache = {}
        self._update_buttons()
        self._plot_stacked()

    def _on_offset_spinbox_changed(self):
        idx = self._selected_series_idx()
        if idx is None: return
        self._offsets[idx] = self.spb_offset.value()
        data = self._series[idx]['data']
        self._cache = {k: v for k, v in self._cache.items()
                       if k[0] != id(data['spectrum'])}
        self._plot_stacked()

    def _reset_offset(self):
        idx = self._selected_series_idx()
        if idx is None: return
        self._offsets[idx] = 0.0
        self.spb_offset.blockSignals(True)
        self.spb_offset.setValue(0.0)
        self.spb_offset.blockSignals(False)
        data = self._series[idx]['data']
        self._cache = {k: v for k, v in self._cache.items()
                       if k[0] != id(data['spectrum'])}
        self._plot_stacked()

    # -----------------------------------------------------------------------
    # View controls
    # -----------------------------------------------------------------------

    def _apply_zoom(self):
        self._view_lo = self.spb_zlo.value()
        self._view_hi = self.spb_zhi.value()
        self._plot_stacked()

    def _reset_zoom(self):
        self._view_lo = None; self._view_hi = None
        # Also exit region-only mode so Full View always shows the complete spectrum
        if self.chk_region_only.isChecked():
            self.chk_region_only.blockSignals(True)
            self.chk_region_only.setChecked(False)
            self.chk_region_only.blockSignals(False)
        self._plot_stacked()

    def _on_smooth_changed(self, value):
        self.lbl_smooth.setText("Off" if value == 0 else str(self._smooth_win()))
        self._cache = {}; self._plot_stacked()

    def _on_display_option_changed(self):
        self._cache = {}; self._plot_stacked()

    # -----------------------------------------------------------------------
    # Waterfall plot
    # -----------------------------------------------------------------------

    def _plot_stacked(self):
        self.ax.clear()
        if not self._series:
            self._empty_plot(); return

        indices = list(range(len(self._series)))
        if self.chk_sort.isChecked():
            indices.sort(key=lambda k: self._to_uM(
                self._series[k]['conc'], self._series[k]['unit']))

        region_mode = self.chk_region_only.isChecked() and self._peak_ppm is not None
        hw   = self.spb_win.value()
        r_lo = (self._peak_ppm - hw) if region_mode else None
        r_hi = (self._peak_ppm + hw) if region_mode else None
        STEP = self.sld_spacing.value() * 1.2
        sel_idx = self._selected_series_idx()

        prepared = []
        for k in indices:
            e = self._series[k]
            ppm, spec = self._prepare_display(e['data'], k)
            if region_mode:
                lo_m, hi_m = min(r_lo, r_hi), max(r_lo, r_hi)
                mask  = (ppm >= lo_m) & (ppm <= hi_m)
                ppm_c = ppm[mask]; spec_c = spec[mask]
                nv = float(np.max(np.abs(spec_c))) if len(spec_c) else 1.0
            else:
                ppm_c = ppm; spec_c = spec
                if self._view_lo is not None:
                    vmin = min(self._view_lo, self._view_hi)
                    vmax = max(self._view_lo, self._view_hi)
                    vmask = (ppm_c >= vmin) & (ppm_c <= vmax)
                    nv = float(np.max(np.abs(spec_c[vmask]))) if vmask.any() else 1.0
                else:
                    nv = float(np.percentile(np.abs(spec_c), 99))
            nv = max(nv, 1e-12)
            spec_n = np.clip(spec_c / nv, -1.0, 1.0)
            ph    = self._phases.get(k, (0.0, 0.0, 0.0, 0.0))
            integ = None
            if ph[0] != 0.0 or ph[1] != 0.0:
                integ = _integrate_region(ppm, spec, ph[2], ph[3])
            prepared.append((k, e, ppm_c, spec_n, integ))

        ytick_pos, ytick_lbl = [], []

        for row_i, (k, e, ppm, spec, integ) in enumerate(prepared):
            ppm_d, spec_d = _minmax_decimate(ppm, spec)
            offset    = row_i * STEP
            ph        = self._phases.get(k, (0.0, 0.0, 0.0, 0.0))
            has_phase = ph[0] != 0.0 or ph[1] != 0.0
            is_active = (k == sel_idx)

            # Per-spectrum colour from palette; active = teal, phased = purple overlay
            base_color = _SPECTRUM_COLORS[k % len(_SPECTRUM_COLORS)]
            if is_active:
                color = '#16A085'; lw = 1.6
            elif has_phase:
                color = '#8E44AD'; lw = 1.1
            else:
                color = base_color; lw = 0.85

            self.ax.plot(ppm_d, spec_d + offset, color=color, linewidth=lw, alpha=0.9)
            self.ax.axhline(offset, color='#C8C8C8', linewidth=0.3, zorder=0)

            # Subtle background band on the active row
            if is_active:
                total_w = abs((self.ax.get_xlim()[0] if self._view_lo is None
                               else self._view_lo) -
                              (self.ax.get_xlim()[1] if self._view_hi is None
                               else self._view_hi))
                self.ax.axhspan(offset - STEP * 0.45, offset + STEP * 0.45,
                                color='#16A085', alpha=0.06, zorder=0)

            c = e['conc']
            if c > 0:
                lbl = f"{int(c) if c == int(c) else c} {e['unit']}"
            else:
                raw = e['data'].get('path', '')
                if raw:
                    parts = raw.replace('\\', '/').rstrip('/').split('/')
                    lbl = '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                else:
                    lbl = e['label']
                if len(lbl) > 22: lbl = '…' + lbl[-20:]

            if is_active: lbl = '▶ ' + lbl
            if has_phase: lbl += f'  φ({ph[0]:.0f}°,{ph[1]:.0f}°)'
            off = self._offsets.get(k, 0.0)
            if off != 0.0: lbl += f'  Δ{off:+.3f}'
            if integ is not None: lbl += f'  ∫={integ:.3g}'

            ytick_pos.append(offset)
            ytick_lbl.append(lbl)

        self.ax.set_yticks(ytick_pos)
        self.ax.set_yticklabels(ytick_lbl, fontsize=8)

        # Store mapping so label clicks can identify which spectrum was clicked
        # _row_map[row_i] = series index k
        self._row_map    = [k for k, e, ppm, spec, integ in prepared]
        self._row_offsets = ytick_pos[:]   # y-data coordinate of each row

        if self._peak_ppm is not None and not region_mode:
            hw2 = self.spb_win.value()
            self.ax.axvspan(self._peak_ppm - hw2, self._peak_ppm + hw2,
                            alpha=0.10, color='#F39C12', zorder=1)
            self.ax.axvline(self._peak_ppm, color='#E67E22',
                            linewidth=1.3, linestyle='--', alpha=0.9, zorder=4)
            self.ax.legend(
                handles=[Line2D([], [], color='#E67E22', linestyle='--', linewidth=1.3,
                                label=f'selected  {self._peak_ppm:.4f} ppm')],
                fontsize=8, loc='upper right', framealpha=0.75, edgecolor='none')

        if region_mode:
            self.ax.axvline(r_lo, color='#F39C12', linewidth=0.8, linestyle=':', alpha=0.7)
            self.ax.axvline(r_hi, color='#F39C12', linewidth=0.8, linestyle=':', alpha=0.7)
            span = abs(r_hi - r_lo); pad = span * 0.1
            self.ax.set_xlim(r_hi + pad, r_lo - pad)
        elif self._view_lo is not None:
            self.ax.set_xlim(max(self._view_lo, self._view_hi),
                             min(self._view_lo, self._view_hi))

        self.ax.invert_xaxis()
        self.ax.set_xlabel("Chemical Shift (ppm)", fontsize=9)
        self.ax.set_ylabel("Concentration / ID", fontsize=9)

        n_phased = sum(1 for p in self._phases.values() if p[0] != 0.0 or p[1] != 0.0)
        parts = [f"{len(self._series)} spectra"]
        if self.chk_sort.isChecked(): parts.append("sorted")
        if n_phased:                  parts.append(f"{n_phased} phased")
        if region_mode:               parts.append("region view")
        self.ax.set_title("NMR Waterfall  ·  " + "  ·  ".join(parts), fontsize=10)

        total = (len(prepared) - 1) * STEP + 1.0
        self.ax.set_ylim(-STEP * 0.5, total + STEP * 0.5)

        self._attach_interactions()
        self.fig.tight_layout()
        self.cvs.draw()

    # -----------------------------------------------------------------------
    # Mouse interactions
    # -----------------------------------------------------------------------

    def _attach_interactions(self):
        for cid in self._cids:
            try: self.cvs.mpl_disconnect(cid)
            except Exception: pass
        self._cids = [
            self.cvs.mpl_connect('button_press_event',   self._on_press),
            self.cvs.mpl_connect('motion_notify_event',  self._on_motion),
            self.cvs.mpl_connect('button_release_event', self._on_release),
            self.cvs.mpl_connect('scroll_event',         self._on_scroll),
        ]

    def _on_press(self, event):
        # ── Click on y-axis label area → select that spectrum ────────────────
        # event.inaxes is None when clicking outside the axes (e.g. on the labels)
        # but event.x/y are always valid pixel coords.
        if event.inaxes is None and event.button == 1 and self._row_offsets:
            # Check if click is in the y-label strip: to the left of the axes
            ax_x0 = self.ax.get_position().x0 * self.fig.get_figwidth() * self.fig.dpi
            if event.x < ax_x0:
                # Convert pixel y to data y coordinate
                inv = self.ax.transData.inverted()
                _, data_y = inv.transform((event.x, event.y))
                # Find nearest row
                dists = [abs(data_y - ry) for ry in self._row_offsets]
                nearest_row = int(np.argmin(dists))
                if nearest_row < len(self._row_map):
                    series_idx = self._row_map[nearest_row]
                    self.table.blockSignals(True)
                    self.table.selectRow(series_idx)
                    self.table.blockSignals(False)
                    self._on_table_selection_changed()
            return

        if event.inaxes is not self.ax or event.xdata is None: return

        # Double-click → reset zoom
        if event.dblclick and event.button == 1:
            self._reset_zoom(); return

        if event.button == 1:
            self._drag_mode = 'zoom'
            self._drag_x0   = event.xdata
            self._drag_y0   = event.ydata
            ylim = self.ax.get_ylim()
            self._zoom_rect = Rectangle(
                (event.xdata, ylim[0]), 0, ylim[1] - ylim[0],
                linewidth=1, edgecolor='#3498DB', facecolor='#3498DB',
                alpha=0.15, zorder=10)
            self.ax.add_patch(self._zoom_rect)
            self.cvs.draw_idle()

    def _on_motion(self, event):
        if self._drag_mode != 'zoom' or self._zoom_rect is None: return
        if event.xdata is None: return
        x0, x1 = self._drag_x0, event.xdata
        lo, hi  = min(x0, x1), max(x0, x1)
        self._zoom_rect.set_x(lo)
        self._zoom_rect.set_width(hi - lo)
        self.cvs.draw_idle()

    def _on_release(self, event):
        if self._drag_mode != 'zoom':
            return
        if self._zoom_rect is not None:
            try: self._zoom_rect.remove()
            except Exception: pass
            self._zoom_rect = None

        if event.xdata is None or abs((event.xdata or 0) - (self._drag_x0 or 0)) < 0.01:
            # Short press with no drag — place peak marker
            if event.xdata is not None:
                self._select_peak(float(event.xdata))
        else:
            # Rubber-band zoom committed
            lo = min(self._drag_x0, event.xdata)
            hi = max(self._drag_x0, event.xdata)
            self._view_lo = hi   # NMR: high ppm on left
            self._view_hi = lo
            self.spb_zlo.blockSignals(True); self.spb_zhi.blockSignals(True)
            self.spb_zlo.setValue(round(self._view_lo, 2))
            self.spb_zhi.setValue(round(self._view_hi, 2))
            self.spb_zlo.blockSignals(False); self.spb_zhi.blockSignals(False)
            self._plot_stacked()

        self._drag_mode = None
        self._drag_x0   = None

    def _select_peak(self, ppm_val):
        self._peak_ppm = ppm_val
        self.spb_peak.setEnabled(True)
        self.spb_peak.setValue(ppm_val)
        self._plot_stacked()
        self._update_buttons()

    def _on_peak_spinbox_changed(self):
        self._peak_ppm = self.spb_peak.value()
        self._plot_stacked(); self._update_buttons()

    def _on_scroll(self, event):
        """Shift+scroll nudges the selected spectrum's ppm offset."""
        if event.inaxes is not self.ax: return
        from PyQt6.QtWidgets import QApplication
        mods = QApplication.keyboardModifiers()
        if not (mods & Qt.KeyboardModifier.ShiftModifier): return
        idx = self._selected_series_idx()
        if idx is None: return
        # Nudge step: 1% of current view width, min 0.001 ppm
        if self._view_lo is not None:
            step = abs(self._view_lo - self._view_hi) * 0.01
        else:
            step = 0.05
        step = max(step, 0.001)
        direction = 1 if event.button == 'up' else -1
        new_offset = self._offsets.get(idx, 0.0) + direction * step
        self._offsets[idx] = round(new_offset, 4)
        self.spb_offset.blockSignals(True)
        self.spb_offset.setValue(self._offsets[idx])
        self.spb_offset.blockSignals(False)
        data = self._series[idx]['data']
        self._cache = {k: v for k, v in self._cache.items()
                       if k[0] != id(data['spectrum'])}
        self._plot_stacked()

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def export_waterfall(self):
        if not self._series: return
        fmt_filters = "PNG image (*.png);;PDF file (*.pdf);;SVG file (*.svg);;All files (*)"
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        if not os.path.isdir(desktop):
            desktop = os.path.expanduser("~")
        default_path = os.path.join(desktop, "waterfall.png")
        filepath, sel = QFileDialog.getSaveFileName(
            self, "Export Waterfall", default_path, fmt_filters)
        if not filepath: return
        fmt = 'png'
        if sel.startswith('PDF'): fmt = 'pdf'
        elif sel.startswith('SVG'): fmt = 'svg'
        try:
            _export_waterfall_bw(self.fig, self.ax, filepath, fmt)
            self.cvs.draw_idle()
            QMessageBox.information(self, "Exported", f"Saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))