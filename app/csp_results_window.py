"""
CSP Results Window
==================
Opens when the user clicks "Run CSP Analysis" in the main CSP window.
Shows Δδ vs concentration and the Kd determination plot side-by-side,
with a summary panel and CSV export.
"""

import os
import csv
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QGroupBox, QFileDialog,
    QMessageBox, QSizePolicy, QSpacerItem,
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class CSPResultsWindow(QMainWindow):
    """
    Standalone window showing the results of a CSP analysis run.
    Re-created and shown fresh each time the user clicks Run CSP Analysis.
    """

    def __init__(self, *, concs_uM, delta_delta, kd_result, centres,
                 labels, ref_idx, peak_ppm, parent_series):
        super().__init__()
        self.setWindowTitle("CSP Analysis Results")
        self.setGeometry(160, 160, 1050, 520)

        self._concs_uM    = concs_uM
        self._dd          = delta_delta
        self._kd          = kd_result
        self._centres     = centres
        self._labels      = labels
        self._ref_idx     = ref_idx
        self._peak_ppm    = peak_ppm
        self._series      = parent_series

        self._build_ui()
        self._populate()

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ── Left: summary + export ──────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(230)
        lv = QVBoxLayout(left)
        lv.setAlignment(Qt.AlignmentFlag.AlignTop)
        lv.setSpacing(8)

        # Summary box
        sum_grp = QGroupBox("Summary")
        sum_v   = QVBoxLayout(sum_grp)
        self.lbl_summary = QLabel("—")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet("font-size: 11px;")
        sum_v.addWidget(self.lbl_summary)
        lv.addWidget(sum_grp)

        # Peak centres table
        ctr_grp = QGroupBox("Fitted Peak Centres")
        ctr_v   = QVBoxLayout(ctr_grp)
        self.lbl_centres = QLabel("—")
        self.lbl_centres.setWordWrap(True)
        self.lbl_centres.setStyleSheet("font-size: 10px; font-family: monospace;")
        ctr_v.addWidget(self.lbl_centres)
        lv.addWidget(ctr_grp)

        # Export
        exp_grp = QGroupBox("Export")
        exp_v   = QVBoxLayout(exp_grp)
        b_csv = QPushButton("Export Results  (CSV)…")
        b_csv.setToolTip("Save Δδ table, Kd, and thermodynamic summary to CSV")
        b_csv.clicked.connect(self.export_csv)
        exp_v.addWidget(b_csv)

        b_fig = QPushButton("Export Plots  (PNG / PDF / SVG)…")
        b_fig.setToolTip("Save the Δδ and Kd plots as an image")
        b_fig.clicked.connect(self.export_plots)
        exp_v.addWidget(b_fig)
        lv.addWidget(exp_grp)

        lv.addItem(QSpacerItem(
            0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        root.addWidget(left)

        # ── Right: Δδ plot + Kd plot ────────────────────────────────────────
        self.fig = Figure(figsize=(8, 4.5), tight_layout=True)
        self.cvs = FigureCanvas(self.fig)
        self.ax_dd = self.fig.add_subplot(121)
        self.ax_kd = self.fig.add_subplot(122)
        root.addWidget(self.cvs)

    # -----------------------------------------------------------------------
    # Populate from analysis data
    # -----------------------------------------------------------------------

    def _populate(self):
        self._plot_dd()
        self._plot_kd()
        self._fill_summary()
        self._fill_centres()
        self.cvs.draw()

    def _plot_dd(self):
        ax = self.ax_dd
        ax.clear()
        concs = self._concs_uM
        dd    = self._dd
        ref   = self._ref_idx

        mask = np.ones(len(concs), dtype=bool)
        mask[ref] = False

        # Colour each point by whether it's above/below reference
        colors = ['#E74C3C' if d < 0 else '#2980B9' for d in dd[mask]]

        ax.scatter(concs[mask], dd[mask], c=colors, s=55, zorder=5)

        # Label each point with its concentration
        for x, y, lbl in zip(concs[mask], dd[mask],
                              [self._labels[i]
                               for i in range(len(concs)) if mask[i]]):
            ax.annotate(lbl, (x, y),
                        textcoords='offset points', xytext=(0, 6),
                        fontsize=7, ha='center', color='#555555')

        ax.axhline(0, color='#95A5A6', linewidth=0.8, linestyle='--')
        ax.set_xlabel("Concentration (µM)", fontsize=9)
        ax.set_ylabel("Δδ (ppm)", fontsize=9)
        ax.set_title(
            f"Chemical Shift Perturbation\npeak: {self._peak_ppm:.4f} ppm",
            fontsize=9)
        ax.tick_params(labelsize=8)

    def _plot_kd(self):
        ax = self.ax_kd
        ax.clear()
        r = self._kd

        # Show failure message for any error condition, including None values
        error_msg = r.get('error')
        kd = r.get('kd_uM')
        s  = r.get('slope')

        if error_msg is not None or kd is None or s is None or s == 0:
            reason = error_msg or "Kd could not be determined (insufficient data or zero slope)"
            ax.text(0.5, 0.5, f"Fit failed:\n{reason}",
                    transform=ax.transAxes, ha='center', va='center',
                    color='#C0392B', fontsize=9, wrap=True)
            ax.set_title("Kd Determination", fontsize=9)
            return

        inv_dd = r['inv_dd']
        conc_f = r['conc_fit']
        b      = r['intercept']
        r2     = r['r_squared']

        ax.scatter(inv_dd, conc_f, color='#2980B9', s=55, zorder=5,
                   label="Data")
        xl = np.linspace(min(inv_dd) * 1.3, max(inv_dd) * 1.1, 200)
        ax.plot(xl, s * xl + b, color='#2C3E50', linewidth=1.5,
                label=f"Fit  R² = {r2:.4f}")
        x_int = -b / s
        ax.axvline(x_int, color='#7F8C8D', linestyle='--',
                   linewidth=1.0, label="x-intercept = −Kd")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel("1/Δδ  (ppm⁻¹)", fontsize=9)
        ax.set_ylabel("[Ligand] (µM)", fontsize=9)
        ax.set_title(
            f"Kd Determination\nKd = {kd:.2f} µM  (R² = {r2:.4f})",
            fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    def _fill_summary(self):
        r = self._kd
        ref_lbl = self._labels[self._ref_idx] if self._ref_idx < len(self._labels) else "—"
        lines = [
            f"Peak  : {self._peak_ppm:.4f} ppm",
            f"Ref   : {ref_lbl}",
            f"N pts : {len(self._concs_uM)}",
            "",
        ]
        kd = r.get('kd_uM')
        if r.get('error') is None and kd is not None:
            lines += [
                f"Kd    = {kd:.2f} µM",
                f"R²    = {r['r_squared']:.4f}",
                f"Slope = {r['slope']:.4f}",
                f"Intercept = {r['intercept']:.4f}",
            ]
        else:
            reason = r.get('error') or "Kd could not be determined"
            lines += [f"Kd fit failed:\n{reason}"]
        self.lbl_summary.setText("\n".join(lines))

    def _fill_centres(self):
        lines = []
        for i, (lbl, c, ctr) in enumerate(
                zip(self._labels, self._concs_uM, self._centres)):
            ref_tag = " (ref)" if i == self._ref_idx else ""
            lines.append(f"{lbl}{ref_tag}\n  {ctr:.4f} ppm")
        self.lbl_centres.setText("\n".join(lines))

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def export_csv(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save CSP Results", "csp_results.csv",
            "CSV files (*.csv);;All files (*)")
        if not filepath:
            return
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            r  = self._kd
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(["CSP Binding Analysis — Export"])
                w.writerow(["Exported", ts])
                w.writerow(["Selected peak (ppm)", f"{self._peak_ppm:.4f}"])
                w.writerow(["Reference spectrum",
                             self._labels[self._ref_idx]])
                w.writerow([])

                w.writerow(["KD FIT RESULTS"])
                if r['error'] is None:
                    w.writerow(["Kd (µM)",     f"{r['kd_uM']:.4f}"])
                    w.writerow(["R²",           f"{r['r_squared']:.6f}"])
                    w.writerow(["Slope",        f"{r['slope']:.6f}"])
                    w.writerow(["Intercept",    f"{r['intercept']:.6f}"])
                else:
                    w.writerow(["Fit status",   "FAILED"])
                    w.writerow(["Error",        r['error']])
                w.writerow([])

                w.writerow(["RAW DATA"])
                w.writerow(["Label", "Conc (µM)", "Fitted centre (ppm)",
                             "Δδ (ppm)", "Ref?"])
                for i, (lbl, conc, ctr, dd) in enumerate(
                        zip(self._labels, self._concs_uM,
                            self._centres, self._dd)):
                    w.writerow([
                        lbl,
                        f"{conc:.4f}",
                        f"{ctr:.6f}",
                        f"{dd:.6f}",
                        "YES" if i == self._ref_idx else "no",
                    ])

            QMessageBox.information(self, "Exported",
                                    f"Results saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def export_plots(self):
        fmt_map = {
            "PNG image (*.png)": ("png", True),
            "PDF file (*.pdf)":  ("pdf", False),
            "SVG file (*.svg)":  ("svg", False),
        }
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Analysis Plots", "csp_analysis.png",
            ";;".join(fmt_map.keys()) + ";;All files (*)")
        if not filepath:
            return
        fmt, needs_dpi = fmt_map.get(selected_filter, ("png", True))
        try:
            kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
            if needs_dpi:
                kw['dpi'] = 300
            self.fig.savefig(filepath, format=fmt, **kw)
            QMessageBox.information(self, "Exported",
                                    f"Plots saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))