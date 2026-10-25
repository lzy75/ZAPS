"""Static report figure from submitted summaries; no image/tensor or model run.

Use bundled Python with reportlab; pdftoppm renders the single-page PDF for QA.
Axes differ between C and stronger comparators, and are explicitly labelled.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib import colors


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent / "experiment_cards" / "evidence"
TASKS = ["super_resolution", "gaussian_deblur", "inpainting", "motion_deblur"]
LABELS = ["SR", "Gaussian", "Inpainting", "Motion"]


def main():
    files = [EVIDENCE / "A46_user_reported_summary_20261025.json",
             EVIDENCE / "A48_user_reported_summary_20261026.json"]
    a46 = json.loads(files[0].read_text())["summary"]
    a48 = json.loads(files[1].read_text())
    groups46 = {r["task"]: r for r in a46["groups"]}
    groups48 = {r["task"]: r for r in a48["groups"]}
    series = [
        [groups46[t]["SELECTED_minus"]["C"]["mean"]["psnr"] for t in TASKS],
        [groups46[t]["SELECTED_minus"]["MATCHED_FIXED"]["mean"]["psnr"] for t in TASKS],
        [groups48[t]["TRUST_C_PSNR"] for t in TASKS],
        [groups48[t]["TRUST_LEGACY_PSNR"] for t in TASKS],
    ]
    canvas = Drawing(1000, 770)
    canvas.add(String(40, 735, "Gains over C do not establish a state-feedback increment",
                      fontName="Helvetica-Bold", fontSize=19))
    canvas.add(String(40, 710, "Submitted summary data; no new UNet calls. Read the different vertical scales.",
                      fontName="Helvetica", fontSize=11))
    panels = [
        (55, 425, "A46: SELECTED minus C", series[0], 0, 7, 1),
        (545, 425, "A46: SELECTED minus MATCHED_FIXED", series[1], -0.5, 0.1, 0.1),
        (55, 115, "A48: TRUST minus C", series[2], 0, 7, 1),
        (545, 115, "A48: TRUST minus LEGACY", series[3], -0.5, 0.1, 0.1),
    ]
    for x, y, title, values, lo, hi, step in panels:
        chart = VerticalBarChart()
        chart.x, chart.y, chart.width, chart.height = x + 30, y + 25, 380, 190
        chart.data = [values]
        chart.categoryAxis.categoryNames = LABELS
        chart.categoryAxis.labels.fontName = "Helvetica"
        chart.categoryAxis.labels.fontSize = 9
        chart.categoryAxis.visibleLabels = False
        chart.valueAxis.valueMin, chart.valueAxis.valueMax = lo, hi
        chart.valueAxis.valueStep = step
        chart.valueAxis.labels.fontSize = 8
        chart.valueAxis.gridStrokeColor = colors.HexColor("#e1e5e9")
        chart.valueAxis.visibleGrid = True
        chart.barWidth = 14
        chart.groupSpacing = 18
        chart.barLabelFormat = "%+.3f"
        chart.barLabels.fontName = "Helvetica"
        chart.barLabels.fontSize = 9
        chart.barLabels.nudge = 5
        for i, v in enumerate(values):
            chart.bars[0, i].fillColor = colors.HexColor("#247a68" if v > 0 else "#b65a49")
            chart.bars[0, i].strokeColor = None
        canvas.add(chart)
        for i, label in enumerate(LABELS):
            canvas.add(String(chart.x + (i + 0.5) * chart.width / len(LABELS),
                              chart.y - 14, label, textAnchor="middle",
                              fontName="Helvetica", fontSize=9))
        zero_y = chart.y + chart.height * (-lo) / (hi - lo)
        canvas.add(Line(chart.x, zero_y, chart.x + chart.width, zero_y,
                        strokeColor=colors.HexColor("#3d4650"), strokeWidth=0.8))
        canvas.add(String(x, y + 245, title, fontName="Helvetica-Bold", fontSize=13))
        canvas.add(String(x, y + 226, "Delta PSNR (dB)", fontName="Helvetica", fontSize=10))
    canvas.add(String(55, 395, "A46: 6 new image clusters; 24 cases. Matching candidates: 0/4 tasks.",
                      fontName="Helvetica", fontSize=11))
    canvas.add(String(55, 65, "A48: 5 reused development image clusters; 20 cases. TRUST = LOCAL; gate changed 0/20 actions.",
                      fontName="Helvetica", fontSize=11))
    canvas.add(String(55, 43, "PSNR alone does not establish the SSIM/LPIPS guards, generalization, covariance value or runtime benefit.",
                      fontName="Helvetica", fontSize=10))
    canvas.add(String(55, 24, "A46 submitted-table arithmetic rechecked; A48 compact user report. Original tensors not independently replayed.",
                      fontName="Helvetica", fontSize=10))
    base = ROOT / "evidence_comparators"
    renderPDF.drawToFile(canvas, str(base.with_suffix(".pdf")))
    renderSVG.drawToFile(canvas, str(base.with_suffix(".svg")))
    executable = shutil.which("pdftoppm")
    if executable:
        subprocess.run([executable, "-r", "120", "-singlefile", "-png",
                        str(base.with_suffix(".pdf")), str(base)], check=True)
    meta = {"source_files": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                             for p in files], "task_order": TASKS, "series": series,
            "method_nfe": 0, "diagnostic_nfe": 0,
            "limits": "Summary-derived visualization only;distinct axes and development/confirmation scopes retained."}
    base.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(base)


if __name__ == "__main__":
    main()
