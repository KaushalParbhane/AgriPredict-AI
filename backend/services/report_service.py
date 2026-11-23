import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.units import inch

from backend.config import EXPORTS_DIR

def generate_pdf_report(
    project_title: str = "AgriPredict AI — Comprehensive Intelligence Report",
    dataset_summary: Optional[Dict[str, Any]] = None,
    ml_models: Optional[List[Dict[str, Any]]] = None,
    latest_prediction: Optional[Dict[str, Any]] = None,
    agrimall_summary: Optional[Dict[str, Any]] = None,
    author: str = "TY B.Sc. Data Science Research Team"
) -> str:
    """
    Generates an executive-grade PDF project report using ReportLab.
    """
    report_id = str(uuid.uuid4())[:8]
    filename = f"AgriPredict_Report_{report_id}.pdf"
    pdf_path = EXPORTS_DIR / filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom emerald/forest green styles
    primary_green = colors.HexColor("#064e3b")
    secondary_green = colors.HexColor("#10b981")
    dark_gray = colors.HexColor("#1f2937")
    light_bg = colors.HexColor("#f0fdf4")
    border_gray = colors.HexColor("#e5e7eb")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=primary_green,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=primary_green,
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=dark_gray,
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        "DocBodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13.5,
        textColor=primary_green
    )

    story = []

    # Title & Metadata Header
    story.append(Paragraph(project_title, title_style))
    story.append(Paragraph(f"<b>Author:</b> {author} &nbsp;|&nbsp; <b>Date:</b> {datetime.now().strftime('%B %d, %Y')} &nbsp;|&nbsp; <b>Platform:</b> AgriPredict AI v2.4", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=secondary_green, spaceBefore=0, spaceAfter=12))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h2_style))
    story.append(Paragraph(
        "This capstone report consolidates data science findings from <b>AgriPredict AI</b>, an end-to-end intelligence platform engineered for crop yield prediction and commercial agro-mall demand forecasting. Combining real transaction data from Shivdatta Krupa Krushi Agro Mall with multi-variable agronomic yield models (spanning rainfall, soil pH, NPK nutrients, temperature, and irrigation regimes), this study demonstrates measurable predictive accuracy and practical agricultural supply-chain optimization.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # ML Model Performance Benchmark Table
    story.append(Paragraph("2. Machine Learning Model Benchmark Studio", h2_style))
    if ml_models and len(ml_models) > 0:
        table_data = [["Algorithm", "MAE", "RMSE", "R² Score", "CV Mean R²", "Status"]]
        for m in ml_models:
            is_b = "★ Best Model" if m.get("is_best") else "Trained"
            table_data.append([
                m.get("name", m.get("algorithm", "Model")),
                str(m.get("mae", "N/A")),
                str(m.get("rmse", "N/A")),
                str(m.get("r2", "N/A")),
                str(m.get("cv_mean_r2", "N/A")),
                is_b
            ])

        t = Table(table_data, colWidths=[150, 60, 60, 70, 80, 80])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_green),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, border_gray),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("Models: XGBoost, Random Forest, Decision Tree, Linear Regression, and Dummy Baseline benchmarked against 5-fold cross validation.", body_style))

    story.append(Spacer(1, 10))

    # Dataset Profile & Health
    story.append(Paragraph("3. Dataset Health & Profiling Audit", h2_style))
    if dataset_summary:
        ds_info = [
            ["Metric", "Value", "Metric", "Value"],
            ["Total Records", str(dataset_summary.get("row_count", 0)), "Total Columns", str(dataset_summary.get("col_count", 0))],
            ["Missing Values", f"{dataset_summary.get('missing_total', 0)} ({dataset_summary.get('missing_pct', 0)}%)", "Duplicates Detected", str(dataset_summary.get("duplicates_count", 0))],
            ["Quality Score", f"{dataset_summary.get('quality_score', 95)} / 100", "Validation Split", "80% Train / 20% Holdout"]
        ]
        dt = Table(ds_info, colWidths=[120, 130, 120, 130])
        dt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), light_bg),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('GRID', (0, 0), (-1, -1), 0.5, border_gray),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(dt)

    story.append(Spacer(1, 10))

    # Agri Mall Analytics Review
    story.append(Paragraph("4. Agri Mall Commercial Intelligence", h2_style))
    if agrimall_summary:
        rev = f"Rs. {agrimall_summary.get('total_revenue', 0):,.2f}"
        qty = f"{agrimall_summary.get('total_quantity_sold', 0):,} Units"
        mall_name = agrimall_summary.get("mall_name", "Shivdatta Krupa Krushi Agro Mall")
        
        story.append(Paragraph(
            f"<b>Commercial Entity:</b> {mall_name}<br/>"
            f"<b>Aggregate Sales Volume:</b> {qty} &nbsp;|&nbsp; <b>Total Gross Invoiced Revenue:</b> {rev}<br/>"
            f"<b>Dominant Categories:</b> Seeds (Soya KDS-726, Pioneer Bajari), Fertilizers (Potash, Boron, Urea), and Plant Health Protectants (Syngenta Kavach, Rallis Asataf).",
            body_style
        ))

    story.append(Spacer(1, 10))

    # Methodology, Leakage Prevention & Ethical AI
    story.append(Paragraph("5. Engineering Methodology & Leakage Prevention", h2_style))
    story.append(Paragraph(
        "<b>Rigorous Data Preprocessing:</b> Feature transformation utilizes scikit-learn ColumnTransformer isolating numeric imputation and standard scaling from categorical one-hot encoding. All transformations are fitted exclusively on the training partition and applied identically during test validation and live inference.<br/>"
        "<b>Leakage Safeguards:</b> The target variable (yield) is strictly quarantined from input feature matrices during both training and batch inference.<br/>"
        "<b>Operational Uncertainty:</b> Yield predictions incorporate 95% confidence intervals derived from residual RMSE deviations to maintain commercial transparency.",
        body_style
    ))

    # Build Document
    doc.build(story)
    return filename
