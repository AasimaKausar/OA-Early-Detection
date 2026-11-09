
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
)


def generate_oa_report(image_path, filename, grade, confidence):
    """Generate a downloadable PDF report for an OA prediction."""

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#174A7E"),
        fontSize=20,
        spaceAfter=8,
    ))

    story = []

    story.append(Paragraph("OA Vision AI", styles["ReportTitle"]))
    story.append(Paragraph(
        "AI-Assisted Osteoarthritis Image Analysis Report",
        styles["Normal"],
    ))
    story.append(Spacer(1, 18))

    story.append(Paragraph("Report Information", styles["Heading2"]))

    information = [
        ["Report Date", datetime.now().strftime("%d-%m-%Y %H:%M")],
        ["Image Filename", str(filename)],
        ["Predicted KL Grade", f"Grade {grade}"],
        ["Model Confidence", f"{float(confidence):.2f}%"],
        ["Analysis Status", "Completed"],
    ]

    table = Table(information, colWidths=[150, 300])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#222222")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story.append(table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Analyzed Knee X-ray", styles["Heading2"]))

    try:
        xray = PDFImage(image_path, width=3.5 * inch, height=3.5 * inch)
        xray.hAlign = "CENTER"
        xray._restrictSize(3.5 * inch, 3.5 * inch)
        story.append(xray)
    except Exception:
        story.append(Paragraph(
            "The X-ray image could not be embedded in this report.",
            styles["Normal"],
        ))

    story.append(Spacer(1, 16))
    story.append(Paragraph("KL Grade Reference", styles["Heading2"]))

    grade_descriptions = {
        0: "No radiographic features of osteoarthritis detected.",
        1: "Doubtful joint-space narrowing or possible osteophytes.",
        2: "Possible osteophytes and possible joint-space narrowing.",
        3: "Multiple osteophytes, definite joint-space narrowing, "
           "and possible bone changes.",
        4: "Large osteophytes, marked joint-space narrowing, "
           "and severe bone changes.",
    }

    description = grade_descriptions.get(
        int(grade),
        "No grade description available.",
    )

    story.append(Paragraph(description, styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Important Disclaimer", styles["Heading2"]))

    story.append(Paragraph(
        "This report contains an AI-generated prediction from a research "
        "prototype. It is not a confirmed medical diagnosis. Model "
        "confidence is not the same as clinical certainty. Please consult "
        "a qualified healthcare professional for medical interpretation.",
        styles["Normal"],
    ))

    document.build(story)

    buffer.seek(0)
    return buffer
