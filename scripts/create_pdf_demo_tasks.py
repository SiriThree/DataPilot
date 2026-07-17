from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def create_task_902() -> None:
    path = Path("data/public/input/task_902/context/policy_report.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    story = [
        Paragraph("Quarterly Operating Policy Report", styles["Title"]),
        Spacer(1, 12),
    ]
    for index in range(1, 12):
        story.append(Paragraph(f"Section {index}: Routine operating note", styles["Heading2"]))
        story.append(
            Paragraph(
                "This section records routine store observations, staffing updates, "
                "and non-critical merchandising notes for regional managers.",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 8))
        story.append(PageBreak())

    story.append(Paragraph("Risk Appendix", styles["Heading2"]))
    story.append(
        Paragraph(
            "Electronics demand risk: Supplier delays created demand risk for Electronics.",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "Furniture demand risk: No material campaign risk was recorded.",
            styles["BodyText"],
        )
    )
    doc.build(story)


def create_task_903() -> None:
    path = Path("data/public/input/task_903/context/table_report.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    table = Table(
        [
            ["Category", "Q1 Sales", "Q2 Sales"],
            ["Electronics", "1200", "900"],
            ["Furniture", "800", "760"],
            ["Toys", "500", "650"],
        ]
    )
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, "black"),
    ]))
    story = [Paragraph("PDF Sales Table Report", getSampleStyleSheet()["Title"]), Spacer(1, 12), table]
    doc.build(story)


if __name__ == "__main__":
    create_task_902()
    create_task_903()
    print("created task_902 and task_903 PDFs")
