import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def create_grading_policy_2023(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("University Grading Policy (Fall 2023)", styles['Title']))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("1. Grading Scale", styles['Heading1']))
    elements.append(Paragraph("The standard grading scale.", styles['Normal']))
    
    data = [
        ['Letter Grade', 'GPA Points'],
        ['A', '4.0'],
        ['B', '3.0'],
        ['C', '2.0'],
        ['D', '1.0'],
        ['F', '0.0']
    ]
    t = Table(data)
    t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    elements.append(t)
    
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("2. Incomplete Grades", styles['Heading1']))
    elements.append(Paragraph("An incomplete grade (I) requires completing at least 75% of the coursework.", styles['Normal']))

    doc.build(elements)

def create_grading_policy_update_2026(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("University Grading Policy UPDATE (Spring 2026)", styles['Title']))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("1. Revised Grading Scale", styles['Heading1']))
    elements.append(Paragraph("Due to grade inflation adjustments, effective Spring 2026, the GPA points for a 'C' grade have been updated.", styles['Normal']))
    
    data = [
        ['Letter Grade', 'GPA Points'],
        ['A', '4.0'],
        ['B', '3.0'],
        ['C', '2.5'],
        ['D', '1.0'],
        ['F', '0.0']
    ]
    t = Table(data)
    t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    elements.append(t)
    doc.build(elements)

if __name__ == "__main__":
    os.makedirs("data/raw_pdfs", exist_ok=True)
    # Remove old file if it exists
    if os.path.exists("data/raw_pdfs/grading_policy.pdf"):
        os.remove("data/raw_pdfs/grading_policy.pdf")
    
    create_grading_policy_2023("data/raw_pdfs/grading_policy_2023.pdf")
    create_grading_policy_update_2026("data/raw_pdfs/grading_policy_2026_update.pdf")
    print("Mock PDFs generated successfully in data/raw_pdfs/")
