import re
from markdown_pdf import Section, MarkdownPdf

# Read the current markdown
with open('SignVision_Refined.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace horizontal rules before H2s with explicit CSS page breaks
text = re.sub(r'\n---\n\n## ', r'\n<div style="page-break-before: always;"></div>\n\n## ', text)

# Write back
with open('SignVision_Refined_Paginated.md', 'w', encoding='utf-8') as f:
    f.write(text)

# Generate PDF
pdf = MarkdownPdf(toc_level=2)
pdf.add_section(Section(text))
pdf.save("SignVision_Final_Report_V3.pdf")
print("Paginated PDF Generated Successfully.")
