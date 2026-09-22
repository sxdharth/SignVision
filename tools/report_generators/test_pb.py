from markdown_pdf import Section, MarkdownPdf

test_md = """
# Page 1
Some content here.
<div style="page-break-before: always;"></div>
# Page 2
This should be on page 2.
<hr style="page-break-before: always;">
# Page 3
This should be on page 3.
"""

pdf = MarkdownPdf()
pdf.add_section(Section(test_md))
pdf.save("test_page_break.pdf")
print("Test PDF saved.")
