from markdown_pdf import Section, MarkdownPdf

pdf = MarkdownPdf(toc_level=2)
pdf.add_section(Section(open("SignVision_Explanation.md").read()))
pdf.save("SignVision_Explanation.pdf")
print("PDF generated successfully.")
