import markdown
from pathlib import Path

md_path = Path('artifacts/report_chapters.md')
html_path = Path('artifacts/report.html')

md_content = md_path.read_text(encoding='utf-8')

md_content = md_content.replace('*(Insert `artifacts/report_assets/system_flow.png` here as Figure 3.1: System architecture and execution flow)*', '<figure><img src="report_assets/system_flow.png" alt="System Flow"><figcaption>Figure 3.1: System architecture and execution flow</figcaption></figure>')
md_content = md_content.replace('*(Insert `artifacts/report_assets/test_metrics.png` here as Figure 4.1: Test accuracy and macro-F1 comparison)*', '<figure><img src="report_assets/test_metrics.png" alt="Test Metrics"><figcaption>Figure 4.1: Test accuracy and macro-F1 comparison</figcaption></figure>')
md_content = md_content.replace('*(Insert `artifacts/report_assets/winner_confusion.png` here as Figure 4.2: Confusion matrix for the recommended HSV histogram model)*', '<figure><img src="report_assets/winner_confusion.png" alt="Confusion Matrix" style="max-width: 500px;"><figcaption>Figure 4.2: Confusion matrix for the recommended HSV histogram model</figcaption></figure>')
md_content = md_content.replace('*(Insert `artifacts/report_assets/prototype_output.png` here as Figure 4.3: Prototype output generated from a held-out test image)*', '<figure><img src="report_assets/prototype_output.png" alt="Prototype Output"><figcaption>Figure 4.3: Prototype output generated from a held-out test image</figcaption></figure>')

html_body = markdown.markdown(md_content, extensions=['tables'])

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Banana Ripeness Assessment - Project Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px;
            background-color: #fcfcfc;
        }}
        h1 {{
            color: #173b2d;
            border-bottom: 2px solid #3f7d5c;
            padding-bottom: 10px;
            margin-top: 50px;
        }}
        h2 {{
            color: #245b45;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #245b45;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9f8;
        }}
        figure {{
            margin: 40px 0;
            text-align: center;
        }}
        img {{
            max-width: 100%;
            height: auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 4px;
            border: 1px solid #eee;
        }}
        figcaption {{
            margin-top: 10px;
            font-style: italic;
            color: #555;
            font-size: 0.9em;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: #ccc;
            margin: 40px 0;
        }}
        li {{
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>"""

html_path.write_text(html_template, encoding='utf-8')
print('HTML Report generated successfully!')
