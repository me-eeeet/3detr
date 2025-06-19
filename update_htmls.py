import argparse
from pathlib import Path

def generate_nav_js(html_files):
    """Generate the content for nav.js based on the list of HTML files."""
    return f"""
const files = {html_files};
const currentFile = location.pathname.split('/').pop();
const currentIndex = files.indexOf(currentFile);

function goNext() {{
if (currentIndex < files.length - 1) {{
    location.href = files[currentIndex + 1];
}}
}}

function goPrev() {{
if (currentIndex > 0) {{
    location.href = files[currentIndex - 1];
}}
}}
""".strip()

def inject_navbar(html_path, nav_bar_html):
    """Inject the navigation bar at the top of the HTML body."""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'nav.js' in content:
        print(f"[SKIP] {html_path.name} already contains nav.js")
        return

    # Insert nav bar after <body> tag if present, else at top of file
    if '<body>' in content:
        content = content.replace('<body>', f'<body>\n{nav_bar_html}\n', 1)
    else:
        content = nav_bar_html + '\n' + content

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] Injected nav bar into {html_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Inject nav bar and JS into local HTML files.")
    parser.add_argument("directory", type=str, help="Path to directory containing HTML files")
    args = parser.parse_args()

    html_dir = Path(args.directory)

    if not html_dir.is_dir():
        print(f"[ERROR] {html_dir} is not a valid directory.")
        return

    html_files = sorted([f.name for f in html_dir.glob("*.html")])
    if not html_files:
        print(f"[INFO] No HTML files found in {html_dir}")
        return

    # Write nav.js
    nav_js_path = html_dir / "nav.js"
    nav_js_content = generate_nav_js(html_files)
    nav_js_path.write_text(nav_js_content, encoding='utf-8')
    print("[OK] Created nav.js")

    # Nav bar HTML to inject
    nav_bar_html = '''
<div style="margin: 30px; text-align: center;">
  <button onclick="goPrev()" style="font-size: 20px; padding: 12px 24px; background-color: white; color: black; border: 2px solid #000; border-radius: 8px; margin-right: 10px;">
    ⬅ Previous
  </button>
  <button onclick="goNext()" style="font-size: 20px; padding: 12px 24px; background-color: white; color: black; border: 2px solid #000; border-radius: 8px;">
    Next ➡
  </button>
</div>
<script src="nav.js"></script>
'''.strip()

    # Inject nav bar into each HTML file
    for file_name in html_files:
        inject_navbar(html_dir / file_name, nav_bar_html)

if __name__ == "__main__":
    main()
