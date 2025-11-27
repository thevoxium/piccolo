import os
import shutil
import markdown
import frontmatter
from jinja2 import Environment, FileSystemLoader

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENT_DIR = os.path.join(BASE_DIR, 'content')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
OUTPUT_DIR = os.path.join(BASE_DIR, 'public')

# Setup Jinja2
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

def clean_output():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

def build_structure():
    """
    Scans content dir and builds a dictionary of pages for navigation.
    """
    site_map = {}
    
    for root, dirs, files in os.walk(CONTENT_DIR):
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                
                # Parse Frontmatter
                post = frontmatter.load(filepath)
                
                # Determine relative path for URL
                rel_path = os.path.relpath(filepath, CONTENT_DIR)
                rel_dir = os.path.dirname(rel_path)
                filename = os.path.basename(rel_path).replace('.md', '.html')
                
                # Category (e.g., 'docs', 'tutorials', or 'root')
                category = rel_dir if rel_dir else 'root'
                
                if category not in site_map:
                    site_map[category] = []
                
                # Add to map
                site_map[category].append({
                    'title': post.get('title', 'Untitled'),
                    'order': post.get('order', 99),
                    'url': filename if category == 'root' else f"{category}/{filename}",
                    'content': markdown.markdown(post.content, extensions=['fenced_code', 'tables']),
                    'active': False 
                })

    # Sort pages by 'order' metadata
    for cat in site_map:
        site_map[cat].sort(key=lambda x: x['order'])
        
    return site_map

def generate_html(site_map):
    template = env.get_template('doc.html')
    
    # Flatten map to iterate all pages
    all_pages = []
    for cat, pages in site_map.items():
        for page in pages:
            page['category'] = cat
            all_pages.append(page)

    for page in all_pages:
        # Determine Output Path
        out_path = os.path.join(OUTPUT_DIR, page['url'])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Calculate root_path for relative CSS links (e.g. "../../")
        depth = page['url'].count('/')
        root_path = '../' * depth if depth > 0 else './'

        # Mark current page as active for sidebar styling
        # We create a deep copy of nav for this render to safely modify 'active' state
        render_nav = {k: [p.copy() for p in v] for k, v in site_map.items()}
        
        # Find current page in render_nav and mark active
        if page['category'] in render_nav:
            for p in render_nav[page['category']]:
                if p['url'] == page['url']:
                    p['active'] = True

        # Render
        with open(out_path, 'w') as f:
            f.write(template.render(
                page=page,
                nav=render_nav,
                root_path=root_path,
                is_landing=(page['url'] == 'index.html')
            ))
            
    print(f"🚀 Site generated in /{OUTPUT_DIR}")

def copy_landing_page():
    """
    Copies the custom index.html from the website directory to the output directory.
    """
    landing_page_source = os.path.join(BASE_DIR, 'landing_page.html')
    if os.path.exists(landing_page_source):
        shutil.copy(landing_page_source, os.path.join(OUTPUT_DIR, 'index.html'))
        print("📄 Copied custom landing page (website/landing_page.html)")
    else:
        print("⚠️ Warning: website/landing_page.html not found")

if __name__ == "__main__":
    clean_output()
    site_map = build_structure()
    generate_html(site_map)
    copy_landing_page()
