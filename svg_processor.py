import os
import xml.etree.ElementTree as ET
import random
import string

def generate_unique_id():
    """Generate a short random hash for IDs."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

def extract_stroke_color(style):
    """Extracts the stroke color from a style attribute string."""
    if 'stroke:' in style:
        parts = style.split(';')
        stroke_entry = [part for part in parts if 'stroke:' in part]
        if stroke_entry:
            return stroke_entry[0].split(':')[1].strip()
    return None

def process_svg(svg_path):
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    ET.register_namespace('', ns['svg'])
    tree = ET.parse(svg_path)
    root = tree.getroot()

    axes_group = root.find(".//*[@id='axes_1']", namespaces=ns)
    legend_group = root.find(".//*[@id='legend_1']", namespaces=ns)

    if not axes_group or not legend_group:
        raise ValueError("SVG does not contain the required 'axes_1' or 'legend_1' groups")
    
    legend_items = list(legend_group)[1:]  # Skip the first element if it's a background patch
    for i in range(0, len(legend_items), 2):
        if i+1 >= len(legend_items):
            continue
        sample_g = legend_items[i]
        label_g = legend_items[i+1]
        
        bounding_g = ET.SubElement(legend_group, 'g', {'id': generate_unique_id(), 'class': 'legend_item'})
        legend_group.remove(sample_g)
        legend_group.remove(label_g)
        bounding_g.append(sample_g)
        bounding_g.append(label_g)
    
    for bounding_g in legend_group.findall('g', namespaces=ns):
        sample_g = bounding_g[0]
        path = sample_g.find('.//svg:path', namespaces=ns)
        if path is None:
            continue
        style = path.get('style')
        if style:
            stroke_color = extract_stroke_color(style)
            if stroke_color:
                for child in axes_group:
                    path = child.find('.//svg:path', namespaces=ns)
                    if path is not None and extract_stroke_color(path.get('style')) == stroke_color:
                        child.set('id', bounding_g.get('id'))
                        child.set('class', 'plot_line')
                        break
    directory, filename = os.path.split(svg_path)
    new_filename = 'p_' + filename
    new_filepath = os.path.join(directory, new_filename)
    tree.write(new_filepath)

# Example usage
# process_svg('example.svg')
