import pandas as pd

def load_slot_points(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["label", "x", "y", "filename", "img_width", "img_height"]

    slot_points = {}
    for _, row in df.iterrows():
        img = row['filename']
        if img not in slot_points:
            slot_points[img] = []
        slot_points[img].append({
            'slot_id': f"{row['label']}_{_}",  # Unique ID
            'x': int(row['x']),
            'y': int(row['y'])
        })
    return slot_points
