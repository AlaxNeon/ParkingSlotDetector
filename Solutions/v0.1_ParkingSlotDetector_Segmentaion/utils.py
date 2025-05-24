import pandas as pd

def load_slot_points(csv_path):
    df = pd.read_csv(csv_path)
    slot_points = {}
    for _, row in df.iterrows():
        fname = row['filename']
        if fname not in slot_points:
            slot_points[fname] = []
        slot_points[fname].append({
            "slot_id": row["slot_id"],
            "x": int(row["x"]),
            "y": int(row["y"]),
            "visibility": row["visibility"].lower()
        })
    return slot_points
