import cv2
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Rectangle size for each parking spot
width, height = 58, 20

# Load existing parking positions if available
try:
    with open('slots_data\parking_slots', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

# Load and prepare image
img = cv2.imread('images\parking_lot.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Function to handle clicks
def on_click(event):
    global posList
    if event.button == 1:  # Left click to add
        x, y = int(event.xdata), int(event.ydata)
        posList.append((x, y))
        print(f"Added: ({x}, {y})")
    elif event.button == 3:  # Right click to remove
        x, y = int(event.xdata), int(event.ydata)
        for i, (x1, y1) in enumerate(posList):
            if x1 < x < x1 + width and y1 < y < y1 + height:
                print(f"Removed: ({x1}, {y1})")
                posList.pop(i)
                break
    redraw()

# Draw rectangles and refresh display
def redraw():
    ax.clear()
    ax.imshow(img_rgb)
    for pos in posList:
        rect = plt.Rectangle(pos, width, height, linewidth=2, edgecolor='magenta', facecolor='none')
        ax.add_patch(rect)
    plt.title("Left-click: Add, Right-click: Remove, Close window to save")
    fig.canvas.draw()

# Create plot
fig, ax = plt.subplots()
redraw()

# Connect mouse click event
cid = fig.canvas.mpl_connect('button_press_event', on_click)

# Show image and interact
plt.show()

# Save when done
with open('slots_data\parking_slots', 'wb') as f:
    pickle.dump(posList, f)

# Safe terminal output
print(f"\nSaved {len(posList)} parking positions to 'parking_slots'")

with open('slots_data\parking_slots', 'rb') as f:
    posList = pickle.load(f)

slot_rows = []
for idx, (x, y) in enumerate(posList, start=1):
    slot_rows.append({'id': idx, 'x': x, 'y': y, 'w': width, 'h': height})  # default size

df = pd.DataFrame(slot_rows)
df.to_csv("slots_data\slots.csv", index=False)
print("Converted parking_slots to slots.csv")