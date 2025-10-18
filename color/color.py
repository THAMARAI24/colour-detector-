import cv2
import pandas as pd
import numpy as np
from io import StringIO

# Embedded CSV data inside the code
csv_data = """color_name,hex,R,G,B
Black,#000000,0,0,0
White,#FFFFFF,255,255,255
Red,#FF0000,255,0,0
Lime,#00FF00,0,255,0
Blue,#0000FF,0,0,255
Yellow,#FFFF00,255,255,0
Cyan,#00FFFF,0,255,255
Magenta,#FF00FF,255,0,255
Gray,#808080,128,128,128
Maroon,#800000,128,0,0
Olive,#808000,128,128,0
Green,#008000,0,128,0
Purple,#800080,128,0,128
Teal,#008080,0,128,128
Navy,#000080,0,0,128
Orange,#FFA500,255,165,0
Pink,#FFC0CB,255,192,203
Brown,#A52A2A,165,42,42
Gold,#FFD700,255,215,0
Silver,#C0C0C0,192,192,192
"""

# Read the CSV data using pandas
color_data = pd.read_csv(StringIO(csv_data))

# Function to get the closest color name based on RGB values
def get_color_name(R, G, B):
    # Kept for backward compatibility but now uses perceptual Lab distance
    name, _ = get_closest_color(R, G, B)
    return name


def get_closest_color(R, G, B):
    """Return (name, hex) of the closest color in the embedded table."""
    # Use CIE Lab distance for perceptual closeness
    try:
        sample_lab = rgb_to_lab(R, G, B)
        labs = color_data[['L', 'a', 'b']].to_numpy(dtype=float)
        # Euclidean distance in Lab
        diffs = labs - sample_lab.reshape(1, 3)
        dists = np.linalg.norm(diffs, axis=1)
        best_idx = int(np.argmin(dists))
        row = color_data.loc[best_idx]
        return row["color_name"], row["hex"]
    except Exception:
        # Fallback to simple RGB absolute distance
        minimum = float('inf')
        best_idx = 0
        for i in range(len(color_data)):
            d = abs(R - int(color_data.loc[i, "R"])) + abs(G - int(color_data.loc[i, "G"])) + abs(B - int(color_data.loc[i, "B"]))
            if d < minimum:
                minimum = d
                best_idx = i
        row = color_data.loc[best_idx]
        return row["color_name"], row["hex"]


def rgb_to_lab(R, G, B):
    """Convert an RGB triple (0-255 ints) to OpenCV Lab (L,a,b) as floats."""
    arr = np.uint8([[[B, G, R]]])
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return lab


def _precompute_table_lab():
    """Precompute Lab values for the embedded color table."""
    rgb = color_data[['R', 'G', 'B']].to_numpy(dtype=np.uint8)
    # Convert to BGR and reshape for cv2
    bgr = rgb[:, ::-1].reshape(-1, 1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    color_data['L'] = lab[:, 0]
    color_data['a'] = lab[:, 1]
    color_data['b'] = lab[:, 2]


# Precompute Lab table at import
_precompute_table_lab()

# Variables for mouse event
clicked = False
r = g = b = xpos = ypos = 0
frame = None
center_sample = False

def draw_function(event, x, y, flags, param):
    global b, g, r, xpos, ypos, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        xpos = x
        ypos = y
        # 'frame' is a module-level variable that will be updated in the main loop
        try:
            # Ensure coordinates are within current frame bounds
            h, w = frame.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                b, g, r = frame[y, x]
            else:
                # Out of bounds click - ignore
                clicked = False
                return
        except Exception:
            # If frame is None or something else goes wrong, ignore the click
            clicked = False
            return
        b = int(b)
        g = int(g)
        r = int(r)

# Start webcam
def main():
    global frame, clicked, center_sample, b, g, r, xpos, ypos

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Color Detector")
    cv2.setMouseCallback("Color Detector", draw_function)

    print("🎨 Color Detector started! Click anywhere on the video window to detect color.")
    print("Press 'q' to quit. Space or 's' to sample center. 'c' to clear sample. 'a' to toggle auto-center-sample.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not found or not working.")
            break

        # Draw a center crosshair (helps when sampling center)
        fh, fw = frame.shape[:2]
        cx, cy = fw // 2, fh // 2
        cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 1)
        cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 1)

        if clicked:
            # Draw rectangle with detected color (fit to frame width)
            h, w = frame.shape[:2]
            rect_right = min(600, w - 20)
            cv2.rectangle(frame, (20, 20), (rect_right, 60), (b, g, r), -1)

            color_name, color_hex = get_closest_color(r, g, b)
            text = f"{color_name} {color_hex}  R={r} G={g} B={b}"

            # Draw a visible marker at the clicked location
            cv2.circle(frame, (xpos, ypos), 10, (255, 255, 255), 2)

            # Choose text color based on brightness
            if r + g + b >= 600:
                text_color = (0, 0, 0)
            else:
                text_color = (255, 255, 255)

            cv2.putText(frame, text, (30, 50), 2, 0.8, text_color, 2, cv2.LINE_AA)

        # Auto-sample center if requested
        if center_sample and frame is not None:
            # sample center pixel
            try:
                pb, pg, pr = frame[cy, cx]
                b, g, r = int(pb), int(pg), int(pr)
                xpos, ypos = cx, cy
                clicked = True
            except Exception:
                pass

        cv2.imshow("Color Detector", frame)

        # Key handling: q quit, space or s sample center, c clear, a toggle auto-center
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') or key == ord('s'):
            # Sample center pixel immediately
            try:
                pb, pg, pr = frame[cy, cx]
                b, g, r = int(pb), int(pg), int(pr)
                xpos, ypos = cx, cy
                clicked = True
            except Exception:
                pass
        elif key == ord('c'):
            clicked = False
        elif key == ord('a'):
            center_sample = not center_sample

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Color Detector closed successfully.")


if __name__ == '__main__':
    main()
