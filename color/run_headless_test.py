from color import get_color_name

# Test samples: (R, G, B) -> expected closest name from embedded CSV
samples = [
    (0, 0, 0),       # Black
    (255, 255, 255), # White
    (254, 0, 0),     # Red
    (0, 254, 0),     # Lime/Green
    (0, 0, 254),     # Blue
    (255, 165, 0),   # Orange
    (128, 128, 0),   # Olive
    (192, 192, 192), # Silver
]

for r, g, b in samples:
    name = get_color_name(r, g, b)
    print(f"R={r} G={g} B={b} -> {name}")
