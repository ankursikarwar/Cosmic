# Define consistent colors for each object category
CATEGORY_COLORS = {
    # Furniture - Browns and warm tones
    "Bed": "#4169E1",           # Royal Blue (Was Saddle Brown)
    "Desk": "#D2691E",          # Chocolate
    "Chair": "#CD853F",         # Peru
    "Table": "#DEB887",         # Burlywood
    "Table Dining": "#BC8F8F",  # Rosy Brown
    "Sofa": "#FF6347",          # Tomato (Was Sienna)
    "Cabinet": "#8B7355",       # Burlywood4
    "Dresser": "#9B6B4C",       # Medium Brown
    "Bookshelf": "#704214",     # Dark Brown
    "Shelf": "#8B4513",         # Saddle Brown (Was Indigo)
    "TV Stand": "#654321",      # Dark Brown
    "Coffee Table": "#C19A6B",  # Camel
    "Side Table": "#B5936B",    # Light Brown
    "Single Cabinet": "#967969", # Light Wood
    
    # Lighting - Yellows and warm lights
    "Lamp": "#FF0000",          # Red (Was Magenta)
    "Ceiling Lamp": "#FFA500",  # Orange
    "Floor Lamp": "#FFCC00",    # Bright Yellow
    "Pendant Lamp": "#FFB347",  # Pastel Orange
    
    # Plants - Greens
    "Plant Container": "#228B22", # Forest Green
    "Plant": "#32CD32",         # Lime Green
    
    # Decor - Various
    "Wall Art": "#9370DB",      # Medium Purple
    "Picture": "#8A2BE2",       # Blue Violet
    "Mirror": "#B0C4DE",        # Light Steel Blue
    "Rug": "#DC143C",           # Crimson
    "Curtain": "#F0E68C",       # Khaki
    
    # Architectural - Grays and neutrals
    "Window": "#87CEEB",        # Sky Blue
    "Door": "#B8860B",          # Dark Goldenrod (Was Maroon)
    "Wall": "#DCDCDC",          # Gainsboro
    
    # Kitchen Appliances - Metallics and whites
    "Sink": "#C0C0C0",          # Silver
    "Oven": "#2F4F4F",          # Dark Slate Gray
    "Fridge": "#4682B4",        # Steel Blue (Was White Smoke)
    "Refrigerator": "#4682B4",  # Steel Blue (Was Light Gray)
    "Microwave": "#A9A9A9",     # Dark Gray
    "Dishwasher": "#D3D3D3",    # Light Gray
    "Stove": "#4A4A4A",         # Charcoal
    
    # Bathroom - Blues and whites
    "Toilet": "#333333",        # Dark Charcoal (Was Alice Blue)
    "Bathtub": "#40E0D0",       # Turquoise (Was Light Cyan)
    "Shower": "#B0E0E6",        # Powder Blue
    
    # Electronics - Dark tones
    "TV": "#191970",            # Midnight Blue
    "Monitor": "#000080",       # Navy
    "Computer": "#1C1C1C",      # Dark Gray
    
    # Storage
    "Wardrobe": "#8B4726",      # Saddle Brown 3
    "Drawer": "#A0826D",        # Beaver
    "Storage Box": "#CD9B6D",   # Tan
    
    # Bedding
    "Pillow": "#FFF0F5",        # Lavender Blush
    "Blanket": "#FFE4E1",       # Misty Rose
    
    # Default fallback color for unlisted categories
    "_default_": "#808080"      # Gray
}

# Fallback pastel colors for any categories not in the mapping
pastel_colors = [
    "#D32F2F", "#C2185B", "#7B1FA2", "#512DA8", "#303F9F",
    "#1976D2", "#0097A7", "#00796B", "#388E3C", "#689F38",
    "#FBC02D", "#F57C00", "#5D4037", "#455A64", "#616161",
    "#AFB42B", "#E64A19", "#9E9E9E", "#212121", "#FFEB3B"
]


def get_color_for_category(category):
    """
    Get the consistent color for a given category.
    
    Args:
        category (str): Object category name
        
    Returns:
        str: Hex color code
    """
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["_default_"])


# Updated plot function section - replace lines 315-324 in the original notebook
def assign_colors_to_objects(all_coordinates, color_map):
    """
    Assign consistent colors to object categories.
    
    Args:
        all_coordinates (dict): Dictionary of object coordinates
        color_map (dict): Dictionary to populate with category->color mappings
    """
    from matplotlib.colors import to_rgba
    
    all_descriptions = {all_coordinates[o][4] for o in all_coordinates.keys()}
    
    # Use predefined colors for known categories
    for desc in all_descriptions:
        if desc in CATEGORY_COLORS:
            color_map[desc] = to_rgba(CATEGORY_COLORS[desc])
        else:
            # Use default color for unknown categories
            color_map[desc] = to_rgba(CATEGORY_COLORS["_default_"])
            print(f"Warning: No predefined color for category '{desc}', using default gray")
