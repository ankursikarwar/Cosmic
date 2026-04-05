import re

object_mapping = {
    "Standing Sink": "Sink",
    "Bathroom Sink": "Sink",
    "Desk Lamp": "Lamp",
    "Floor Lamp": "Lamp",
    "Bar Chair": "Chair",
    "Office Chair": "Chair",
    "Single Cabinet": "Cabinet",
    "Kitchen Cabinet": "Cabinet",
    "Cell Shelf": "Shelf",
    "Large Shelf": "Shelf",
    "Triangle Shelf": "Shelf",
    "Wall Shelf": "Shelf",
    "T V Stand": "Shelf",
    "Simple Bookcase": "Shelf",
    "Simple Desk": "Desk",
    "Coffee Table":"Centre Table",
    "Plant Container": "Plant Container",
    "Large Plant Container": "Plant Container",
    "Glass Panel Door": "Door",
    "Lite Door": "Door",
    "Louver Door": "Door",
    "Panel Door": "Door",
    "Beverage Fridge": "Fridge",
}

allowed_categories = [
    "Sofa",
    "Lamp",
    "Monitor",
    "Desk",
    "Shelf",
    "Door",
    "Window",
    "Cabinet",
    "Side Table",
    "Centre Table",
    "T V",
    "Bed",
    "Sink",
    "Plant Container",
    "Toilet",
    "Bathtub",
    "Oven",
    "Dishwasher",
    "Fridge",
    "Wall Art"
]

def get_category(key):
    out_key = re.sub(r"[0-9]+", "", key).strip()
    if out_key in object_mapping:
        return object_mapping[out_key]
    return out_key