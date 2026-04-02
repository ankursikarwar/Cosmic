import json
import os
import argparse
from PIL import Image, ImageDraw, ImageFont

PADDING_RATIO = 0.05

def draw_bounding_boxes(image_path, json_path, camera_key, output_dir, box_thickness=3, font_size=16):
    with open(json_path, "r") as f:
        data = json.load(f)

    objects = data.get(camera_key, [])
    if not objects:
        print(f"No objects found for {camera_key}")
        # return None

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving output images in: {output_dir}")

    img = Image.open(image_path)
    W, H = img.size

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)  # macOS
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)  # Linux
            except:
                font = ImageFont.load_default()

    # Work on a single copy of the image
    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    for obj in objects:
        name = objects[obj]["name"]
        x_min, y_min, x_max, y_max = objects[obj]["bbox_2d"]
        color = (255, 0, 0)  # Could randomize if you want different colors

        # Convert normalized coordinates to pixel coordinates
        left = int(x_min * W)
        right = int(x_max * W)
        top = int((1 - y_max) * H)
        bottom = int((1 - y_min) * H)

        bbox_width = right - left
        bbox_height = bottom - top
        pad_x = int(bbox_width * PADDING_RATIO)
        pad_y = int(bbox_height * PADDING_RATIO)

        left = max(0, left - pad_x)
        right = min(W, right + pad_x)
        top = max(0, top - pad_y)
        bottom = min(H, bottom + pad_y)

        if top >= bottom or left >= right:
            print(f"Skipping {name}: Invalid bounding box coordinates.")
            continue

        # Draw the bounding box
        for thickness_offset in range(box_thickness):
            draw.rectangle(
                [left - thickness_offset, top - thickness_offset,
                 right + thickness_offset, bottom + thickness_offset],
                outline=color
            )

        # Label text
        label_text = f"{name}"
        bbox_text = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        label_x = max(0, left)
        label_y = max(0, top - text_height - 4)
        if label_y < 0:
            label_y = top + 2

        # Draw background box for label
        draw.rectangle(
            [label_x - 2, label_y - 2,
             label_x + text_width + 2, label_y + text_height + 2],
            fill=color
        )

        # Draw text
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        draw.text((label_x, label_y), label_text, fill=text_color, font=font)

    # Save the final image with all boxes
    output_path = os.path.join(output_dir, f"{camera_key}_all_boxes.png")
    img_with_boxes.save(output_path)
    print(f"✅ Saved {len(objects)} bounding boxes for {camera_key} at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--llm_detected_objects_json", type=str, required=True)
    parser.add_argument("--camera_key", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--box_thickness", type=int, default=3)
    parser.add_argument("--font_size", type=int, default=16)
    args = parser.parse_args()

    draw_bounding_boxes(args.image, args.llm_detected_objects_json, args.camera_key, args.output_dir, args.box_thickness, args.font_size)
