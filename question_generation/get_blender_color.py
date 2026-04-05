import bpy
import json
import numpy as np
import os
import argparse
import sys

# Filter out Blender's arguments and only parse our own
script_args = []
i = 0
while i < len(sys.argv):
    if sys.argv[i] == "--output_json" and i + 1 < len(sys.argv):
        script_args.extend([sys.argv[i], sys.argv[i + 1]])
        i += 2
    elif sys.argv[i].startswith("--output_json="):
        script_args.append(sys.argv[i])
        i += 1
    else:
        i += 1

parser = argparse.ArgumentParser()
parser.add_argument("--output_json", type=str, default="./blender_colors.json")
args = parser.parse_args(script_args)
OUTPUT_JSON = args.output_json




blender_colors = {}


toilet_material = ['ceramic']
sink_material = ['ceramic', 'brushed', 'basic', 'galvanized', 'grained', 'hammered']
fridge_material = ['basic', 'brushed', 'galvanized', 'grained', 'hammered']
oven_material = ['basic', 'brushed', 'galvanized', 'grained', 'hammered']
mattress_material = ['fine', 'leather']
bedframe_material = ['wood', 'ceramic', 'plaster', 'brushed', 'galvanized', 'grained', 'hammered']
lamp_material = ['lamp', 'coarse', 'fine', 'leather', 'sofa']
side_table_material = ['wood', 'marble', 'plastic']
door_material = ['wood', 'plastic', 'metal', 'shelves']
coffee_table_materials = ['wood', 'marble', 'plastic']
sofa_material = ['leather', 'fine_knit']
window_materials = ['fine_knit', 'plastic', 'shader_wood', 'lamp']
plant_container_materials = ['brushed', 'galvanized', 'grained', 'hammered', 'ceramic', 'marble']

def get_socket_value(socket, visited=None):
    """Get the actual value from a socket, tracing through links if necessary."""
    if visited is None:
        visited = set()
    
    # If not linked, return the default value
    if not socket.is_linked:
        return getattr(socket, "default_value", None)
    
    # If linked, trace back through the connection
    for link in socket.links:
        from_node = link.from_node
        if id(from_node) in visited:
            continue
        visited.add(id(from_node))
        
        # Get the output socket that's connected
        output_socket = link.from_socket
        
        # Handle different node types
        if from_node.type == "VALUE":
            # Value node - if input is linked, trace it; otherwise use input default
            if "Value" in from_node.inputs:
                input_socket = from_node.inputs["Value"]
                if input_socket.is_linked:
                    return get_socket_value(input_socket, visited)
                else:
                    return getattr(input_socket, "default_value", None)
            # Fallback to output default value
            return getattr(output_socket, "default_value", None)
        elif from_node.type == "RGB":
            # RGB node - if input is linked, trace it; otherwise use input default
            if "Color" in from_node.inputs:
                input_socket = from_node.inputs["Color"]
                if input_socket.is_linked:
                    return get_socket_value(input_socket, visited)
                else:
                    return getattr(input_socket, "default_value", None)
            # Fallback to output default value
            return getattr(output_socket, "default_value", None)
        elif from_node.type == "COMBINE_COLOR":
            # Combine color node - combine R, G, B values by tracing inputs
            r = get_socket_value(from_node.inputs["Red"], visited) if "Red" in from_node.inputs else 0.0
            g = get_socket_value(from_node.inputs["Green"], visited) if "Green" in from_node.inputs else 0.0
            b = get_socket_value(from_node.inputs["Blue"], visited) if "Blue" in from_node.inputs else 0.0
            # Ensure values are floats
            r = float(r) if r is not None else 0.0
            g = float(g) if g is not None else 0.0
            b = float(b) if b is not None else 0.0
            return (r, g, b, 1.0)  # Return as RGBA tuple
        elif from_node.type == "SEPARATE_COLOR":
            # Separate color node - get the specific channel
            if "Color" in from_node.inputs:
                color = get_socket_value(from_node.inputs["Color"], visited)
                if color is not None:
                    # Convert color to list to handle bpy_prop_array
                    try:
                        color_list = list(color)
                    except (TypeError, ValueError):
                        color_list = []
                    
                    # Determine which output we're connected to
                    for output_idx, output in enumerate(from_node.outputs):
                        if output == output_socket:
                            if output_idx == 0:  # Red
                                return float(color_list[0]) if len(color_list) > 0 else 0.0
                            elif output_idx == 1:  # Green
                                return float(color_list[1]) if len(color_list) > 1 else 0.0
                            elif output_idx == 2:  # Blue
                                return float(color_list[2]) if len(color_list) > 2 else 0.0
        elif from_node.type == "MIX":
            # Mix node - try to get the Color output
            if hasattr(from_node, "outputs") and "Color" in from_node.outputs:
                output_socket = from_node.outputs["Color"]
                if not output_socket.is_linked:
                    return getattr(output_socket, "default_value", None)
                # If linked, trace through (but we're already in a recursive call, so this might not work)
                # Better to try to get Color1 or Color2
            # Try Color1 as fallback
            if "Color1" in from_node.inputs:
                color1_input = from_node.inputs["Color1"]
                if not color1_input.is_linked:
                    return getattr(color1_input, "default_value", None)
                else:
                    return get_socket_value(color1_input, visited)
        elif from_node.type == "TEX_NOISE":
            # Noise Texture - procedural, can't get static value easily
            # Try to get Color output if available
            if hasattr(from_node, "outputs") and "Color" in from_node.outputs:
                output_socket = from_node.outputs["Color"]
                if not output_socket.is_linked:
                    return getattr(output_socket, "default_value", None)
        else:
            # For other node types, try to get the output default value
            # If output has a default value, use it
            if hasattr(output_socket, "default_value"):
                value = output_socket.default_value
                if value is not None:
                    return value
            # If no default value, try to trace inputs if they exist
            # This handles nodes like math nodes, mix nodes, etc.
            if hasattr(from_node, "inputs") and len(from_node.inputs) > 0:
                # Try the first input (common for many node types)
                first_input = from_node.inputs[0]
                if first_input.is_linked:
                    return get_socket_value(first_input, visited)
                else:
                    return getattr(first_input, "default_value", None)
    
    # Fallback: return None if we can't determine the value
    return None

def trace_color_links(socket, material_name, obj, depth=1, visited=None):
    """Recursively trace linked color inputs and print color values."""
    if visited is None:
        visited = set()

    indent = "    " * depth

    if not socket.is_linked:
        value = getattr(socket, "default_value", None)
        if value is not None:
            # Convert bpy_prop_array to list and ensure all values are floats
            rgba = [float(c) for c in list(value)]
            rgb_255 = [round(c * 255) for c in rgba[:3]]
            print(f"{indent}→ Base Color RGBA: {rgba}")
            print(f"{indent}→ Base Color RGB (0–255): {rgb_255}")
            
            if 'sink' in obj.name.lower():
                if obj.name in blender_colors:
                    pass
                else:
                    blender_colors[obj.name] = {
                        "name": obj.name,
                        "material": material_name,
                        "type": "base_color",
                        "color": rgb_255
                    }
            else:
                blender_colors[obj.name] = {
                    "name": obj.name,
                    "material": material_name,
                    "type": "base_color",
                    "color": rgb_255
                }
        return

    for link in socket.links:
        from_node = link.from_node
        if from_node in visited:
            continue
        visited.add(from_node)

        print(f"{indent}→ Linked to node: {from_node.name} ({from_node.type})")
        if from_node.type == "RGB":
            # Get actual RGB color value from the Color input
            # RGB nodes have a Color input that can be linked or have a default value
            color_value = None
            if "Color" in from_node.inputs:
                color_input = from_node.inputs["Color"]
                # Get the value, tracing through links if needed
                color_value = get_socket_value(color_input)
            else:
                # Fallback: try to get from output socket's default value
                if hasattr(from_node, "outputs") and len(from_node.outputs) > 0:
                    output_socket = from_node.outputs[0]
                    color_value = getattr(output_socket, "default_value", None)
            
            if color_value is not None:
                # Handle tuple/list/bpy_prop_array or single value
                # Try to convert to list first (handles bpy_prop_array)
                try:
                    value_list = list(color_value)
                    if len(value_list) > 0:
                        rgba = [float(c) for c in value_list]
                    else:
                        # Empty list, use default
                        rgba = [0.0, 0.0, 0.0, 1.0]
                except (TypeError, ValueError):
                    # If it's not iterable, try to convert directly to float
                    try:
                        c_float = float(color_value)
                        rgba = [c_float, c_float, c_float, 1.0]
                    except (TypeError, ValueError):
                        # If all else fails, use default
                        rgba = [0.0, 0.0, 0.0, 1.0]
                
                # Ensure we have at least 3 values
                while len(rgba) < 3:
                    rgba.append(0.0)
                if len(rgba) < 4:
                    rgba.append(1.0)
                
                # Convert to RGB 0-255 (ensure all values are floats)
                rgb_255 = []
                for c in rgba[:3]:
                    c_float = float(c)
                    if c_float <= 1.0:
                        rgb_255.append(round(c_float * 255))
                    else:
                        rgb_255.append(round(c_float))
                print(f"{indent}  RGB value: {rgba[:3]}")
                print(f"{indent}  RGB (0-255): {rgb_255}")
                blender_colors[obj.name] = {
                    "name": obj.name,
                    "material": material_name,
                    "type": "rgb",
                    "color": rgb_255
                }
        elif from_node.type == "COMBINE_COLOR":
            # Get actual values (tracing through links if needed)
            red_value = get_socket_value(from_node.inputs['Red']) if 'Red' in from_node.inputs else None
            green_value = get_socket_value(from_node.inputs['Green']) if 'Green' in from_node.inputs else None
            blue_value = get_socket_value(from_node.inputs['Blue']) if 'Blue' in from_node.inputs else None
            
            print(f"{indent}  Red value: {red_value}")
            print(f"{indent}  Green value: {green_value}")
            print(f"{indent}  Blue value: {blue_value}")
            
            # If we have all three values, combine them and save
            if red_value is not None and green_value is not None and blue_value is not None:
                # Ensure values are in 0-1 range (handle if they're already 0-255)
                # Convert to float first to handle bpy_prop_array
                r_float = float(red_value)
                g_float = float(green_value)
                b_float = float(blue_value)
                
                r = r_float if r_float <= 1.0 else r_float / 255.0
                g = g_float if g_float <= 1.0 else g_float / 255.0
                b = b_float if b_float <= 1.0 else b_float / 255.0
                
                rgb_255 = [round(r * 255), round(g * 255), round(b * 255)]
                print(f"{indent}  Combined RGB (0-255): {rgb_255}")
                blender_colors[obj.name] = {
                    "name": obj.name,
                    "material": material_name,
                    "type": "combine_color",
                    "color": rgb_255
                }
        elif from_node.type == "MIX":
            # Mix node - blend between Color1 and Color2 based on Fac
            # Try to get both colors and the mix factor
            color1_value = None
            color2_value = None
            fac_value = None
            
            if "Color1" in from_node.inputs:
                color1_value = get_socket_value(from_node.inputs["Color1"])
            if "Color2" in from_node.inputs:
                color2_value = get_socket_value(from_node.inputs["Color2"])
            if "Fac" in from_node.inputs:
                fac_value = get_socket_value(from_node.inputs["Fac"])
            
            print(f"{indent}  Color1: {color1_value}")
            print(f"{indent}  Color2: {color2_value}")
            print(f"{indent}  Fac (mix factor): {fac_value}")
            
            # Try to get the output color (mixed result)
            if hasattr(from_node, "outputs") and "Color" in from_node.outputs:
                output_color = get_socket_value(from_node.outputs["Color"])
                if output_color is None:
                    # Try to get default value
                    output_socket = from_node.outputs["Color"]
                    if not output_socket.is_linked:
                        output_color = getattr(output_socket, "default_value", None)
                
                if output_color is not None:
                    # Convert to RGB
                    try:
                        color_list = list(output_color)
                        if len(color_list) >= 3:
                            rgba = [float(c) for c in color_list[:3]]
                            rgba.append(1.0 if len(color_list) < 4 else float(color_list[3]))
                        else:
                            rgba = [0.0, 0.0, 0.0, 1.0]
                    except (TypeError, ValueError):
                        rgba = [0.0, 0.0, 0.0, 1.0]
                    
                    # Ensure we have at least 3 values
                    while len(rgba) < 3:
                        rgba.append(0.0)
                    if len(rgba) < 4:
                        rgba.append(1.0)
                    
                    rgb_255 = []
                    for c in rgba[:3]:
                        c_float = float(c)
                        if c_float <= 1.0:
                            rgb_255.append(round(c_float * 255))
                        else:
                            rgb_255.append(round(c_float))
                    
                    print(f"{indent}  Mixed RGB (0-255): {rgb_255}")
                    blender_colors[obj.name] = {
                        "name": obj.name,
                        "material": material_name,
                        "type": "mix",
                        "color": rgb_255,
                        "color1": color1_value,
                        "color2": color2_value,
                        "fac": fac_value
                    }
            # If we can't get output, try to use Color1 or Color2 as fallback
            elif color1_value is not None:
                try:
                    color_list = list(color1_value)
                    if len(color_list) >= 3:
                        rgba = [float(c) for c in color_list[:3]]
                        rgba.append(1.0 if len(color_list) < 4 else float(color_list[3]))
                    else:
                        rgba = [0.0, 0.0, 0.0, 1.0]
                except (TypeError, ValueError):
                    rgba = [0.0, 0.0, 0.0, 1.0]
                
                rgb_255 = []
                for c in rgba[:3]:
                    c_float = float(c)
                    if c_float <= 1.0:
                        rgb_255.append(round(c_float * 255))
                    else:
                        rgb_255.append(round(c_float))
                
                print(f"{indent}  Using Color1 RGB (0-255): {rgb_255}")
                blender_colors[obj.name] = {
                    "name": obj.name,
                    "material": material_name,
                    "type": "mix_color1",
                    "color": rgb_255
                }
        elif from_node.type == "TEX_NOISE":
            # Noise Texture node - procedural texture, can't get exact RGB without evaluation
            # But we can try to get Color output if available
            if hasattr(from_node, "outputs") and "Color" in from_node.outputs:
                output_socket = from_node.outputs["Color"]
                if not output_socket.is_linked:
                    # If output is not linked, try to get default value
                    color_value = getattr(output_socket, "default_value", None)
                    if color_value is not None:
                        try:
                            color_list = list(color_value)
                            if len(color_list) >= 3:
                                rgba = [float(c) for c in color_list[:3]]
                                rgba.append(1.0 if len(color_list) < 4 else float(color_list[3]))
                            else:
                                rgba = [0.0, 0.0, 0.0, 1.0]
                        except (TypeError, ValueError):
                            rgba = [0.0, 0.0, 0.0, 1.0]
                        
                        rgb_255 = []
                        for c in rgba[:3]:
                            c_float = float(c)
                            if c_float <= 1.0:
                                rgb_255.append(round(c_float * 255))
                            else:
                                rgb_255.append(round(c_float))
                        
                        print(f"{indent}  Noise Texture Color (0-255): {rgb_255}")
                        blender_colors[obj.name] = {
                            "name": obj.name,
                            "material": material_name,
                            "type": "noise_texture",
                            "color": rgb_255
                        }
                else:
                    # Output is linked, note that this is a procedural texture
                    print(f"{indent}  Noise Texture (procedural - cannot extract static RGB)")
                    # We could potentially evaluate at a specific coordinate, but that's complex
                    # For now, just note that it's a procedural texture
                    blender_colors[obj.name] = {
                        "name": obj.name,
                        "material": material_name,
                        "type": "noise_texture_procedural",
                        "color": None,
                        "note": "Procedural texture - requires evaluation at specific coordinates"
                    }


        # Check for color or base color inputs to trace further
        for key in ("Base Color", "Color", "A", "Red", "Green", "Blue"):
            if key in from_node.inputs:
                trace_color_links(from_node.inputs[key], material_name, obj, depth + 1, visited)



def process_materials(obj, target_materials):
    """Find and print Base Color info for matching materials."""
    print(f"Object: {obj.name}")
    for slot, mat in enumerate(obj.data.materials):
        print(f"  Material: {mat.name}")
        if not mat:
            continue

        if any(material in mat.name.lower() for material in target_materials):
            print(f"  Found matching material: {mat.name}")

            for prop in mat.keys():
                print(f"    Property - {prop}: {mat[prop]}")

            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    base_input = node.inputs.get("Base Color")
                    if base_input:
                        print(f"    Node: {node.name} ({node.type})")
                        trace_color_links(base_input, mat.name, obj)
            else:
                print("    Material does not use nodes.")


# MAIN LOOP
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.data and not obj.hide_render:
        name = obj.name.lower()
        print("name", name)

        if "plant" in name and "spawn_asset" in name:
            process_materials(obj, plant_container_materials)

        if "window" in name and "spawn_asset" in name:
            process_materials(obj, window_materials)

        if "sofa" in name and "spawn_asset" in name:
            process_materials(obj, sofa_material)

        if "coffee" in name and "spawn_asset" in name:
            process_materials(obj, coffee_table_materials)

        if "door" in name and "spawn_asset" in name:
            process_materials(obj, door_material)

        if "side" in name and "spawn_asset" in name:
            process_materials(obj, side_table_material)

        if "lamp" in name and "spawn_asset" in name:
            process_materials(obj, lamp_material)

        if "bedframe" in name and "spawn_asset" in name:
            process_materials(obj, bedframe_material)

        if "mattress" in name and "spawn_asset" in name:
            process_materials(obj, mattress_material)

        if "oven" in name and "spawn_asset" in name:
            process_materials(obj, oven_material)

        if "fridge" in name and "spawn_asset" in name:
            process_materials(obj, fridge_material)

        if "sink" in name and "spawn_asset" in name:
            process_materials(obj, sink_material)

        if "toilet" in name and "spawn_asset" in name:
            process_materials(obj, toilet_material)


with open(OUTPUT_JSON, 'w') as f:
    json.dump(blender_colors, f, indent=4)