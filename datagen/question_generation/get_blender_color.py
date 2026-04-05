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

bathtub_material = ['ceramic']
dining_material = ['marble', 'wood', 'plastic']
toilet_material = ['ceramic']
chair_material = ['wood', 'plastic']
sink_material = ['ceramic', 'brushed', 'basic', 'galvanized', 'grained', 'hammered']
fridge_material = ['basic', 'brushed', 'galvanized', 'grained', 'hammered', 'metal']
oven_material = ['basic', 'brushed', 'galvanized', 'grained', 'hammered', 'metal']
mattress_material = ['fine', 'leather', 'coarse']
bedframe_material = ['wood', 'ceramic', 'plaster', 'brushed', 'galvanized', 'grained', 'hammered', 'plastic']
lamp_material = ['lamp', 'coarse', 'fine', 'leather', 'sofa']
side_table_material = ['wood', 'marble', 'plastic']
door_material = ['wood', 'plastic', 'metal', 'shelves']
coffee_table_materials = ['wood', 'marble', 'plastic']
sofa_material = ['leather', 'fine_knit', 'coarse_knit']
window_materials = ['shader_wood', 'fine_knit', 'plastic', 'lamp', 'shader_shelves']
plant_container_materials = ['brushed', 'galvanized', 'grained', 'hammered', 'ceramic', 'marble']
simpledesk_material = ['wood', 'shader_shelves_black_wood', 'shader_shelves_white', 'shader_shelves_wood']




def linear_to_srgb(c):
    if c <= 0.0031308:
        return 12.92 * c
    else:
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def get_true_rgb_from_node(from_node):
    # Make sure we are in the evaluated depsgraph context
    depsgraph = bpy.context.evaluated_depsgraph_get()
    node_tree = from_node.id_data.evaluated_get(depsgraph)

    # Find the corresponding evaluated node
    eval_node = node_tree.nodes.get(from_node.name)
    if eval_node and len(eval_node.outputs) > 0:
        color_rgba = eval_node.outputs[0].default_value
        return list(color_rgba[:3])
    else:
        # fallback if something goes wrong
        return list(from_node.outputs[0].default_value[:3])


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
            # rgb_value = get_true_rgb_from_node(from_node)
            # rgb_255 = [int(round(c * 255)) for c in rgb_value]
            # print(f"{indent}  RGB (0-255): {rgb_255}")
            
            # blender_colors[obj.name] = {
            #     "name": obj.name,
            #     "material": material_name,
            #     "type": "rgb",
            #     "color": rgb_255
            # }
            # RGB node - Color INPUT contains the actual RGBA values
            # Prioritize Color input first (this is where the RGBA values are stored)
            if "Color" in from_node.inputs:
                input_socket = from_node.inputs["Color"]
                if not input_socket.is_linked:
                    # Unlinked input - this is where the actual RGBA values are
                    return getattr(input_socket, "default_value", None)
                else:
                    # Linked input - trace through to get the value
                    return get_socket_value(input_socket, visited)
            
            # Fallback: try Color output if input didn't work
            color_output = None
            rgba_output = None
            for output in from_node.outputs:
                if output.name == "Color":
                    color_output = output
                elif output.name == "RGBA":
                    rgba_output = output
            
            if color_output is not None:
                return getattr(color_output, "default_value", None)
            elif rgba_output is not None:
                return getattr(rgba_output, "default_value", None)
            elif hasattr(output_socket, "default_value"):
                return getattr(output_socket, "default_value", None)
            
            return None
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
        elif from_node.type == "TEX_WHITE_NOISE":
            # White Noise Texture - procedural, similar to regular noise
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

        print(f"{indent}  Value: {value}")

        if value is not None:

            # rgba = [float(c) for c in list(value)]            
            # rgb_255 = [round(c * 255) for c in rgba[:3]]

            color_linear = value[:3]
            color_srgb = [linear_to_srgb(c) for c in color_linear]
            rgb_255 = [int(round(c * 255)) for c in color_srgb]

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
        
        #########################################################
        if from_node.type == "RGB":
            color_linear = from_node.outputs["Color"].default_value[:3]
            color_srgb = [linear_to_srgb(c) for c in color_linear]
            color_255 = [int(round(c * 255)) for c in color_srgb]

            print(f"{indent}  Color linear: {color_linear}")
            print(f"{indent}  Color srgb: {color_srgb}")
            print(f"{indent}  Color 255: {color_255}")

            blender_colors[obj.name] = {
                "name": obj.name,
                "material": material_name,
                "type": "rgb",
                "color": color_255,            # what you see in Blender UI
            }
        #########################################################


        #########################################################
        elif from_node.type == "COMBINE_COLOR":
            red_value = get_socket_value(from_node.inputs['Red']) if 'Red' in from_node.inputs else None
            green_value = get_socket_value(from_node.inputs['Green']) if 'Green' in from_node.inputs else None
            blue_value = get_socket_value(from_node.inputs['Blue']) if 'Blue' in from_node.inputs else None
            
            print(f"{indent}  Red value: {red_value}")
            print(f"{indent}  Green value: {green_value}")
            print(f"{indent}  Blue value: {blue_value}")
            
            if red_value is not None and green_value is not None and blue_value is not None:
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
        #########################################################


        #########################################################
        elif from_node.type == "HUE_SAT":
            value_linear = from_node.inputs["Color"].default_value[:3]
            value_srgb = [linear_to_srgb(c) for c in value_linear]
            value_255 = [round(c * 255) for c in value_srgb]
            print(f"{indent}  Hue Sat: {value_linear}")
            print(f"{indent}  Hue Sat srgb: {value_srgb}")
            print(f"{indent}  Hue Sat 255: {value_255}")
            blender_colors[obj.name] = {
                "name": obj.name,
                "material": material_name,
                "type": "hue_sat",
                "color": value_255
            }
        #########################################################


        #########################################################
        elif from_node.type == "MIX":
            print(f"{indent}  Mix: {from_node.inputs}")
            value = from_node.inputs[7].default_value[:3]
            value_srgb = [linear_to_srgb(c) for c in value]
            value_255 = [round(c * 255) for c in value_srgb]
            print(f"{indent}  Mix: {value}")
            print(f"{indent}  Mix srgb: {value_srgb}")
            print(f"{indent}  Mix 255: {value_255}")
            blender_colors[obj.name] = {
                "name": obj.name,
                "material": material_name,
                "type": "mix",
                "color": value_255
            }
            return
        #########################################################


        #########################################################
        # elif from_node.type == "VALTORGB":
        #     print(f"{indent} ValToRgb: {from_node.inputs}")
        #     value = from_node.outputs["Color"].default_value[:3]
        #     value_srgb = [linear_to_srgb(c) for c in value]
        #     value_255 = [round(c * 255) for c in value_srgb]
        #     print(f"{indent}  ValToRgb: {value}")
        #     print(f"{indent}  ValToRgb srgb: {value_srgb}")
        #     print(f"{indent}  ValToRgb 255: {value_255}")
        #     blender_colors[obj.name] = {
        #         "name": obj.name,
        #         "material": material_name,
        #         "type": "valtorgb",
        #         "color": value_255
        #     }
        #     return
        #########################################################


        # elif from_node.type == "MIX":
        #     # Mix node - blend between Color1 and Color2 based on Fac
        #     # Try to get both colors and the mix factor
        #     color1_value = None
        #     color2_value = None
        #     fac_value = None
            
        #     if "Color1" in from_node.inputs:
        #         color1_input = from_node.inputs["Color1"]
        #         color1_value = get_socket_value(color1_input)
        #         # If Color1 is linked, also trace through it to get the actual color
        #         if color1_value is None and color1_input.is_linked:
        #             # Try to get the color from the linked node
        #             for link in color1_input.links:
        #                 linked_node = link.from_node
        #                 print(f"{indent}    Color1 linked to: {linked_node.name} ({linked_node.type})")
        #     if "Color2" in from_node.inputs:
        #         color2_input = from_node.inputs["Color2"]
        #         color2_value = get_socket_value(color2_input)
        #         # If Color2 is linked, also trace through it to get the actual color
        #         if color2_value is None and color2_input.is_linked:
        #             # Try to get the color from the linked node
        #             for link in color2_input.links:
        #                 linked_node = link.from_node
        #                 print(f"{indent}    Color2 linked to: {linked_node.name} ({linked_node.type})")
        #     if "Fac" in from_node.inputs:
        #         fac_value = get_socket_value(from_node.inputs["Fac"])
            
        #     print(f"{indent}  Color1: {color1_value}")
        #     print(f"{indent}  Color2: {color2_value}")
        #     print(f"{indent}  Fac (mix factor): {fac_value}")
            
        #     # Try to get the output color (mixed result)
        #     if hasattr(from_node, "outputs") and "Color" in from_node.outputs:
        #         output_color = get_socket_value(from_node.outputs["Color"])
        #         if output_color is None:
        #             # Try to get default value
        #             output_socket = from_node.outputs["Color"]
        #             if not output_socket.is_linked:
        #                 output_color = getattr(output_socket, "default_value", None)
                
        #         if output_color is not None:
        #             # Convert to RGB
        #             try:
        #                 color_list = list(output_color)
        #                 if len(color_list) >= 3:
        #                     rgba = [float(c) for c in color_list[:3]]
        #                     rgba.append(1.0 if len(color_list) < 4 else float(color_list[3]))
        #                 else:
        #                     rgba = [0.0, 0.0, 0.0, 1.0]
        #             except (TypeError, ValueError):
        #                 rgba = [0.0, 0.0, 0.0, 1.0]
                    
        #             # Ensure we have at least 3 values
        #             while len(rgba) < 3:
        #                 rgba.append(0.0)
        #             if len(rgba) < 4:
        #                 rgba.append(1.0)
                    
        #             rgb_255 = []
        #             for c in rgba[:3]:
        #                 c_float = float(c)
        #                 if c_float <= 1.0:
        #                     rgb_255.append(round(c_float * 255))
        #                 else:
        #                     rgb_255.append(round(c_float))
                    
        #             print(f"{indent}  Mixed RGB (0-255): {rgb_255}")
        #             blender_colors[obj.name] = {
        #                 "name": obj.name,
        #                 "material": material_name,
        #                 "type": "mix",
        #                 "color": rgb_255,
        #                 "color1": color1_value,
        #                 "color2": color2_value,
        #                 "fac": fac_value
        #             }
        #     # If we can't get output, try to use Color1 or Color2 as fallback
        #     elif color1_value is not None:
        #         try:
        #             color_list = list(color1_value)
        #             if len(color_list) >= 3:
        #                 rgba = [float(c) for c in color_list[:3]]
        #                 rgba.append(1.0 if len(color_list) < 4 else float(color_list[3]))
        #             else:
        #                 rgba = [0.0, 0.0, 0.0, 1.0]
        #         except (TypeError, ValueError):
        #             rgba = [0.0, 0.0, 0.0, 1.0]
                
        #         rgb_255 = []
        #         for c in rgba[:3]:
        #             c_float = float(c)
        #             if c_float <= 1.0:
        #                 rgb_255.append(round(c_float * 255))
        #             else:
        #                 rgb_255.append(round(c_float))
                
        #         print(f"{indent}  Using Color1 RGB (0-255): {rgb_255}")
        #         blender_colors[obj.name] = {
        #             "name": obj.name,
        #             "material": material_name,
        #             "type": "mix_color1",
        #             "color": rgb_255
        #         }
        # elif from_node.type == "TEX_NOISE":
        #     # Noise Texture node - procedural texture, can't get exact RGB without evaluation
        #     # But we can try to get Color output if available
        #     if hasattr(from_node, "outputs") and "Color" in from_node.outputs:
        #         output_socket = from_node.outputs["Color"]
        #         if not output_socket.is_linked:
        #             # If output is not linked, try to get default value
        #             color_value = getattr(output_socket, "default_value", None)
        #             if color_value is not None:
        #                 try:
        #                     color_list = list(color_value)
        #                     if len(color_list) >= 3:
        #                         rgba = [float(c) for c in color_list[:3]]
        #                         rgba.append(1.0 if len(color_list) < 4 else float(color_list[3]))
        #                     else:
        #                         rgba = [0.0, 0.0, 0.0, 1.0]
        #                 except (TypeError, ValueError):
        #                     rgba = [0.0, 0.0, 0.0, 1.0]
                        
        #                 rgb_255 = []
        #                 for c in rgba[:3]:
        #                     c_float = float(c)
        #                     if c_float <= 1.0:
        #                         rgb_255.append(round(c_float * 255))
        #                     else:
        #                         rgb_255.append(round(c_float))
                        
        #                 print(f"{indent}  Noise Texture Color (0-255): {rgb_255}")
        #                 blender_colors[obj.name] = {
        #                     "name": obj.name,
        #                     "material": material_name,
        #                     "type": "noise_texture",
        #                     "color": rgb_255
        #                 }
        #         else:
        #             # Output is linked, note that this is a procedural texture
        #             print(f"{indent}  Noise Texture (procedural - cannot extract static RGB)")
        #             # We could potentially evaluate at a specific coordinate, but that's complex
        #             # For now, just note that it's a procedural texture
        #             blender_colors[obj.name] = {
        #                 "name": obj.name,
        #                 "material": material_name,
        #                 "type": "noise_texture_procedural",
        #                 "color": None,
        #                 "note": "Procedural texture - requires evaluation at specific coordinates"
        #             }
        # elif from_node.type == "TEX_WHITE_NOISE":
        #     # White Noise Texture node - procedural texture, similar to regular noise
        #     # Try to get Color output if available
        #     if hasattr(from_node, "outputs") and "Color" in from_node.outputs:
        #         output_socket = from_node.outputs["Color"]
        #         if not output_socket.is_linked:
        #             # If output is not linked, try to get default value
        #             color_value = getattr(output_socket, "default_value", None)
        #             if color_value is not None:
        #                 try:
        #                     color_list = list(color_value)
        #                     if len(color_list) >= 3:
        #                         rgba = [float(c) for c in color_list[:3]]
        #                         rgba.append(1.0 if len(color_list) < 4 else float(color_list[3]))
        #                     else:
        #                         rgba = [0.0, 0.0, 0.0, 1.0]
        #                 except (TypeError, ValueError):
        #                     rgba = [0.0, 0.0, 0.0, 1.0]
                        
        #                 rgb_255 = []
        #                 for c in rgba[:3]:
        #                     c_float = float(c)
        #                     if c_float <= 1.0:
        #                         rgb_255.append(round(c_float * 255))
        #                     else:
        #                         rgb_255.append(round(c_float))
                        
        #                 print(f"{indent}  White Noise Texture Color (0-255): {rgb_255}")
        #                 blender_colors[obj.name] = {
        #                     "name": obj.name,
        #                     "material": material_name,
        #                     "type": "white_noise_texture",
        #                     "color": rgb_255
        #                 }
        #         else:
        #             # Output is linked, note that this is a procedural texture
        #             # But we can still try to get a value if it's connected to something we can trace
        #             print(f"{indent}  White Noise Texture (procedural - attempting to extract RGB)")
        #             # Try to get the value by tracing through the output
        #             color_value = get_socket_value(output_socket)
        #             if color_value is not None:
        #                 try:
        #                     color_list = list(color_value)
        #                     if len(color_list) >= 3:
        #                         rgba = [float(c) for c in color_list[:3]]
        #                         rgba.append(1.0 if len(color_list) < 4 else float(color_list[3]))
        #                     else:
        #                         rgba = [0.0, 0.0, 0.0, 1.0]
        #                 except (TypeError, ValueError):
        #                     rgba = [0.0, 0.0, 0.0, 1.0]
                        
        #                 rgb_255 = []
        #                 for c in rgba[:3]:
        #                     c_float = float(c)
        #                     if c_float <= 1.0:
        #                         rgb_255.append(round(c_float * 255))
        #                     else:
        #                         rgb_255.append(round(c_float))
                        
        #                 print(f"{indent}  White Noise Texture RGB (0-255): {rgb_255}")
        #                 blender_colors[obj.name] = {
        #                     "name": obj.name,
        #                     "material": material_name,
        #                     "type": "white_noise_texture",
        #                     "color": rgb_255
        #                 }
        #             else:
        #                 print(f"{indent}  White Noise Texture (procedural - cannot extract static RGB)")
        #                 blender_colors[obj.name] = {
        #                     "name": obj.name,
        #                     "material": material_name,
        #                     "type": "white_noise_texture_procedural",
        #                     "color": None,
        #                     "note": "Procedural texture - requires evaluation at specific coordinates"
        #                 }

        # elif from_node.type == "HUE_SAT":
        #     # print the 


        # Check for color or base color inputs to trace further
        for key in ("Base Color", "Color", "A", "Red", "Green", "Blue"):
            if key in from_node.inputs:
                trace_color_links(from_node.inputs[key], material_name, obj, depth + 1, visited)



def process_materials(obj, target_materials):
    """Find and print Base Color info for matching materials."""
    print(f"Object: {obj.name}")
    # for slot, mat in enumerate(obj.data.materials):
    mat = None
        
    if "window" in obj.name.lower():
        if len(obj.data.materials) > 1:
            mat = obj.data.materials[1]
        else:
            mat = obj.data.materials[0]
        print(f"  Material: {mat.name}")
    if "bed" in obj.name.lower():
        mat = obj.data.materials[1]
        print(f"  Material: {mat.name}")
    if "plant" in obj.name.lower():
        mat = obj.data.materials[0]
        print(f"  Material: {mat.name}")
    if "mattress" in obj.name.lower():
        mat = obj.data.materials[0]
        print(f"  Material: {mat.name}")
    if "lamp" in obj.name.lower():
        mat = obj.data.materials[0]
        print(f"  Material: {mat.name}")
    if "door" in obj.name.lower():
        mat = obj.data.materials[1]
        print(f"  Material: {mat.name}")
    if "simpledesk" in obj.name.lower():
        mat = obj.data.materials[1]
        print(f"  Material: {mat.name}")
    if "sink" in obj.name.lower():
        mat = obj.data.materials[0]
        print(f"  Material: {mat.name}")
    if "toilet" in obj.name.lower():
        mat = obj.data.materials[0]
        print(f"  Material: {mat.name}")
    if "bath" in obj.name.lower():
        mat = obj.data.materials[0]
        print(f"  Material: {mat.name}")
    if "dining" in obj.name.lower():
        mat = obj.data.materials[1]
        print(f"  Material: {mat.name}")
    if "chair" in obj.name.lower():
        mat = obj.data.materials[1]
        print(f"  Material: {mat.name}")
    if "sofa" in obj.name.lower():
        mat = obj.data.materials[0]
        print(f"  Material: {mat.name}")
    if "side" in obj.name.lower():
        mat = obj.data.materials[1]
        print(f"  Material: {mat.name}")
    if "coffee" in obj.name.lower():
        mat = obj.data.materials[1]
        print(f"  Material: {mat.name}")
    if "oven" in obj.name.lower():
        mat = obj.data.materials[0]
        print(f"  Material: {mat.name}")
    if "fridge" in obj.name.lower():
        mat = obj.data.materials[2]
        print(f"  Material: {mat.name}")
            
    if not mat:
        print(f"  No material found")
        return

    if any(material in mat.name.lower() for material in target_materials):
        print(f"  Found matching material: {mat.name}")

        if "shader_shelves_wood" in mat.name.lower():
            blender_colors[obj.name] = {
                "name": obj.name,
                "material": mat.name,
                "type": "base_color",
                "color": [
                    200,
                    157,
                    124
                ]
            }
            return
        if "shader_shelves_white" in mat.name.lower():
            blender_colors[obj.name] = {
                "name": obj.name,
                "material": mat.name,
                "type": "base_color",
                "color": [
                    int(round(0.955 * 255)),
                    int(round(0.955 * 255)),
                    int(round(0.955 * 255))
                ]
            }
            return
        if "shader_shelves_black_wood" in mat.name.lower():
            blender_colors[obj.name] = {
                "name": obj.name,
                "material": mat.name,
                "type": "base_color",
                "color": [
                    49,
                    47,
                    47
                ]
            }
            return

        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                print(f"    Node: {node.name} ({node.type})")
                base_input = node.inputs.get("Base Color")
                main_color = node.inputs.get("Main Color")
                if base_input:
                    print(f"    Node: {node.name} ({node.type})")
                    print(f"    Base Input: {base_input}")
                    print(f"    Material: {mat.name}")
                    print(f"    Object: {obj.name}")
                    trace_color_links(base_input, mat.name, obj)
                if main_color:
                    print(f"    Node: {node.name} ({node.type})")
                    print(f"    Main Color: {main_color}")
                    print(f"    Material: {mat.name}")
                    print(f"    Object: {obj.name}")
                    trace_color_links(main_color, mat.name, obj)
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

        if "bed" in name and "spawn_asset" in name:
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

        if "bath" in name and "spawn_asset" in name:
            process_materials(obj, bathtub_material)

        if "dining" in name and "spawn_asset" in name:
            process_materials(obj, dining_material)

        if "chair" in name and "spawn_asset" in name:
            process_materials(obj, chair_material)

        # if "simpledesk" in name and "spawn_asset" in name:
        #     process_materials(obj, simpledesk_material)


with open(OUTPUT_JSON, 'w') as f:
    json.dump(blender_colors, f, indent=4)