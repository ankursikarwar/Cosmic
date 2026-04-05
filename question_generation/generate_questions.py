import argparse
import json
import random
import numpy as np
from utils import get_category, allowed_categories
INTERSECTION_OBJS = 0
UNION_OBJS = 0

def parse_chosen_neighbor(chosen_neighbor_str):
    if not chosen_neighbor_str or chosen_neighbor_str is None:
        return (None, None)
    sorted_categories = sorted(allowed_categories, key=len, reverse=True)
    for category in sorted_categories:
        if category in chosen_neighbor_str:
            color = chosen_neighbor_str.replace(category, "").strip()
            return (color, category)
    return (None, None)

options_dict = {
    "front":["front", "behind", "left", "right"],
    "behind":["front", "behind", "left", "right"],
    "left":["front", "behind", "left", "right"],
    "right":["front", "behind", "left", "right"],
    "front-left":["front-left", "front-right", "behind-left", "behind-right"],
    "front-right":["front-left", "front-right", "behind-left", "behind-right"],
    "behind-left":["front-left", "front-right", "behind-left", "behind-right"],
    "behind-right":["front-left", "front-right", "behind-left", "behind-right"]
}

def get_sets(data_json, ground_json):
    no_description_objects = set()
    for cam_keys in data_json:
        for obj_keys in data_json[cam_keys]:
            # Check if description is empty (not description_difficulty, as objects can have
            # a description with just category name but empty marker array)
            description = data_json[cam_keys][obj_keys].get("description", "")
            if description == "" or description is None:
                no_description_objects.add(obj_keys)
    print("====== Objects with No Unique Descriptions ======")
    print(no_description_objects)

    cam_0_visible = set(data_json["camera_0_0"].keys())
    cam_0_visible_ground = set(ground_json["camera_0_0"].keys())
    cam_1_visible = set(data_json["camera_1_0"].keys())
    cam_1_visible_ground = set(ground_json["camera_1_0"].keys())

    common_objs_or = cam_0_visible.intersection(cam_1_visible)
    unique_agent1_or = cam_0_visible - cam_1_visible_ground
    unique_agent2_or = cam_1_visible - cam_0_visible_ground

    not_visible_agent1 = cam_0_visible_ground - cam_0_visible
    not_visible_agent2 = cam_1_visible_ground - cam_1_visible

    print("\n====== Not visible ======")
    print(f"Not visible to agent 1: {not_visible_agent1}")
    print(f"Not visible to agent 2: {not_visible_agent2}")
    avoid_set = set([get_category(item) for item in not_visible_agent1]).union(set([get_category(item) for item in not_visible_agent2]))
    avoid_set = avoid_set.union(no_description_objects)

    common_objs = set([item for item in common_objs_or if (get_category(item) not in avoid_set and item not in avoid_set)])
    unique_agent1 = set([item for item in unique_agent1_or if (get_category(item) not in avoid_set and item not in avoid_set)])
    unique_agent2 = set([item for item in unique_agent2_or if (get_category(item) not in avoid_set and item not in avoid_set)])

    print("\n====== Eliminating ======")
    print(f"Eliminating: {len(common_objs_or)-len(common_objs)} from Common Objects: {common_objs_or-common_objs}")
    print(f"Eliminating: {len(unique_agent1_or)-len(unique_agent1)} from Agent 1 Objects: {unique_agent1_or-unique_agent1}")
    print(f"Eliminating: {len(unique_agent2_or)-len(unique_agent2)} from Agent 2 Objects: {unique_agent2_or-unique_agent2}")

    print("\n====== Lengths ======")
    print(f"{len(common_objs)} in Common Objects")
    print(f"{len(unique_agent1)} in Agent 1 Objects")
    print(f"{len(unique_agent2)} in Agent 2 Objects\n")

    all_objs = common_objs.union(unique_agent1).union(unique_agent2)
    common_objs = set([obj for obj in list(common_objs) if get_category(obj) in allowed_categories])
    unique_agent1 = set([obj for obj in list(unique_agent1) if get_category(obj) in allowed_categories])
    unique_agent2 = set([obj for obj in list(unique_agent2) if get_category(obj) in allowed_categories])
    all_objs = set([obj for obj in list(all_objs) if get_category(obj) in allowed_categories])
    return common_objs, unique_agent1, unique_agent2, all_objs

def get_sets_counting(data_json, ground_json):
    # IGNORE DESCRIPTION UNIQUENESS IN COUNTING
    cam_0_visible = set(data_json["camera_0_0"].keys())
    cam_0_visible_ground = set(ground_json["camera_0_0"].keys())
    cam_1_visible = set(data_json["camera_1_0"].keys())
    cam_1_visible_ground = set(ground_json["camera_1_0"].keys())

    common_objs_or = cam_0_visible.intersection(cam_1_visible)
    unique_agent1_or = cam_0_visible - cam_1_visible_ground
    unique_agent2_or = cam_1_visible - cam_0_visible_ground

    not_visible_agent1 = cam_0_visible_ground - cam_0_visible
    not_visible_agent2 = cam_1_visible_ground - cam_1_visible

    print("\n====== Not visible ======")
    print(f"Not visible to agent 1: {not_visible_agent1}")
    print(f"Not visible to agent 2: {not_visible_agent2}")
    avoid_set = set([get_category(item) for item in not_visible_agent1]).union(set([get_category(item) for item in not_visible_agent2]))
    # avoid_set = set([get_category(item) for item in not_visible_agent1]).intersection(set([get_category(item) for item in not_visible_agent2]))

    common_objs = set([item for item in common_objs_or if (get_category(item) not in avoid_set and item not in avoid_set)])
    unique_agent1 = set([item for item in unique_agent1_or if (get_category(item) not in avoid_set and item not in avoid_set)])
    unique_agent2 = set([item for item in unique_agent2_or if (get_category(item) not in avoid_set and item not in avoid_set)])

    print("\n====== Eliminating ======")
    print(f"Eliminating: {len(common_objs_or)-len(common_objs)} from Common Objects: {common_objs_or-common_objs}")
    print(f"Eliminating: {len(unique_agent1_or)-len(unique_agent1)} from Agent 1 Objects: {unique_agent1_or-unique_agent1}")
    print(f"Eliminating: {len(unique_agent2_or)-len(unique_agent2)} from Agent 2 Objects: {unique_agent2_or-unique_agent2}")

    print("\n====== Lengths ======")
    print(f"{len(common_objs)} in Common Objects")
    print(f"{len(unique_agent1)} in Agent 1 Objects")
    print(f"{len(unique_agent2)} in Agent 2 Objects\n")

    all_objs = common_objs.union(unique_agent1).union(unique_agent2)
    common_objs = set([obj for obj in list(common_objs) if get_category(obj) in allowed_categories])
    unique_agent1 = set([obj for obj in list(unique_agent1) if get_category(obj) in allowed_categories])
    unique_agent2 = set([obj for obj in list(unique_agent2) if get_category(obj) in allowed_categories])
    all_objs = set([obj for obj in list(all_objs) if get_category(obj) in allowed_categories])
    return common_objs, unique_agent1, unique_agent2, all_objs

def get_descriptions(data_json):
    """
    Builds descriptions for all objects from the raw data.
    """
    desc_lookup = {}
    for cam in ["camera_0_0", "camera_1_0"]:
        for obj in data_json[cam]:
            description = data_json[cam][obj]["description"]
            if description == "":
                desc_lookup[obj] = None
            else:
                desc_lookup[obj] = description

    return desc_lookup

def get_description_difficulty(data_json):
    """
    Builds descriptions for all objects from the raw data.
    """
    desc_diff_lookup = {}
    for cam in ["camera_0_0", "camera_1_0"]:
        for obj in data_json[cam]:
            description = data_json[cam][obj]["description"]
            description_diff = data_json[cam][obj]["description_difficulty"]
            if description == "":
                desc_diff_lookup[obj] = 0
            else:
                desc_diff_lookup[obj] = description_diff

    return desc_diff_lookup

def get_ground_truth_counts(data_json):
    cam_0_visible = set(data_json["camera_0_0"].keys())
    cam_1_visible = set(data_json["camera_1_0"].keys())
    set_union = cam_0_visible.union(cam_1_visible)
    counts = {}
    for obj in set_union:
        key = get_category(obj)
        counts[key] = counts.get(key, 0) + 1
    return counts

def generate_counting_questions(data_json, ground_json, num_options):
    """
    Generates counting questions based on object occurrences in the scene.
    """
    common_objs, unique_agent1, unique_agent2, all_objs = get_sets_counting(data_json, ground_json)
    cam_0_visible = unique_agent1.union(common_objs)
    cam_1_visible = unique_agent2.union(common_objs)

    counts = {}
    difficulty1 = {}
    difficulty2 = {}
    ground_truth_counts = get_ground_truth_counts(ground_json)

    for obj in all_objs:
        key = get_category(obj)
        counts[key] = counts.get(key, 0) + 1
        if obj in cam_0_visible:
            difficulty1[key] = difficulty1.get(key, 0) + 1
        if obj in cam_1_visible:
            difficulty1[key] = difficulty1.get(key, 0) + 1
        if obj in cam_0_visible and obj in cam_1_visible:
            difficulty2[key] = difficulty2.get(key, 0) + 1

    questions = []

    for key, count in counts.items():
        if counts[key] != ground_truth_counts[key]:
            print(f"Counts not matched!: {key}")
            continue

        options = set()
        options.add(count)
        difficulty_val = difficulty1.get(key, 1)
        if difficulty_val not in options:
            options.add(difficulty_val)

        possible_undercounts = [i for i in range(1, count) if i != difficulty_val]
        if possible_undercounts:
            undercount = random.choice(possible_undercounts)
            options.add(undercount)

        possible_overcounts = [i for i in range(count + 1, max(difficulty_val+1, 5)) if i != difficulty_val]
        if possible_overcounts:
            overcount = random.choice(possible_overcounts)
            options.add(overcount)

        while len(options) < num_options:
            rand_val = random.randint(1, max(count, difficulty_val, 4))
            options.add(rand_val)

        options = list(options)
        random.shuffle(options)
        correct_answer_index = options.index(count)
        prob = random.random()
        asking_to = "agent_1" if prob<0.5 else "agent_2"
        q = {
            "question": f"What is the total number of {key} in the room?",
            "question_both_views": f"What is the total number of {key} in the room?",
            "asking_to":asking_to,
            "options": [str(opt) for opt in options],
            "correct_index": correct_answer_index,
            "difficulty_sum": difficulty1.get(key, 0),
            "difficulty_int": difficulty2.get(key, 0),
            "question_object": get_category(key),
            "scene_intersection": INTERSECTION_OBJS,
            "scene_union": UNION_OBJS
        }
        questions.append(q)

    return questions

def generate_anchor_questions(data_json, ground_json, num_options, num_buckets):
    desc_lookup = get_descriptions(data_json)
    desc_diff_lookup = get_description_difficulty(data_json)
    bbox_lookup = {}
    for cam in ["camera_0_0", "camera_1_0"]:
        for obj in data_json[cam]:
            bbox_lookup.setdefault(obj, []).append(data_json[cam][obj]["bbox_2d"])

    common_objs, unique_agent1, unique_agent2, all_objs = get_sets(data_json, ground_json)
    questions = []

    def compute_visual_difficulty(obj):
        bboxes = bbox_lookup.get(obj, [])
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        avg_area = sum(areas) / len(areas) if areas else 0.0
        return avg_area

    def make_question(anchor_obj, asking_to, distractors_self, distractors_other):
        cat_anchor = get_category(anchor_obj)
        same_cat_distractors = [o for o in (distractors_self + distractors_other) if get_category(o) == cat_anchor and o != anchor_obj]
        special_possible = (
            len(distractors_self) > 0
            and len(distractors_other) > 0
            and len(same_cat_distractors) > 0
        )

        if special_possible and num_options >= 4:
            try:
                d1 = random.choice(distractors_self)
                d2 = random.choice(distractors_other)
                filtered_same_cat = [d for d in same_cat_distractors if d not in {d1, d2}]
                if filtered_same_cat:
                    d3 = random.choice(filtered_same_cat)
                    question_type = "hard_negative"
                else:
                    remaining = [d for d in (distractors_self + distractors_other) if d not in {d1, d2}]
                    if remaining:
                        d3 = random.choice(remaining)
                        question_type = "one_random"
                    else:
                        raise IndexError
                total_distractors = [d1, d2, d3]
            except IndexError:
                # fallback if any list is empty during sampling
                total_distractors = distractors_self + distractors_other
                random.shuffle(total_distractors)
                total_distractors = total_distractors[: num_options - 1]
                question_type = "all_random"
        else:
            # fallback: old random method
            total_distractors = distractors_self + distractors_other
            random.shuffle(total_distractors)
            total_distractors = total_distractors[: num_options - 1]
            question_type = "all_random"

        cross_view = [d for d in total_distractors if d in distractors_other]

        correct_idx = random.randrange(num_options)
        options = [None] * num_options
        options[correct_idx] = anchor_obj

        for i, d in zip([i for i in range(num_options) if i != correct_idx], total_distractors):
            options[i] = d

        if None in options:
            return None

        q = {
            "question": "Which of the following objects is visible in both your and your partner's views of the room?",
            "question_both_views": "Which of the following objects is visible in both views of the room?",
            "asking_to": asking_to,
            "options": [desc_lookup.get(o, o) for o in options],
            "option_categories": [get_category(o) for o in options],
            "correct_index": correct_idx,
            "difficulty": compute_visual_difficulty(anchor_obj),
            "description_difficulty": [desc_diff_lookup.get(o, 0) for o in options],
            "distractor_difficulty": cross_view,  # number of cross-view distractors
            "question_type": question_type,  # new key
            "scene_intersection": INTERSECTION_OBJS,
            "scene_union": UNION_OBJS
        }

        return q

    for common in common_objs:
        q1 = make_question(common, "agent_1", list(unique_agent1), list(unique_agent2))
        q2 = make_question(common, "agent_2", list(unique_agent2), list(unique_agent1))

        if q1:
            questions.append(q1)
        if q2:
            questions.append(q2)

    return questions

def generate_relative_distance_questions(data_json, ground_json, dist_threshold):
    """
    Generates both 'closest' and 'farthest' relative distance questions.
    Each question asks which object is closest/farthest to a common anchor.
    The correct-wrong distance gap must exceed the given threshold.
    Distance is computed based on the closest bounding box corners.
    """

    # Step 1: Get object sets
    common_objs, unique_agent1, unique_agent2, all_objs = get_sets(data_json, ground_json)

    # Step 2: Prepare lookups
    corners_lookup = {}
    desc_lookup = get_descriptions(data_json)
    desc_diff_lookup = get_description_difficulty(data_json)
    used_option_sets = set()

    # Step 3: Store all 3D corners for distance computations
    for cam in ["camera_0_0", "camera_1_0"]:
        for obj in data_json[cam]:
            corners_lookup[obj] = np.array(data_json[cam][obj]["bbox_3d_corners"])

    questions = []

    for anchor in common_objs:
        question_description_markers = data_json["camera_0_0"][anchor]['description_difficulty']
        if 2 in question_description_markers:
            continue
        question_color = data_json["camera_0_0"][anchor]['color'] if 0 in question_description_markers else None
        question_object_category = get_category(anchor)

        if len(unique_agent1) < 1 or len(unique_agent2) < 1:
            continue

        def is_option_valid(opt_obj, cam_key):
            opt_description_markers = data_json[cam_key][opt_obj].get('description_difficulty', [])
            if 2 not in opt_description_markers:
                return True
            opt_chosen_neighbor = data_json[cam_key][opt_obj].get('chosen_neighbor')
            if not opt_chosen_neighbor:
                return True
            opt_neighbor_color, opt_neighbor_category = parse_chosen_neighbor(opt_chosen_neighbor)
            if question_object_category is None:
                return True
            if opt_neighbor_category != question_object_category:
                return True
            if question_color is None:
                return True
            if question_color is not None and opt_neighbor_color != question_color:
                return True
            return False

        # Filter unique_agent1 and unique_agent2 to only include valid options
        unique_agent1_filtered = [obj for obj in unique_agent1 if is_option_valid(obj, "camera_0_0")]
        unique_agent2_filtered = [obj for obj in unique_agent2 if is_option_valid(obj, "camera_1_0")]

        if len(unique_agent1_filtered) < 1 or len(unique_agent2_filtered) < 1:
            continue

        for _ in range(50):
            # Determine which distribution is possible
            possible_distributions = []
            if len(unique_agent1_filtered) >= 2 and len(unique_agent2_filtered) >= 2:
                possible_distributions.append((2, 2))
            if len(unique_agent1_filtered) >= 1 and len(unique_agent2_filtered) >= 3:
                possible_distributions.append((1, 3))
            if len(unique_agent1_filtered) >= 3 and len(unique_agent2_filtered) >= 1:
                possible_distributions.append((3, 1))

            if not possible_distributions:
                continue

            # Randomly choose a distribution
            agent1_count, agent2_count = random.choice(possible_distributions)
            agent_distribution = f"{agent1_count}-{agent2_count}"

            # Sample according to chosen distribution
            opts_agent1 = random.sample(unique_agent1_filtered, agent1_count)
            opts_agent2 = random.sample(unique_agent2_filtered, agent2_count)
            options = opts_agent1 + opts_agent2
            option_set = frozenset(options)
            if option_set in used_option_sets:
                continue
            used_option_sets.add(option_set)

            if any(o not in corners_lookup or anchor not in corners_lookup for o in options):
                continue

            # Compute distances
            anchor_corners = corners_lookup[anchor]
            dists = {}
            for o in options:
                obj_corners = corners_lookup[o]
                diff = anchor_corners[:, None, :] - obj_corners[None, :, :]
                pairwise_dists = np.linalg.norm(diff, axis=-1)
                dists[o] = pairwise_dists.min()

            # ---------- Closest Question ----------
            sorted_opts = sorted(dists.items(), key=lambda x: x[1])
            correct_obj, correct_dist = sorted_opts[0]
            second_obj, second_dist = sorted_opts[1]
            if (second_dist - correct_dist) >= dist_threshold:
                options_closest = options.copy()
                random.shuffle(options_closest)
                correct_index = options_closest.index(correct_obj)
                prob = random.random()
                asking_to = "agent_1" if prob < 0.5 else "agent_2"
                visible_set = unique_agent1 if asking_to == "agent_1" else unique_agent2
                ans_present_in_view = correct_obj in visible_set

                q = {
                    "question": f"Which of the following objects is closest to the {desc_lookup.get(anchor, anchor)}?",
                    "question_both_views": f"Which of the following objects is closest to the {desc_lookup.get(anchor, anchor)}?",
                    "asking_to": asking_to,
                    "options": [desc_lookup.get(o, o) for o in options_closest],
                    "option_distances": [dists[o] for o in options_closest],
                    "option_categories": [get_category(o) for o in options_closest],
                    "correct_index": correct_index,
                    "difficulty": round(second_dist - correct_dist, 3),
                    "description_difficulty": desc_diff_lookup.get(anchor, None),
                    "question_object": get_category(anchor),
                    "ans_present_in_view": ans_present_in_view,
                    "question_type": "closest",
                    "agent_distribution": agent_distribution,
                    "scene_intersection": INTERSECTION_OBJS,
                    "scene_union": UNION_OBJS
                }
                questions.append(q)

            # ---------- Farthest Question ----------
            sorted_opts_far = sorted(dists.items(), key=lambda x: x[1], reverse=True)
            correct_obj_far, correct_dist_far = sorted_opts_far[0]
            second_obj_far, second_dist_far = sorted_opts_far[1]
            if (correct_dist_far - second_dist_far) >= dist_threshold:
                options_far = options.copy()
                random.shuffle(options_far)
                correct_index_far = options_far.index(correct_obj_far)
                prob = random.random()
                asking_to = "agent_1" if prob < 0.5 else "agent_2"
                visible_set = unique_agent1 if asking_to == "agent_1" else unique_agent2
                ans_present_in_view = correct_obj_far in visible_set

                q_far = {
                    "question": f"Which of the following objects is farthest from the {desc_lookup.get(anchor, anchor)}?",
                    "question_both_views": f"Which of the following objects is farthest from the {desc_lookup.get(anchor, anchor)}?",
                    "asking_to": asking_to,
                    "options": [desc_lookup.get(o, o) for o in options_far],
                    "option_distances": [dists[o] for o in options_far],
                    "option_categories": [get_category(o) for o in options_far],
                    "correct_index": correct_index_far,
                    "difficulty": round(correct_dist_far - second_dist_far, 3),
                    "description_difficulty": desc_diff_lookup.get(anchor, None),
                    "question_object": get_category(anchor),
                    "ans_present_in_view": ans_present_in_view,
                    "question_type": "farthest",
                    "agent_distribution": agent_distribution,
                    "scene_intersection": INTERSECTION_OBJS,
                    "scene_union": UNION_OBJS
                }
                questions.append(q_far)

    return questions

def get_orientation(p_world, T, tolerance=22.5): #22.5 #15 #12.5
    p_world_h = np.ones(4)
    p_world_h[:3] = p_world
    p_cam = T @ p_world_h
    x, z = p_cam[0], p_cam[2]

    angle = np.degrees(np.arctan2(x, -z))
    angle = (angle + 360) % 360
    direction_angles = {
        "front": 0,
        "front-right": 45,
        "right": 90,
        "behind-right": 135,
        "behind": 180,
        "behind-left": 225,
        "left": 270,
        "front-left": 315
    }

    # print("\n====== Marking Angles ======")
    # for key in direction_angles:
    #     angle = direction_angles[key]
    #     min_angle = (360 + angle - tolerance)%360
    #     max_angle = (angle + tolerance)%360
    #     print(f"Labelling: {key} as {min_angle} to {max_angle}")

    for label, ref in direction_angles.items():
        diff = min(abs(angle - ref), 360 - abs(angle - ref))
        if diff <= tolerance:
            return (angle, label)
    return (angle, None)

def generate_spatial_orientation_questions(data_json, ground_json, cam_data_path, perspective=False):
    """
    Generates spatial orientation questions from a single agent's perspective
    or for perspective taking.
    """
    common_objs, unique_agent1, unique_agent2, all_objs = get_sets(data_json, ground_json)

    desc_lookup = get_descriptions(data_json)
    desc_diff_lookup = get_description_difficulty(data_json)

    loc_lookup = {}
    for cam in ["camera_0_0", "camera_1_0"]:
        for obj in data_json[cam]:
            if "bb" in obj:
                center = data_json[cam][obj]["location"]
            else:
                corners = np.array(data_json[cam][obj]["bbox_3d_corners"])
                center = corners.mean(axis=0).tolist()
            loc_lookup[obj] = center

    with open(cam_data_path, "r") as f:
        all_cams = json.load(f)

    T1 = np.array(all_cams["camera_0_0"]["T"])
    T2 = np.array(all_cams["camera_1_0"]["T"])

    R1 = T1[:3, :3]
    t1 = T1[:3, 3]
    cam_pos1 = -R1.T @ t1

    R2 = T2[:3, :3]
    t2 = T2[:3, 3]
    cam_pos2 = -R2.T @ t2

    ORIENTATION_LABELS = ["front-left", "front", "front-right", "right", "behind-right", "behind", "behind-left", "left"]
    DISTANCE_THRESHOLD = 0.0 #2.5
    questions = []

    if not perspective:
        for obj in unique_agent2:
            angle1, orientation1 = get_orientation(loc_lookup[obj], T1)
            angle2, orientation2 = get_orientation(loc_lookup[obj], T2)

            obj_center = np.array(loc_lookup[obj])
            dist1 = np.linalg.norm(obj_center - cam_pos1)
            dist2 = np.linalg.norm(obj_center - cam_pos2)

            if dist1 < DISTANCE_THRESHOLD:
                print(f"Skipping {obj} — too close to camera_0_0 ({dist1:.2f}m)")
                continue

            if orientation1 is None:
                continue
            options = options_dict[orientation1].copy()
            random.shuffle(options)
            q = {
                "question": f"From your perspective, in which direction is {desc_lookup.get(obj, obj)}?",
                "question_both_views": f"From the viewpoint of the first image, in which direction is {desc_lookup.get(obj, obj)}?",
                "options": options,
                "correct_index": options.index(orientation1),
                "difficulty": 1 if orientation1 in ["front", "left", "right", "behind"] else 2,
                "angle": angle1,
                "description_difficulty":desc_diff_lookup.get(obj, None),
                "distance": dist1,
                "asking_to": "agent_1",
                "question_object": get_category(obj),
                "other_agent_angle": angle2,
                "other_agent_distance": dist2,
                "scene_intersection": INTERSECTION_OBJS,
                "scene_union": UNION_OBJS
            }
            questions.append(q)

        for obj in unique_agent1:
            angle1, orientation1 = get_orientation(loc_lookup[obj], T1)
            angle2, orientation2 = get_orientation(loc_lookup[obj], T2)

            obj_center = np.array(loc_lookup[obj])
            dist1 = np.linalg.norm(obj_center - cam_pos1)
            dist2 = np.linalg.norm(obj_center - cam_pos2)

            if dist2 < DISTANCE_THRESHOLD:
                print(f"Skipping {obj} — too close to camera_0_0 ({dist2:.2f}m)")
                continue

            if orientation2 is None:
                continue
            options = options_dict[orientation2].copy()
            random.shuffle(options)
            q = {
                "question": f"From your perspective, in which direction is {desc_lookup.get(obj, obj)}?",
                "question_both_views": f"From the viewpoint of the second image, in which direction is {desc_lookup.get(obj, obj)}?",
                "options": options,
                "correct_index": options.index(orientation2),
                "difficulty": 1 if orientation2 in ["front", "left", "right", "behind"] else 2,
                "angle": angle2,
                "description_difficulty":desc_diff_lookup.get(obj, None),
                "distance": dist2,
                "asking_to": "agent_2",
                "question_object": get_category(obj),
                "other_agent_angle": angle1,
                "other_agent_distance": dist1,
                "scene_intersection": INTERSECTION_OBJS,
                "scene_union": UNION_OBJS
            }
            questions.append(q)

    else:
        for obj in unique_agent1:
            angle1, orientation1 = get_orientation(loc_lookup[obj], T1)
            angle2, orientation2 = get_orientation(loc_lookup[obj], T2)

            obj_center = np.array(loc_lookup[obj])
            dist1 = np.linalg.norm(obj_center - cam_pos1)
            dist2 = np.linalg.norm(obj_center - cam_pos2)

            if dist2 < DISTANCE_THRESHOLD:
                print(f"Skipping {obj} — too close to camera_0_0 ({dist2:.2f}m)")
                continue

            if orientation2 is None:
                continue
            options = options_dict[orientation2].copy()
            random.shuffle(options)

            q = {
                "question": f"From your partner's perspective, in which direction is {desc_lookup.get(obj, obj)}?",
                "question_both_views": f"From the viewpoint of the second image, in which direction is {desc_lookup.get(obj, obj)}?",
                "options": options,
                "correct_index": options.index(orientation2),
                "difficulty": 1 if orientation2 in ["front", "left", "right", "behind"] else 2,
                "angle": angle2,
                "description_difficulty":desc_diff_lookup.get(obj, None),
                "distance": dist2,
                "asking_to": "agent_1",
                "question_object": get_category(obj),
                "other_agent_angle": angle1,
                "other_agent_distance": dist1,
                "scene_intersection": INTERSECTION_OBJS,
                "scene_union": UNION_OBJS
            }
            questions.append(q)

        for obj in unique_agent2:
            angle1, orientation1 = get_orientation(loc_lookup[obj], T1)
            angle2, orientation2 = get_orientation(loc_lookup[obj], T2)

            obj_center = np.array(loc_lookup[obj])
            dist1 = np.linalg.norm(obj_center - cam_pos1)
            dist2 = np.linalg.norm(obj_center - cam_pos2)

            if dist1 < DISTANCE_THRESHOLD:
                print(f"Skipping {obj} — too close to camera_0_0 ({dist1:.2f}m)")
                continue

            if orientation1 is None:
                continue

            # Remove neighbors (-1, +1 in circular list)
            options = options_dict[orientation1].copy()
            random.shuffle(options)
            q = {
                "question": f"From your partner's perspective, in which direction is {desc_lookup.get(obj, obj)}?",
                "question_both_views": f"From the viewpoint of the first image, in which direction is {desc_lookup.get(obj, obj)}?",
                "options": options,
                "correct_index": options.index(orientation1),
                "difficulty": 1 if orientation1 in ["front", "left", "right", "behind"] else 2,
                "angle": angle1,
                "description_difficulty":desc_diff_lookup.get(obj, None),
                "distance": dist1,
                "asking_to": "agent_2",
                "question_object": get_category(obj),
                "other_agent_angle": angle2,
                "other_agent_distance": dist2,
                "scene_intersection": INTERSECTION_OBJS,
                "scene_union": UNION_OBJS
            }
            questions.append(q)

    return questions

def main():
    parser = argparse.ArgumentParser(
        description="Generate quiz questions from scene data."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="./visible_objects_with_descriptions.json",
        help="Path to the visible objects JSON file.",
    )
    parser.add_argument(
        "--ground_truth_json",
        type=str,
        default="./visible_objects.json",
        help="Path to the visible objects JSON file.",
    )
    parser.add_argument(
        "--cam_data_file",
        type=str,
        default="./cameras.json",
        help="Path to the camera data JSON file.",
    )
    parser.add_argument(
        "--num_options",
        type=int,
        default=4,
        help="Number of options for each multiple-choice question.",
    )
    parser.add_argument(
        "--num_buckets",
        type=int,
        default=100,
        help="Number of buckets for anchor question difficulty.",
    )
    parser.add_argument(
        "--dist_threshold",
        type=float,
        default=0.7, #1.0,
        help="Distance threshold for relative distance questions.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="questions.json",
        help="Path to the output JSON file."
    )
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input_json, "r") as f:
        data = json.load(f)
    with open(args.ground_truth_json, "r") as f:
        ground_data = json.load(f)
    global INTERSECTION_OBJS
    INTERSECTION_OBJS = len(set(ground_data["camera_0_0"].keys()).intersection(set(ground_data["camera_1_0"].keys())))
    global UNION_OBJS
    UNION_OBJS = len(set(ground_data["camera_0_0"].keys()).union(set(ground_data["camera_1_0"].keys())))

    questions_output = {
        "counting_questions": generate_counting_questions(data, ground_data, args.num_options),
        "anchor_questions": generate_anchor_questions(
            data, ground_data, args.num_options, args.num_buckets
        ),
        "relative_distance_questions": generate_relative_distance_questions(
            data, ground_data, args.dist_threshold
        ),
        "spatial_orientation_questions": generate_spatial_orientation_questions(
            data, ground_data, args.cam_data_file, perspective=False
        ),
        "perspective_taking_questions": generate_spatial_orientation_questions(
            data, ground_data, args.cam_data_file, perspective=True
        ),
    }

    with open(args.output_json, "w") as f:
        json.dump(questions_output, f, indent=2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error occurred: {e}")