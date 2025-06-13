from math import ceil

import numpy as np
from PIL import ImageDraw, Image
# from sklearn.cluster import DBSCAN


def draw_boxes(input_image, boxes_xyxy, outline="red"):
    new_image = ImageDraw.Draw(input_image)

    for idx, text_box in enumerate(boxes_xyxy):
        new_image.rectangle(((text_box[0], text_box[1]), (text_box[2], text_box[3])), outline=outline)

    return input_image


def pad_box(box_xyxy, pad_size, image_width, image_height):
    return (
        min(0, box_xyxy[0]-pad_size),
        min(0, box_xyxy[1]-pad_size),
        max(image_width, box_xyxy[2]+pad_size),
        max(image_height, box_xyxy[3]+pad_size),
    )


def xyxyxyxy_to_xyxy(input_bound):
    return (
        min(input_bound[::2]),
        min(input_bound[1::2]),
        max(input_bound[::2]),
        max(input_bound[1::2]),
    )


def merge_text_images(image_box_list, max_width=768, gap=3):
    merged_image_dict = dict()
    merged_image_index = 0
    image_template = None

    for image, box_xyxy in sorted(image_box_list, key=lambda x: x[0].width, reverse=True):
        if image_template is not None:
            if y_offset + gap + image.height > image_template.height:
                merged_image_dict[merged_image_index]["merged_image"] = image_template
                image_template = None
            
            else:
                y_offset += gap
                draw.line((
                    (0, int(y_offset - gap/2)), 
                    (merged_image_width, int(y_offset - gap/2))
                ), fill="red")

        if image_template is None:
            y_offset = 0

            merged_image_width = max(max(image.width, image.height), max_width)
            image_template = Image.new(image.mode, (merged_image_width, merged_image_width), (0, 0, 0))
            draw = ImageDraw.Draw(image_template) 
            merged_image_index += 1
            merged_image_dict[merged_image_index] = {"merged_image": None, "trd_box_list": list(), "merged_box_list": list()}

        image_template.paste(image, (0, y_offset))
        merged_image_dict[merged_image_index]["trd_box_list"].append(box_xyxy)
        merged_image_dict[merged_image_index]["merged_box_list"].append((0, y_offset, box_xyxy[2], y_offset+image.height))
        y_offset += image.height

    merged_image_dict[merged_image_index]["merged_image"] = image_template
    merged_image_dict[merged_image_index]["merged_box_list"].append((0, y_offset, box_xyxy[2], y_offset+image.height))
    return merged_image_dict


def pad_image_to_square(input_image, background_color=(255, 255, 255)):
    bg_size = max(input_image.size)
    padded_image = Image.new("RGB", (bg_size, bg_size), background_color)

    x_offset = (bg_size - input_image.width) // 2
    y_offset = (bg_size - input_image.height) // 2

    padded_image.paste(input_image, (x_offset, y_offset))

    return padded_image, x_offset, y_offset


def chunk_image(image, max_width=768, max_height=768, overlap_size=100):
    assert overlap_size < min(max_height, max_width)

    image_width, image_height = image.size
    chunk_count_horizontal = ceil(image_width / max_width)
    chunk_count_vertical = ceil(image_height / max_width)
    chunk_size_horizontal = ceil((image_width - overlap_size) / chunk_count_horizontal) + overlap_size
    chunk_size_vertical = ceil((image_height - overlap_size) / chunk_count_vertical) + overlap_size

    cropped_image_list = []
    for i in range(chunk_count_vertical):
        for j in range(chunk_count_horizontal):
            cropped_image_list.append(image.crop((
                chunk_size_horizontal * j,
                chunk_size_vertical * i,
                chunk_size_horizontal * (j + 1) + overlap_size,
                chunk_size_vertical * (i + 1) + overlap_size,
            )))

    return cropped_image_list


# def group_boxes_to_paragraphs_dbscan(boxes_xyxy, eps=50, min_samples=1, x_weight=0.2):
#     """
#     Group line boxes into paragraphs using DBSCAN clustering.
#     eps: maximum distance between lines to be considered in the same paragraph (in pixels).
#     x_weight: scale factor for horizontal distance (smaller = less sensitive to x, larger = more sensitive).
#     Returns a list of merged paragraph boxes [x_min, y_min, x_max, y_max].
#     """
#     if not boxes_xyxy:
#         return []

#     # Use both y_center and x_center for clustering
#     box_features = []
#     for (x_min, y_min, x_max, y_max) in boxes_xyxy:
#         y_center = (y_min + y_max) / 2
#         x_center = (x_min + x_max) / 2
#         box_features.append([y_center, x_center * x_weight])
#     box_features = np.array(box_features)

#     clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(box_features)

#     paragraphs = []
#     for label in set(clustering.labels_):
#         indices = np.where(clustering.labels_ == label)[0]
#         group = [boxes_xyxy[i] for i in indices]
#         x_min = min(b[0] for b in group)
#         y_min = min(b[1] for b in group)
#         x_max = max(b[2] for b in group)
#         y_max = max(b[3] for b in group)
#         paragraphs.append([x_min, y_min, x_max, y_max])

#     return paragraphs


def is_mostly_inside(rect1, rect2, threshold=0.8):
    """Check if rect1 is mostly inside rect2 based on area overlap"""
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2

    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    rect1_area = (x1_max - x1_min) * (y1_max - y1_min)

    return inter_area / rect1_area >= threshold


def map_florence2_to_trd_result(florence2_result_list, trd_box_list, merged_box_list, box_overlap_ratio=0.8):
    florence2_resulty_list = [(text, xyxyxyxy_to_xyxy(box)) for text, box in sorted(florence2_result_list, key=lambda x: min(x[1][::2]))]

    result_dict = dict()
    for text, florence2_box_xyxy in florence2_resulty_list:
        result_used = False

        for trd_box_xyxy, merged_box_xyxy in zip(trd_box_list, merged_box_list):
            if any([
                florence2_box_xyxy[3] <= merged_box_xyxy[1],
                florence2_box_xyxy[1] >= merged_box_xyxy[3],
                not is_mostly_inside(florence2_box_xyxy, merged_box_xyxy, box_overlap_ratio),
            ]):
                continue

            trd_box_xyxy = tuple(trd_box_xyxy)
            result_used = True
            if trd_box_xyxy not in result_dict:
                result_dict[trd_box_xyxy] = list()

            result_dict[trd_box_xyxy].append(text)
            break

        if not result_used:
            print(f"Florence2 result not used: \nText: {text}\nBox: {florence2_box_xyxy}")

    result_list = [(' - '.join(text), box_xyxy) for box_xyxy, text in result_dict.items()]

    return result_list
