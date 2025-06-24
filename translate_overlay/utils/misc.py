import os
from math import ceil, log, log2

import numpy as np
from PIL import ImageDraw, Image
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

from .logger import setup_logger


logger = setup_logger()


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


def group_boxes_to_paragraphs(text_box_list, eps=0.2, gap_ratio=0.5, x_align_thresh=10):

    def normalize(value, threshold=10):
        return round(value / threshold) * threshold

    boxes_xyxy = [box for _, box in text_box_list]
    if not boxes_xyxy:
        return []

    height_list = [i[3] - i[1] for i in boxes_xyxy]
    x_mins = [box[0] for box in boxes_xyxy]
    x_maxs = [box[2] for box in boxes_xyxy]
    y_mins = [box[1] for box in boxes_xyxy]
    y_maxs = [box[3] for box in boxes_xyxy]

    width_range = max(x_maxs) - min(x_mins)
    height_range = max(y_maxs) - min(y_mins)
    min_height = min(height_list)
    min_x = min(x_mins)
    min_y = min(y_mins)

    box_features = []
    for (x_min, y_min, x_max, y_max) in boxes_xyxy:
        height = y_max - y_min
        y_center = y_min + height / 2
        box_features.append([
            # (normalize(x_min, x_align_thresh) - min_x) / width_range * 1.0,   # left alignment, normalized
            (x_min - min_x) / width_range * 1.0,   # left alignment, normalized
            (y_center - min_y) / height_range * 1.0, # vertical position, normalized
            # (height - min_height) / height_range * 1.0 # height, normalized
        ])
    box_features = np.array(box_features)

    dist_matrix = squareform(pdist(box_features, metric='euclidean'))
    for i in dist_matrix.tolist():
        print([f"{j:.5}".ljust(7) for j in i])

    distance_threshold = min(0.3, 1 / log(max(1, len(boxes_xyxy))))
    logger.info(f"Distance threshold: {distance_threshold}")
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold,
        linkage="ward"
    ).fit(box_features)

    paragraph_groups = []
    original_groups = []
    for label in set(clustering.labels_):
        indices = np.where(clustering.labels_ == label)[0]
        group = [text_box_list[i] for i in indices]
        # Sort group by y_min
        group = sorted(group, key=lambda tb: tb[1][1])
        original_groups.append(group)  # Save original group

        # Post-process: split group if gap too large or left alignment differs
        current_para = [group[0]]
        for prev, curr in zip(group, group[1:]):
            prev_box = prev[1]
            curr_box = curr[1]
            prev_height = prev_box[3] - prev_box[1]
            curr_height = curr_box[3] - curr_box[1]
            min_h = min(prev_height, curr_height)
            vertical_gap = curr_box[1] - prev_box[3]

            # Normalized x_min for left alignment
            prev_x_norm = prev_box[0]
            curr_x_norm = curr_box[0]

            # Split if vertical gap too large or left alignment differs
            if vertical_gap > gap_ratio * min_h or x_align_thresh < abs(prev_x_norm - curr_x_norm):
                paragraph_groups.append(current_para)
                current_para = [curr]
            else:
                current_para.append(curr)

        # Add last paragraph in group
        if current_para:
            paragraph_groups.append(current_para)

    logger.info(f"Num cluster labels: {len(set(clustering.labels_))}")
    logger.info(f"Num final groups: {len(paragraph_groups)}")

    return paragraph_groups, original_groups


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




if __name__ == "__main__":
    import sys
    # Example usage
    image = Image.open(r"E:\work\1-personal\Florence-2-base\images\Screenshot_2025-06-10_172012.png")
    boxes_xyxy = [[10, 50, 139, 71], [48, 1040, 75, 1053], [48, 1054, 75, 1069], [74, 14, 123, 29], [296, 12, 481, 31], [302, 52, 619, 74], [384, 300, 1071, 343], [386, 348, 699, 385], [386, 399, 507, 420], [386, 438, 1123, 467], [386, 472, 1107, 495], [386, 500, 1109, 529], [386, 532, 801, 555], [388, 664, 941, 699], [386, 724, 1117, 747], [386, 754, 1125, 779], [386, 786, 1121, 809], [386, 816, 781, 839], [386, 872, 677, 905], [386, 928, 1127, 953], [386, 958, 1139, 983], [386, 990, 1073, 1013], [396, 602, 557, 625], [641, 208, 688, 226], [720, 206, 839, 225], [873, 204, 934, 228], [882, 122, 1037, 149], [970, 206, 1037, 225], [1072, 208, 1149, 225], [1184, 206, 1277, 225], [1226, 426, 1353, 447], [1226, 468, 1529, 493], [1226, 500, 1397, 521], [1226, 538, 1485, 563], [1226, 570, 1461, 593], [1226, 608, 1501, 633], [1226, 640, 1397, 661], [1226, 678, 1515, 701], [1226, 707, 1531, 733], [1226, 748, 1515, 769], [1226, 778, 1413, 802], [1226, 818, 1503, 841], [1226, 848, 1477, 870], [1242, 310, 1297, 329], [1480, 52, 1525, 71], [1480, 310, 1493, 323], [1658, 44, 1905, 71], [1746, 1044, 1829, 1065], [1844, 1056, 1903, 1071], [1856, 1040, 1903, 1053]]

    # image = Image.open(r"E:\work\1-personal\Florence-2-base\images\Screenshot_2025-06-14_171710.png")
    # boxes_xyxy = [[97, 27, 189, 42], [100, 71, 389, 89], [99, 145, 308, 164], [99, 167, 304, 184], [100, 251, 482, 278], [99, 285, 750, 302], [98, 304, 745, 323], [99, 325, 737, 342], [99, 632, 568, 656], [99, 663, 723, 681], [98, 684, 496, 702], [99, 781, 708, 805], [99, 813, 733, 830], [99, 835, 749, 852], [99, 856, 746, 873], [113, 370, 369, 390], [114, 393, 193, 410], [113, 444, 349, 461], [113, 463, 362, 484], [113, 492, 360, 510], [113, 513, 364, 532], [123, 112, 236, 129], [115, 351, 202, 368], [115, 421, 354, 440], [115, 535, 341, 552], [139, 205, 205, 219], [138, 221, 349, 238], [139, 584, 205, 600], [139, 601, 652, 618], [139, 733, 205, 749], [138, 750, 637, 769], [307, 111, 350, 131], [383, 112, 540, 128], [413, 351, 486, 369], [413, 370, 654, 391], [413, 393, 493, 410], [413, 421, 588, 437], [413, 444, 610, 458], [413, 464, 662, 483], [414, 492, 653, 511], [413, 513, 628, 531], [415, 535, 650, 550], [561, 112, 620, 130], [583, 71, 625, 86], [603, 632, 712, 654], [664, 71, 717, 86]]

    # image = Image.open(r"E:\work\1-personal\Florence-2-base\images\Screenshot_2025-05-18_172138.png")
    # boxes_xyxy = [[10, 50, 139, 71], [48, 1040, 75, 1053], [46, 1052, 61, 1069], [74, 14, 123, 29], [204, 802, 377, 821], [214, 190, 597, 207], [206, 274, 277, 289], [214, 304, 519, 328], [206, 454, 305, 469], [214, 518, 525, 542], [214, 690, 523, 709], [214, 724, 523, 745], [214, 836, 527, 855], [214, 904, 389, 923], [216, 234, 357, 249], [216, 344, 231, 357], [218, 378, 231, 393], [216, 408, 523, 427], [218, 490, 231, 503], [216, 590, 231, 605], [216, 626, 231, 639], [218, 660, 231, 673], [218, 762, 229, 775], [218, 874, 231, 887], [218, 1010, 231, 1023], [240, 106, 351, 129], [244, 342, 405, 361], [244, 486, 523, 507], [244, 590, 585, 609], [244, 622, 497, 641], [244, 654, 581, 678], [244, 760, 361, 777], [244, 870, 575, 891], [244, 940, 339, 957], [244, 974, 521, 991], [246, 376, 431, 395], [246, 558, 615, 575], [246, 1006, 601, 1027], [298, 14, 425, 31], [302, 52, 679, 75], [362, 942, 501, 957], [382, 106, 621, 125], [440, 908, 511, 923], [454, 378, 591, 393], [524, 14, 707, 29], [662, 186, 773, 205], [670, 676, 835, 695], [670, 758, 911, 777], [672, 1002, 799, 1021], [692, 246, 885, 265], [692, 272, 1067, 289], [692, 324, 1039, 347], [690, 350, 1047, 369], [690, 407, 1047, 430], [690, 432, 1011, 452], [692, 490, 913, 511], [692, 516, 1029, 533], [690, 572, 857, 591], [692, 598, 965, 615], [690, 652, 931, 673], [692, 734, 903, 755], [690, 816, 1083, 835], [692, 842, 1009, 859], [692, 896, 1083, 919], [692, 922, 1017, 943], [692, 976, 1029, 1000], [732, 14, 933, 31], [846, 186, 875, 203], [910, 1040, 1057, 1065], [956, 14, 969, 27], [1018, 106, 1071, 125], [1114, 108, 1269, 125], [1204, 676, 1583, 695], [1204, 1002, 1333, 1021], [1208, 354, 1219, 367], [1206, 410, 1445, 429], [1208, 434, 1219, 447], [1208, 516, 1547, 533], [1206, 598, 1375, 613], [1208, 760, 1219, 773], [1206, 816, 1509, 835], [1206, 842, 1219, 855], [1206, 924, 1337, 941], [1224, 246, 1419, 265], [1222, 267, 1563, 290], [1224, 326, 1445, 349], [1222, 350, 1569, 369], [1222, 432, 1543, 452], [1224, 488, 1689, 511], [1224, 572, 1383, 593], [1224, 654, 1455, 673], [1224, 732, 1455, 756], [1222, 758, 1551, 777], [1224, 842, 1527, 859], [1224, 898, 1355, 919], [1224, 980, 1571, 1001], [1310, 108, 1351, 125], [1394, 108, 1429, 125], [1448, 104, 1613, 128], [1468, 186, 1563, 203], [1510, 58, 1523, 71], [1591, 185, 1701, 206], [1660, 56, 1675, 69], [1693, 43, 1903, 72], [1744, 1042, 1827, 1065], [1844, 1056, 1903, 1071], [1856, 1040, 1903, 1053]]


    # Test box merge
    text_box_list = [("", i) for i in boxes_xyxy]
    merged_boxes_xyxy = []

    grouped_boxes_xyxy = group_boxes_to_paragraphs(text_box_list)
    for i in grouped_boxes_xyxy:
        box_list = [j[1] for j in i]
        box_list_transposed = list(zip(*box_list))

        merged_boxes_xyxy.append((
            min(box_list_transposed[0]), 
            min(box_list_transposed[1]), 
            max(box_list_transposed[2]), 
            max(box_list_transposed[3]), 
        ))
    # End test box merge
    print(merged_boxes_xyxy)


    image = draw_boxes(image, boxes_xyxy, "yellow")
    image = draw_boxes(image, merged_boxes_xyxy, "red")

    image.show()

