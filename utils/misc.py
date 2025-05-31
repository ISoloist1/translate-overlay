import difflib
from math import ceil

import numpy as np
from PIL import ImageDraw, Image
from sklearn.cluster import DBSCAN


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


def group_boxes_to_paragraphs_dbscan(boxes_xyxy, eps=50, min_samples=1, x_weight=0.2):
    """
    Group line boxes into paragraphs using DBSCAN clustering.
    eps: maximum distance between lines to be considered in the same paragraph (in pixels).
    x_weight: scale factor for horizontal distance (smaller = less sensitive to x, larger = more sensitive).
    Returns a list of merged paragraph boxes [x_min, y_min, x_max, y_max].
    """
    if not boxes_xyxy:
        return []

    # Use both y_center and x_center for clustering
    box_features = []
    for (x_min, y_min, x_max, y_max) in boxes_xyxy:
        y_center = (y_min + y_max) / 2
        x_center = (x_min + x_max) / 2
        box_features.append([y_center, x_center * x_weight])
    box_features = np.array(box_features)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(box_features)

    paragraphs = []
    for label in set(clustering.labels_):
        indices = np.where(clustering.labels_ == label)[0]
        group = [boxes_xyxy[i] for i in indices]
        x_min = min(b[0] for b in group)
        y_min = min(b[1] for b in group)
        x_max = max(b[2] for b in group)
        y_max = max(b[3] for b in group)
        paragraphs.append([x_min, y_min, x_max, y_max])

    return paragraphs


def merge_horizontal_ocr(left_ocr, right_ocr, left_width, overlap_size, y_tol=15, min_similarity=0.8):
    """
    Merge OCR results from two horizontally adjacent image chunks with overlap.
    Handles slight OCR variations and repeated text in the left segment.

    left_ocr, right_ocr: list of dicts {'text': str, 'box': [x_min, y_min, x_max, y_max]}
    left_width: width of the left chunk (before overlap)
    overlap_size: width of the overlap region
    y_tol: tolerance for matching lines vertically
    min_similarity: minimum similarity ratio to consider texts as potentially overlapping

    Returns: merged list of dicts
    """

    # Find candidate pairs at the overlap region
    merged = []
    used_right = set()
    for l in left_ocr:
        lx_min, ly_min, lx_max, ly_max = l['box']

        # Only consider left boxes near the right edge (in overlap)
        if lx_max <= left_width - overlap_size:
            # Not in overlap, keep as is
            merged.append(l)
            continue

        # Limit search to the overlapping part of the left text
        # Calculate the approximate number of characters in the overlap region
        overlap_char = int(overlap_size / (lx_max - lx_min) * len(l['text']))
        overlap_start = max(0, len(l['text']) - overlap_char)

        for idx, r in enumerate(right_ocr):
            rx_min, ry_min, rx_max, ry_max = r['box']
            rx_min += left_width
            rx_max += left_width
            
            # Only consider right boxes near the left edge (in overlap)
            if rx_min >= left_width + overlap_size or abs(ly_min - ry_min) >= y_tol:
                continue

            # Check similarity of texts
            sm = difflib.SequenceMatcher(None, l['text'], r['text'])
            
            if sm.ratio() <= min_similarity:  # Overall similarity check
                continue

            # Find longest match within the overlapping region of the left text
            match = sm.find_longest_match(overlap_start, len(l['text']), 0, len(r['text']))
            
            # If a significant overlap exists
            if match.size > 2:
                # Merge texts, removing duplicate part
                merged_text = l['text'][:match.a] + r['text']
                # Merge boxes
                merged_box = [
                    min(lx_min, rx_min),
                    min(ly_min, ry_min),
                    max(lx_max, rx_max),
                    max(ly_max, ry_max)
                ]
                merged.append({'text': merged_text, 'box': merged_box})
                used_right.add(idx)
                break
        
        else:
            # No match found, keep as is
            merged.append(l)

    # Add right_ocr items not already merged
    for idx, r in enumerate(right_ocr):
        if idx not in used_right:
            merged.append(r)

    return merged

