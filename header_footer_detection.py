#!/usr/bin/python3
import fitz
import regex as re
from chardet import detect
from fuzzywuzzy import fuzz

## Theses functions are based on the article by Lin (2003): https://www.researchgate.net/publication/221253782_Header_and_Footer_Extraction_by_Page-Association



def weight(ind, line_count, weights):

    if ind == 0:
        w = weights["first"]
    elif ind == 1:
        w = weights["second"]
    elif ind == 2:
        w = weights["third"]
    elif ind == line_count:
        w = weights["first"]
    elif ind == line_count - 1:
        w = weights["second"]
    elif ind == line_count - 2:
        w = weights["third"]
    else:
        w = weights["default"]
    return w


def position(block, page):
    page_dim = page.rect
    bblock = fitz.Rect(block[0], block[1], block[2], block[3])
    # some bugs on portait oriented files
    if not page_dim.contains(bblock) and page_dim[2] > page_dim[3]:
        bblock = fitz.Rect(block[1], block[0], block[3], block[2])
    div1 = fitz.Rect(0, 0, page_dim[2], page_dim[3] / 2)
    div2 = fitz.Rect(0, page_dim[1] - (page_dim[1] / 2), page_dim[2], page_dim[3])
    if div1.contains(bblock):
        return "header"
    elif div2.contains(bblock):
        return "footer"
    else:
        return False


def base_similarity(txt1, txt2, num=False):
    if num:
        txt1 = re.sub("\d", "@", txt1)
        txt2 = re.sub("\d", "@", txt2)

    matched_char = len(set(txt1).intersection(set(txt2)))
    longest = max(len(txt1), len(txt2))
    return matched_char / longest


def geo_similarity(bb1, bb2):

    bb1 = {"x1": bb1[0], "x2": bb1[2], "y1": bb1[1], "y2": bb1[3]}
    bb2 = {"x1": bb2[0], "x2": bb2[2], "y1": bb2[1], "y2": bb2[3]}
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def exist(i, lst):
    try:
        test = lst[i]
        return True
    except IndexError:
        return False


def get_candidates(doc, weights, win):
    """Returns :
    1 ) a dictionary with all the candidate headers and footers blocks in the pdf.
    key = page, value = list of blocks in that page.

    2 ) a dictionary of the position for each candidate block (header or footer position)
    """
    pages = [page for page in doc]
    dic_pages = dict()
    dic_pos = dict()

    # iteration throug pages
    for j in range(0, len(pages)):
        # print("page ", j)
        dic_blocks = dict()

        page = pages[j]
        lines_txt_j = [
            block[4]
            for block in page.get_text("blocks")
            if block[4].replace("\n", "").strip() and not block[4].startswith("<image")
        ]
        lines_bbox_j = [
            block[:4]
            for block in page.get_text("blocks")
            if block[4].replace("\n", "").strip() and not block[4].startswith("<image")
        ]
        full_blocks_j = [
            block
            for block in page.get_text("blocks")
            if block[4].replace("\n", "").strip() and not block[4].startswith("<image")
        ]
        # iteration throug lines
        for i in range(0, len(lines_txt_j)):

            sims = []
            # iteration through surrounding pages
            for k in range(max(j - win, 0), min(j + win, len(pages))):

                page_k = pages[k]
                lines_txt_k = [
                    block[4]
                    for block in page_k.get_text("blocks")
                    if block[4].replace("\n", "").strip()
                    and not block[4].startswith("<image")
                ]
                lines_bbox_k = [
                    block[:4]
                    for block in page_k.get_text("blocks")
                    if block[4].replace("\n", "").strip()
                    and not block[4].startswith("<image")
                ]

                if lines_txt_k:
                    if not exist(i, lines_txt_k):
                        w = weight(i, len(lines_txt_j), weights)
                        bs = base_similarity(lines_txt_j[i], lines_txt_k[-1])
                        gs = geo_similarity(lines_bbox_j[i], lines_bbox_k[-1])
                    else:

                        w = weight(i, len(lines_txt_j), weights)
                        bs = base_similarity(lines_txt_j[i], lines_txt_k[i])
                        gs = geo_similarity(lines_bbox_j[i], lines_bbox_k[i])
                else:
                    continue

                similarity = bs * gs
                weighted_similarity = similarity * w
                sims.append(weighted_similarity)
            score_line_i = sum(sims)
            dic_blocks[full_blocks_j[i]] = score_line_i
        # print(dic_blocks)
        candidates = [b for b in dic_blocks if dic_blocks[b] > 0.8]
        candidates_pos = {b: position(b, page) for b in candidates}

        dic_pages[j] = candidates
        dic_pos[j] = candidates_pos

    return dic_pages, dic_pos


########


