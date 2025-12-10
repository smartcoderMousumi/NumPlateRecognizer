# ocrModule.py  -- FAST + improved confusion handling
"""
Fast OCR pipeline with improved confusion handling for:
 - A <-> 4 <-> M
 - I <-> J <-> 1
 - 0 <-> O <-> D
Keeps speed-oriented settings while applying small, prioritized substitutions
and a cheap state-code repair for the first two characters.
API: read_plate_text(plate_img: np.ndarray) -> (best_text: str, processed_img: np.ndarray)
"""

import cv2
import numpy as np
import pytesseract
import re
from itertools import product
from typing import Tuple, List

# ---------- CONFIG (fast + tuned) ----------
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PSM_LIST = [7, 8]             # single-line, single-word (fast)
UPSCALE = 1.6                 # smaller upscale for speed
MAX_VARIANTS = 80
MAX_COMBINATIONS = 80

# Confusion map expanded and prioritized
CONFUSION_MAP = {
    # digits <-> letters groups we want to handle
    "0": ["O", "D"],
    "O": ["0", "D"],
    "D": ["0", "O"],

    "1": ["I", "J", "L"],
    "I": ["1", "J", "L"],
    "J": ["1", "I"],
    "L": ["1", "I"],

    "4": ["A", "M"],
    "A": ["4", "M"],
    "M": ["A", "4"],

    "2": ["Z"],
    "Z": ["2"],

    "5": ["S"],
    "S": ["5"],

    "8": ["B"],
    "B": ["8"],
}

PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$"),
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]\d{3}$"),
]
STATE_CODES = {
    "AN","AP","AR","AS","BR","CH","CT","DL","DN","GA","GJ","HP","HR","JH","JK","KA","KL",
    "LA","LD","MH","ML","MN","MP","MZ","NL","OR","PB","PY","RJ","SK","TG","TN","TR","UP","UT","WB"
}

# ---------- utils ----------
def _clean_text(s: str) -> str:
    return "".join(ch for ch in (s or "").upper() if ch.isalnum())

def looks_like_plate(s: str) -> bool:
    s = (s or "").upper()
    for p in PLATE_PATTERNS:
        if p.match(s):
            return True
    if len(s) >= 2 and s[:2] in STATE_CODES and any(ch.isdigit() for ch in s):
        return True
    return False

def expected_mask_for_plate_length(length: int) -> List[str]:
    # 'A' for alpha, 'D' for digit
    if length == 10:
        return ['A','A','D','D','A','A','D','D','D','D']
    if length == 9:
        return ['A','A','D','D','A','D','D','D','D'][:9]
    return ['A' if i < 2 else 'D' for i in range(length)]

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev = list(range(lb+1))
    for i in range(1, la+1):
        cur = [i] + [0]*lb
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
        prev = cur
    return prev[lb]

# ---------- preprocessing (fast) ----------
def _basic_gray(plate_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    return gray

def _clahe(gray: np.ndarray) -> np.ndarray:
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray)
    except Exception:
        return gray

def _prepare_for_ocr_fast(img_gray: np.ndarray, upscale: float = UPSCALE) -> List[np.ndarray]:
    h,w = img_gray.shape[:2]
    new_w = max(int(w * upscale), w+1)
    new_h = max(int(h * upscale), h+1)
    small = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    _, th = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    th2 = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return [th, th2]

# ---------- OCR helper (fast) ----------
def ocr_with_confidence_fast(img_bin: np.ndarray, psm: int) -> Tuple[str, float]:
    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={WHITELIST}"
    try:
        data = pytesseract.image_to_data(img_bin, output_type=pytesseract.Output.DICT, config=config)
        texts = []
        confs = []
        n = len(data.get('text', []))
        for i in range(n):
            txt = (data['text'][i] or "").strip()
            conf_raw = data['conf'][i]
            try:
                conf = float(conf_raw)
            except Exception:
                conf = -1.0
            if txt != "" and conf > 0:
                texts.append(txt)
                confs.append(conf)
        if texts:
            return "".join(texts), float(np.median(np.array(confs))) if confs else 0.0
    except Exception:
        pass
    raw = pytesseract.image_to_string(img_bin, config=config)
    return raw.strip(), 0.0

# ---------- small variant generation (position-aware, prioritized) ----------
def generate_variants_from_string_fast(s: str, max_combinations: int = MAX_COMBINATIONS) -> List[str]:
    s = (s or "").upper()
    mask = expected_mask_for_plate_length(len(s))
    opts = []
    for i, ch in enumerate(s):
        choices = [ch]
        if ch in CONFUSION_MAP:
            # prioritize alternatives that match expected type
            expected = mask[i] if i < len(mask) else None
            preferred = []
            fallback = []
            for alt in CONFUSION_MAP[ch]:
                if expected == 'D' and alt.isdigit():
                    preferred.append(alt)
                elif expected == 'A' and alt.isalpha():
                    preferred.append(alt)
                else:
                    fallback.append(alt)
            # place preferred first then fallback
            for c in preferred + fallback:
                if c not in choices:
                    choices.append(c)
        opts.append(list(dict.fromkeys(choices)))
    variants = []
    for prod in product(*opts):
        variants.append("".join(prod))
        if len(variants) >= max_combinations:
            break
    return variants

def pick_best_by_mask_fast(candidates: List[str]) -> str:
    if not candidates:
        return ""
    # prefer exact pattern
    for c in candidates:
        if looks_like_plate(c):
            return c
    # simple mask scoring + state bonus
    best = None
    best_score = -1e9
    for c in candidates:
        score = 0.0
        mask = expected_mask_for_plate_length(len(c))
        for i, ch in enumerate(c):
            expected = mask[i] if i < len(mask) else None
            if expected == 'A' and ch.isalpha():
                score += 2.0
            elif expected == 'D' and ch.isdigit():
                score += 2.0
        if len(c) >= 2 and c[:2] in STATE_CODES:
            score += 1.5
        score += 0.01 * len(c)
        if score > best_score:
            best_score = score
            best = c
    return best or (candidates[0] if candidates else "")

# ---------- cheap state-code repair ----------
def repair_state_code_prefix(candidate: str) -> str:
    """
    If the first two chars are not a known state code, try small substitutions
    for the first two chars using the confusion map to see if a valid state code appears.
    Returns either repaired candidate or original if no repair found.
    """
    if not candidate or len(candidate) < 2:
        return candidate
    prefix = candidate[:2].upper()
    if prefix in STATE_CODES:
        return candidate
    # build small set of variants modifying only first 2 chars
    first_opts = [prefix[0]] + (CONFUSION_MAP.get(prefix[0], []) if prefix[0] in CONFUSION_MAP else [])
    second_opts = [prefix[1]] + (CONFUSION_MAP.get(prefix[1], []) if prefix[1] in CONFUSION_MAP else [])
    tried = []
    for a in first_opts:
        for b in second_opts:
            code = (a + b).upper()
            if code in STATE_CODES:
                # replace prefix
                repaired = code + candidate[2:]
                return repaired
            tried.append(code)
    return candidate

# ---------- MAIN fast read_plate_text ----------
def read_plate_text(plate_img: np.ndarray) -> Tuple[str, np.ndarray]:
    """
    Fast OCR pipeline: return (best_text, processed_image)
    """
    # quick resize: keep width reasonable
    h,w = plate_img.shape[:2]
    if w > 800:
        scale = 800.0 / w
        plate_img = cv2.resize(plate_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    gray = _basic_gray(plate_img)
    variants = [gray, _clahe(gray)]

    attempts = []  # (text, conf, proc)
    for var in variants:
        thresh_list = _prepare_for_ocr_fast(var, upscale=UPSCALE)
        for proc in thresh_list:
            for psm in PSM_LIST:
                text, conf = ocr_with_confidence_fast(proc, psm)
                cleaned = _clean_text(text)
                if cleaned:
                    attempts.append((cleaned, conf, proc.copy()))

    if not attempts:
        proc = _prepare_for_ocr_fast(gray, upscale=UPSCALE)[0]
        return "", proc

    # aggregate by text
    agg = {}
    for txt, conf, img in attempts:
        if txt not in agg:
            agg[txt] = {"confs": [], "img": img}
        agg[txt]["confs"].append(conf)

    scored = []
    for txt, info in agg.items():
        median_conf = float(np.median(np.array(info["confs"]))) if info["confs"] else 0.0
        bonus = 1.5 if looks_like_plate(txt) else 0.0
        if len(txt) >= 2 and txt[:2] in STATE_CODES:
            bonus += 0.5
        score = median_conf + bonus + 0.01 * len(txt)
        scored.append((score, txt, info["img"], median_conf))

    scored.sort(reverse=True, key=lambda x: (x[0], x[3]))
    top_texts = [t for _, t, _, _ in scored[:5]]

    # quick state-code repair on top candidate(s)
    final_choice = None
    for score, txt, img, conf in scored:
        # try repairing prefix if needed
        repaired = repair_state_code_prefix(txt)
        if repaired != txt and looks_like_plate(repaired):
            return repaired, img
        if looks_like_plate(txt):
            return txt, img
    # expand small variants from top_texts
    pool = set(top_texts)
    for t in top_texts:
        for v in generate_variants_from_string_fast(t, max_combinations=MAX_VARIANTS):
            pool.add(v)

    chosen = pick_best_by_mask_fast(list(pool))

    if not chosen:
        chosen = scored[0][1]
        return chosen, scored[0][2]

    # final: if chosen prefix wrong try repair
    final_repaired = repair_state_code_prefix(chosen)
    if final_repaired != chosen and looks_like_plate(final_repaired):
        # use processed image of top candidate (best fallback)
        return final_repaired, scored[0][2]

    # return processed image associated with chosen if available
    for score, t, img, conf in scored:
        if t == chosen:
            return chosen, img
    return chosen, scored[0][2]
