# ocrModule.py
"""
Improved OCR module for ANPR:
- multiple preprocessing variants (CLAHE, sharpening, bilateral denoise)
- multiple channel/threshold variants
- upsizing before OCR
- uses image_to_data to compute per-attempt median confidence
- tries multiple --psm modes and aggregates candidates
- position-aware post-processing using expected masks and a confusion map
- bounded variant generation to avoid combinatorial explosion
API:
    read_plate_text(plate_img: np.ndarray) -> (best_text: str, processed_img: np.ndarray)
"""
import cv2
import numpy as np
import pytesseract
import re
from itertools import product
from typing import Tuple, List

# ---------------- CONFIG ----------------
# Set this path to where your tesseract.exe is installed
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Whitelist and PSM modes
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PSM_LIST = [7, 8, 6, 11]  # useful page segmentation modes to try

# Confusion map (common visual confusions)
CONFUSION_MAP = {
    "0": ["O", "D"],
    "O": ["0", "D"],
    "D": ["0", "O"],
    "1": ["I", "L"],
    "I": ["1", "L"],
    "L": ["1", "I"],
    "2": ["Z"],
    "Z": ["2"],
    "5": ["S"],
    "S": ["5"],
    "8": ["B"],
    "B": ["8"],
}

# Plate patterns and state codes (common Indian codes — extend as needed)
PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$"),  # e.g., WB20AB1234
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]\d{3}$"),
]
STATE_CODES = {
    "AN","AP","AR","AS","BR","CH","CT","DL","DN","GA","GJ","HP","HR","JH","JK","KA","KL",
    "LA","LD","MH","ML","MN","MP","MZ","NL","OR","PB","PY","RJ","SK","TG","TN","TR","UP","UT","WB"
}

# Limits to avoid explosion
MAX_VARIANTS = 250
MAX_COMBINATIONS = 200

# ----------------- UTILITIES -----------------
def _clean_text(s: str) -> str:
    return "".join(ch for ch in (s or "").upper() if ch.isalnum())

def looks_like_plate(s: str) -> bool:
    s = (s or "").upper()
    for p in PLATE_PATTERNS:
        if p.match(s):
            return True
    # quick heuristic: plausible if starts with state code and contains digits
    if len(s) >= 2 and s[:2] in STATE_CODES and any(ch.isdigit() for ch in s):
        return True
    return False

def expected_mask_for_plate_length(length: int) -> List[str]:
    if length == 10:
        return ['A','A','D','D','A','A','D','D','D','D']
    if length == 9:
        return ['A','A','D','D','A','D','D','D','D'][:9]
    # fallback: first two letters then digits
    return ['A' if i < 2 else 'D' for i in range(length)]

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb+1))
    for i in range(1, la+1):
        cur = [i] + [0]*lb
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
        prev = cur
    return prev[lb]

# ---------------- PREPROCESSING ----------------
def _clahe_sharpen(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    c = clahe.apply(gray)
    b = cv2.bilateralFilter(c, 9, 75, 75)
    gaussian = cv2.GaussianBlur(b, (0,0), 3)
    unsharp = cv2.addWeighted(b, 1.6, gaussian, -0.6, 0)
    return unsharp

def _basic_preprocess(plate_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray

def _get_variants(plate_img: np.ndarray) -> List[np.ndarray]:
    """Return grayscale variants to try (unique by simple checksum)."""
    gray = _basic_preprocess(plate_img)
    variants = [gray]
    try:
        variants.append(_clahe_sharpen(gray))
    except Exception:
        pass
    try:
        eq = cv2.equalizeHist(gray)
        variants.append(eq)
    except Exception:
        pass
    # add color channels (B,G,R)
    try:
        b,g,r = cv2.split(plate_img)
        variants.extend([b,g,r])
    except Exception:
        pass
    # dedupe by sum signature
    uniq = []
    seen = set()
    for v in variants:
        key = int(v.sum() % (1<<30))
        if key not in seen:
            uniq.append(v)
            seen.add(key)
    return uniq

def _prepare_for_ocr(img_gray: np.ndarray, upscale: float = 2.5) -> List[np.ndarray]:
    """Given a gray image, produce thresholded variants suitable for OCR."""
    outputs = []
    h, w = img_gray.shape[:2]
    new_w = max(int(w * upscale), w+1)
    new_h = max(int(h * upscale), h+1)
    big = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Otsu
    try:
        _, th_otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        outputs.append(th_otsu)
    except Exception:
        pass
    # adaptive
    try:
        th_adapt = cv2.adaptiveThreshold(big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 10)
        outputs.append(th_adapt)
    except Exception:
        pass
    # ensure dark text on light background
    final = []
    for o in outputs:
        if np.mean(o) < 127:
            final.append(cv2.bitwise_not(o))
        else:
            final.append(o)
    # morphological closing to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    final2 = []
    for f in final:
        try:
            f2 = cv2.morphologyEx(f, cv2.MORPH_CLOSE, kernel, iterations=1)
            final2.append(f2)
        except Exception:
            final2.append(f)
    return final2 if final2 else [cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)]

# ---------------- OCR with confidence ----------------
def ocr_with_confidence(img_bin: np.ndarray, psm: int) -> Tuple[str, float, List[float], str]:
    """
    Returns (joined_text, median_confidence, list_of_char_confidences, raw_joined_text)
    Uses image_to_data to collect per-word confidences and maps them approximately to character-level confidences.
    """
    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={WHITELIST}"
    try:
        data = pytesseract.image_to_data(img_bin, output_type=pytesseract.Output.DICT, config=config)
    except Exception:
        # fallback
        raw = pytesseract.image_to_string(img_bin, config=config)
        return raw.strip(), 0.0, [], raw.strip()

    texts = []
    confs = []
    raw_parts = []
    n = len(data.get('text', []))
    for i in range(n):
        txt = (data['text'][i] or "").strip()
        conf_raw = data['conf'][i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0
        if txt != "":
            raw_parts.append(txt)
            texts.append(txt)
            if conf > 0:
                # approx: assign same conf to each char of the word
                confs.extend([conf] * len(txt))
    if confs:
        median_conf = float(np.median(np.array(confs)))
    else:
        median_conf = 0.0

    if texts:
        joined = "".join(texts)
        raw_joined = " ".join(raw_parts).strip()
        return joined, median_conf, confs, raw_joined

    raw = pytesseract.image_to_string(img_bin, config=config)
    return raw.strip(), 0.0, [], raw.strip()

# ---------------- Variant generation (position-aware) ----------------
def generate_variants_from_string(s: str, max_combinations: int = MAX_COMBINATIONS) -> List[str]:
    """
    Generate position-aware variants using CONFUSION_MAP but filtered by expected A/D mask.
    Caps total combinations.
    """
    s = (s or "").upper()
    mask = expected_mask_for_plate_length(len(s))
    opts = []
    for i, ch in enumerate(s):
        choices = [ch]
        if ch in CONFUSION_MAP:
            alts = CONFUSION_MAP[ch]
            expected = mask[i] if i < len(mask) else None
            filtered = []
            fallback = []
            for a in alts:
                if expected == 'D' and a.isdigit():
                    filtered.append(a)
                elif expected == 'A' and a.isalpha():
                    filtered.append(a)
                else:
                    fallback.append(a)
            for c in filtered + fallback:
                if c not in choices:
                    choices.append(c)
        opts.append(choices)

    variants = []
    for prod in product(*opts):
        variants.append("".join(prod))
        if len(variants) >= max_combinations:
            break
    return variants

# ---------------- Scoring and selection ----------------
def pick_best_by_mask(candidates: List[str]) -> str:
    if not candidates:
        return ""
    # 1) exact pattern match
    for c in candidates:
        if looks_like_plate(c):
            return c
    # 2) mask-based scoring with state bonus and small length bonus
    scored = []
    for c in candidates:
        score = 0.0
        mask = expected_mask_for_plate_length(len(c))
        for i, ch in enumerate(c):
            expected = mask[i] if i < len(mask) else None
            if expected == 'A' and ch.isalpha():
                score += 2.0
            elif expected == 'D' and ch.isdigit():
                score += 2.0
            elif ch.isalnum():
                score += 0.2
            else:
                score -= 0.5
        if len(c) >= 2 and c[:2] in STATE_CODES:
            score += 1.5
        score += 0.01 * len(c)
        scored.append((score, c))
    scored.sort(reverse=True)
    best_score, best_candidate = scored[0]
    # tie-break among top 3 using edit distance to mask-template
    def mask_to_template(mask):
        return "".join('A' if x == 'A' else '0' for x in mask)
    template = mask_to_template(expected_mask_for_plate_length(len(best_candidate)))
    top_k = [c for _, c in scored[:3]]
    best = best_candidate
    best_dist = _levenshtein(best, template)
    for cand in top_k:
        d = _levenshtein(cand, template)
        if d < best_dist:
            best = cand
            best_dist = d
    return best

# ----------------- MAIN FUNCTION -----------------
def read_plate_text(plate_img: np.ndarray) -> Tuple[str, np.ndarray]:
    """
    Input: cropped BGR plate image
    Output: (best_text, processed_image_used_for_best)
    """
    attempts = []  # tuples (cleaned_text, median_conf, proc_image)

    gray_variants = _get_variants(plate_img)
    for gray in gray_variants:
        thresh_variants = _prepare_for_ocr(gray)
        for proc in thresh_variants:
            for psm in PSM_LIST:
                raw_text, median_conf, char_confs, raw_joined = ocr_with_confidence(proc, psm)
                cleaned = _clean_text(raw_text)
                if cleaned:
                    attempts.append((cleaned, float(median_conf), proc.copy()))

    if not attempts:
        # fallback: return empty and first processed variant for display
        fallback_proc = _prepare_for_ocr(_basic_preprocess(plate_img))[0]
        return "", fallback_proc

    # aggregate per unique candidate
    agg = {}
    for text, conf, img in attempts:
        if text not in agg:
            agg[text] = {"confs": [], "img": img}
        agg[text]["confs"].append(conf)

    scored_candidates = []
    for text, info in agg.items():
        conf_med = float(np.median(np.array(info["confs"]))) if info["confs"] else 0.0
        pattern_bonus = 2.0 if looks_like_plate(text) else 0.0
        state_bonus = 1.0 if len(text) >= 2 and text[:2] in STATE_CODES else 0.0
        len_bonus = 0.01 * len(text)
        score = conf_med + pattern_bonus + state_bonus + len_bonus
        scored_candidates.append((score, text, info["img"], conf_med))

    scored_candidates.sort(reverse=True, key=lambda x: (x[0], x[3]))

    # collect top N texts for mask-guided refinement
    top_texts = [t for _, t, _, _ in scored_candidates[:8]]

    # if any top text already looks like a plate, pick the highest-scoring such text
    for score, text, img, conf in scored_candidates:
        if looks_like_plate(text):
            return text, img

    # expand using position-aware variants from top_texts but bounded
    candidate_pool = set(top_texts)
    for txt in top_texts:
        for v in generate_variants_from_string(txt, max_combinations=MAX_VARIANTS):
            candidate_pool.add(v)

    # final choice using mask/pattern guidance
    chosen = pick_best_by_mask(list(candidate_pool))

    # if still nothing, fallback to highest-scored candidate
    if not chosen:
        chosen = scored_candidates[0][1]
        proc_img = scored_candidates[0][2]
        return chosen, proc_img

    # find a processed image associated with chosen (best conf if exist)
    for score, text, img, conf in scored_candidates:
        if text == chosen:
            return chosen, img

    # fallback to top attempt processed image
    return chosen, scored_candidates[0][2]
