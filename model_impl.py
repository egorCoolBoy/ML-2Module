from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import easyocr
except Exception:
    easyocr = None

try:
    from alpr.logging_utils import get_logger

    LOGGER = get_logger(__name__)
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LOGGER = logging.getLogger(__name__)


class _RussianPlateDecoder:
    """Правила формата госномера РФ и восстановление строки после OCR."""

    ALLOWED_LETTERS = "АВЕКМНОРСТУХ"

    LATIN_TO_CYRILLIC = str.maketrans(
        {
            "A": "А",
            "B": "В",
            "C": "С",
            "E": "Е",
            "H": "Н",
            "K": "К",
            "M": "М",
            "N": "Н",
            "O": "О",
            "P": "Р",
            "R": "Р",
            "S": "С",
            "T": "Т",
            "X": "Х",
            "Y": "У",
        }
    )

    PATTERNS_ALL = (
        re.compile(r"^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$"),
        re.compile(r"^\d{4}[АВЕКМНОРСТУХ]{2}\d{2,3}$"),
        re.compile(r"^[АВЕКМНОРСТУХ]{2}\d{5,6}$"),
    )
    PATTERN_CIVIL = re.compile(r"^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$")
    PATTERN_TRAILER = re.compile(r"^\d{4}[АВЕКМНОРСТУХ]{2}\d{2,3}$")
    PATTERN_MOTO = re.compile(r"^[АВЕКМНОРСТУХ]{2}\d{5,6}$")

    # Только для запасного coerce-пути (после beam). Н/У вместо Т для 1/7 — меньше ложных «ТТ».
    LETTER_FIXUPS = {
        "0": "О",
        "1": "Н",
        "3": "Е",
        "4": "А",
        "5": "С",
        "7": "У",
        "8": "В",
    }
    DIGIT_FIXUPS = {
        "А": "4",
        "В": "8",
        "Б": "6",
        "З": "3",
        "О": "0",
        "Т": "7",
    }

    REGION_CODES = {
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "21",
        "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
        "32", "33", "34", "35", "36", "37", "38", "39", "40", "41",
        "42", "43", "44", "45", "46", "47", "48", "49", "50", "51",
        "52", "53", "54", "55", "56", "57", "58", "59", "60", "61",
        "62", "63", "64", "65", "66", "67", "68", "69", "70", "71",
        "72", "73", "74", "75", "76", "77", "78", "79", "80", "81",
        "82", "83", "84", "85", "86", "87", "88", "89", "90", "91",
        "92", "93", "94", "95", "96", "97", "98", "99", "102", "113",
        "116", "121", "122", "123", "124", "125", "126", "130", "134",
        "136", "138", "142", "147", "150", "152", "154", "159", "161",
        "163", "164", "173", "174", "177", "178", "186", "190", "193",
        "196", "197", "198", "199", "277", "299", "702", "716", "725",
        "750", "761", "763", "777", "790", "797", "799",
    }

    # Буква в позиции L, OCR дал цифру — штраф и альтернативы (Н/У/М вместо одной только Т).
    LETTER_SLOT_FALLBACKS = {
        "0": (("О", 0.15),),
        "1": (("Н", 0.38), ("Т", 0.55), ("М", 0.58)),
        "3": (("Е", 0.20),),
        "4": (("А", 0.20),),
        "5": (("С", 0.25),),
        "7": (("У", 0.32), ("Т", 0.42)),
        "8": (("В", 0.20),),
    }
    DIGIT_SLOT_FALLBACKS = {
        "О": (("0", 0.15),),
        "В": (("8", 0.30),),
        "З": (("3", 0.20),),
        "Т": (("7", 0.35),),
        "Б": (("6", 0.35),),
    }

    BEAM_WIDTH = 20
    BEAM_MAX_PENALTY = 5.0
    TOP_CANDIDATES = 12

    CIVIL_TEMPLATE_8 = ["L", "D", "D", "D", "L", "L", "D", "D"]
    CIVIL_TEMPLATE_9 = ["L", "D", "D", "D", "L", "L", "D", "D", "D"]

    @staticmethod
    def normalize_ocr_string(text: str) -> str:
        return re.sub(r"[\W_]+", "", text.upper(), flags=re.UNICODE)

    @classmethod
    def to_cyrillic(cls, text: str) -> str:
        return text.translate(cls.LATIN_TO_CYRILLIC)

    @classmethod
    def filter_allowed_chars(cls, text: str) -> str:
        pool = set(cls.ALLOWED_LETTERS + "0123456789")
        return "".join(ch for ch in text if ch in pool)

    @classmethod
    def format_with_space(cls, compact: str) -> str:
        if len(compact) in (8, 9):
            return f"{compact[:6]} {compact[6:]}"
        return compact

    @classmethod
    def matches_any_pattern(cls, compact: str) -> bool:
        return any(p.fullmatch(compact) for p in cls.PATTERNS_ALL)

    @classmethod
    def matches_civil(cls, compact: str) -> bool:
        return bool(cls.PATTERN_CIVIL.fullmatch(compact))

    @classmethod
    def is_valid_region(cls, region: str) -> bool:
        return region in cls.REGION_CODES

    @classmethod
    def _coerce_civil_single_pass(cls, candidate: str) -> str | None:
        if len(candidate) not in (8, 9):
            return None

        letter_idx = {0, 4, 5}
        digit_idx = {1, 2, 3}
        digit_idx.update({6, 7} if len(candidate) == 8 else {6, 7, 8})

        out: list[str] = []
        for i, ch in enumerate(candidate):
            if i in letter_idx:
                if ch in cls.ALLOWED_LETTERS:
                    out.append(ch)
                elif (r := cls.LETTER_FIXUPS.get(ch)) and r in cls.ALLOWED_LETTERS:
                    out.append(r)
                else:
                    return None
            elif i in digit_idx:
                if ch.isdigit():
                    out.append(ch)
                elif (r := cls.DIGIT_FIXUPS.get(ch)) and r.isdigit():
                    out.append(r)
                else:
                    return None
            else:
                return None

        fixed = "".join(out)
        return fixed if cls.matches_civil(fixed) else None

    @classmethod
    def decode_char_options(cls, ch: str, slot: str) -> list[tuple[str, float]]:
        if slot == "L":
            opts: list[tuple[str, float]] = []
            if ch in cls.ALLOWED_LETTERS:
                opts.append((ch, 0.0))
            for cand, pen in cls.LETTER_SLOT_FALLBACKS.get(ch, ()):
                if cand in cls.ALLOWED_LETTERS:
                    opts.append((cand, pen))
            return opts if opts else [(ch, 1.2)]

        opts = []
        if ch.isdigit():
            opts.append((ch, 0.0))
        for cand, pen in cls.DIGIT_SLOT_FALLBACKS.get(ch, ()):
            if cand.isdigit():
                opts.append((cand, pen))
        return opts if opts else [(ch, 1.2)]

    @classmethod
    def beam_search_civil(cls, raw_text: str) -> list[tuple[str, float]]:
        cleaned = cls.normalize_ocr_string(raw_text)
        if not cleaned:
            return []

        cleaned = cls.filter_allowed_chars(cls.to_cyrillic(cleaned))
        if len(cleaned) < 8:
            return []

        results: list[tuple[str, float]] = []
        for win_len in (8, 9):
            if len(cleaned) < win_len:
                continue
            template = cls.CIVIL_TEMPLATE_8 if win_len == 8 else cls.CIVIL_TEMPLATE_9
            for start in range(0, len(cleaned) - win_len + 1):
                window = cleaned[start : start + win_len]
                beam: list[tuple[str, float]] = [("", 0.0)]
                for pos, src in enumerate(window):
                    slot = template[pos]
                    next_beam: list[tuple[str, float]] = []
                    for prefix, p0 in beam:
                        for ch, p1 in cls.decode_char_options(src, slot):
                            next_beam.append((prefix + ch, p0 + p1))
                    next_beam.sort(key=lambda x: x[1])
                    beam = next_beam[: cls.BEAM_WIDTH]

                for cand, penalty in beam:
                    if cls.matches_civil(cand):
                        if not cls.is_valid_region(cand[6:]):
                            penalty += 0.45
                        results.append((cand, penalty))

        results.sort(key=lambda x: x[1])
        return results[: cls.TOP_CANDIDATES]

    @classmethod
    def extract_civil_plate(cls, text: str) -> str | None:
        cleaned = cls.filter_allowed_chars(cls.to_cyrillic(cls.normalize_ocr_string(text)))
        if len(cleaned) < 8:
            return None

        beam = cls.beam_search_civil(text)
        if beam and beam[0][1] < cls.BEAM_MAX_PENALTY:
            return beam[0][0]

        windows = [cleaned]
        for n in (8, 9):
            if len(cleaned) >= n:
                for s in range(0, len(cleaned) - n + 1):
                    windows.append(cleaned[s : s + n])

        for w in windows:
            if fixed := cls._coerce_civil_single_pass(w):
                return fixed
        return None

    @classmethod
    def extract_plate(cls, text: str) -> str | None:
        cleaned = cls.filter_allowed_chars(cls.to_cyrillic(cls.normalize_ocr_string(text)))
        if len(cleaned) < 8:
            return None

        civil = cls.extract_civil_plate(cleaned)
        if civil is not None:
            return cls.format_with_space(civil)

        windows = [cleaned]
        for n in (8, 9):
            if len(cleaned) >= n:
                for s in range(0, len(cleaned) - n + 1):
                    windows.append(cleaned[s : s + n])

        for w in windows:
            if cls.PATTERN_TRAILER.fullmatch(w) or cls.PATTERN_MOTO.fullmatch(w):
                return cls.format_with_space(w)
        return None

    @classmethod
    def ocr_hypothesis_variants(cls, text: str) -> list[str]:
        cleaned = cls.normalize_ocr_string(text)
        if not cleaned:
            return []

        base = cls.to_cyrillic(cleaned)
        variants = [
            base,
            base.replace("I", "1").replace("L", "1"),
            base.replace("Q", "0").replace("D", "0"),
            base.replace("V", "В").replace("W", "В"),
            base.replace("R", "Р").replace("N", "Н").replace("S", "С"),
            base.replace("S", "5").replace("Z", "2"),
            base.replace("G", "6").replace("B", "8"),
        ]
        unique: list[str] = []
        seen: set[str] = set()
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                unique.append(v)
        return unique

    @classmethod
    def score_candidate(cls, text: str, ocr_confidence: float) -> float:
        compact = text.replace(" ", "")
        score = ocr_confidence
        if re.fullmatch(r"[АВЕКМНОРСТУХ0-9]+", compact):
            score += 0.10
        if cls.matches_any_pattern(compact):
            score += 0.35
        if cls.matches_civil(compact):
            score += 0.40
            region = compact[6:]
            score += 0.12 if cls.is_valid_region(region) else -0.18
        return score


class _PlateTracker:
    """Сопоставление рамок между кадрами и голосование по тексту номера."""

    def __init__(
        self,
        distance_threshold: float = 90.0,
        max_missed_frames: int = 8,
        max_votes: int = 14,
    ) -> None:
        self._distance_threshold = distance_threshold
        self._max_missed = max_missed_frames
        self._max_votes = max_votes
        self._next_id = 1
        self._tracks: dict[int, dict] = {}

    @staticmethod
    def _bbox_center(bbox: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _bbox_diagonal(bbox: list[float]) -> float:
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        return float((w * w + h * h) ** 0.5)

    def assign_track_id(self, bbox: list[float], already_used: set[int]) -> int:
        cx, cy = self._bbox_center(bbox)
        diag = self._bbox_diagonal(bbox)
        limit = max(self._distance_threshold, diag * 0.9)

        best_id = -1
        best_dist = float("inf")
        for tid, tr in self._tracks.items():
            if tid in already_used:
                continue
            tx, ty = tr["center"]
            d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_id = tid

        if best_id != -1 and best_dist <= limit:
            return best_id

        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "center": (cx, cy),
            "last_bbox": bbox,
            "missed": 0,
            "votes": [],
        }
        return tid

    def update_track(
        self,
        track_id: int,
        bbox: list[float],
        plate_text: str | None,
        plate_confidence: float | None,
    ) -> None:
        tr = self._tracks[track_id]
        tr["center"] = self._bbox_center(bbox)
        tr["last_bbox"] = bbox
        tr["missed"] = 0
        if plate_text:
            tr["votes"].append((plate_text, float(plate_confidence or 0.0)))
            if len(tr["votes"]) > self._max_votes:
                tr["votes"] = tr["votes"][-self._max_votes :]

    @staticmethod
    def consensus_from_votes(votes: list[tuple[str, float]]) -> tuple[str | None, float | None]:
        if not votes:
            return None, None

        score_by: dict[str, float] = defaultdict(float)
        conf_by: dict[str, float] = defaultdict(float)
        count_by: dict[str, int] = defaultdict(int)

        for text, conf in votes:
            score_by[text] += max(conf, 0.01)
            conf_by[text] = max(conf_by[text], conf)
            count_by[text] += 1

        best_text = None
        best_score = -1.0
        for text, sc in score_by.items():
            sc += 0.25 * count_by[text]
            if sc > best_score:
                best_text = text
                best_score = sc

        if best_text is None:
            return None, None
        # Одного кадра с уверенным OCR достаточно для «стабильной» подписи (камера).
        if count_by[best_text] < 2 and conf_by[best_text] < 0.45:
            return None, None
        return best_text, conf_by[best_text]

    def consensus_for(self, track_id: int) -> tuple[str | None, float | None]:
        return self.consensus_from_votes(self._tracks[track_id]["votes"])

    def last_plate_reading(self, track_id: int) -> tuple[str, float] | None:
        """Последний успешный OCR по треку (если текущий кадр не распознался)."""
        tr = self._tracks.get(track_id)
        if not tr:
            return None
        votes = tr.get("votes", [])
        return votes[-1] if votes else None

    def mark_missed_and_prune(self, matched_ids: set[int]) -> None:
        stale: list[int] = []
        for tid, tr in self._tracks.items():
            if tid not in matched_ids:
                tr["missed"] += 1
                if tr["missed"] > self._max_missed:
                    stale.append(tid)
        for tid in stale:
            self._tracks.pop(tid, None)


class _PlateCropOcr:
    """Подготовка кропа и вызов EasyOCR с постобработкой через _RussianPlateDecoder."""

    EASYOCR_ALLOWLIST = "0123456789ABEKMHOPCTXYАВЕКМНОРСТУХ"

    def __init__(
        self,
        reader: object | None,
        min_accept_score: float = 0.25,
        ocr_fast: bool = False,
    ) -> None:
        self._reader = reader
        # На камере кропы часто шумные/размытые — слишком высокий порог давал пустой текст при живой съёмке.
        self._min_accept_score = min_accept_score
        # Меньше вариантов препроцесса = быстрее кадр (важно для реального времени на CPU).
        self._ocr_fast = ocr_fast

    @staticmethod
    def preprocess(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return crop
        enlarged = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
        equalized = cv2.equalizeHist(denoised)
        return cv2.GaussianBlur(equalized, (3, 3), 0)

    def image_variants(self, crop: np.ndarray) -> list[np.ndarray]:
        base = self.preprocess(crop)
        variants: list[np.ndarray] = [crop, base]
        if base.size == 0:
            return variants

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(base)
        variants.append(clahe)
        if self._ocr_fast:
            return variants

        binary = cv2.adaptiveThreshold(
            base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
        )
        binary_inv = cv2.adaptiveThreshold(
            base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
        )
        variants.extend([binary, binary_inv])
        otsu_thr, otsu = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu_thr > 0:
            variants.extend([otsu, cv2.bitwise_not(otsu)])
        variants.append(cv2.rotate(base, cv2.ROTATE_180))
        return variants

    def read_plate_text(self, crop: np.ndarray) -> tuple[str | None, float | None]:
        if self._reader is None or crop.size == 0:
            return None, None

        decoder = _RussianPlateDecoder
        best_text: str | None = None
        best_conf: float | None = None
        best_score: float | None = None

        for img in self.image_variants(crop):
            try:
                lines = self._reader.readtext(
                    img,
                    detail=1,
                    paragraph=False,
                    allowlist=self.EASYOCR_ALLOWLIST,
                    decoder="beamsearch",
                    beamWidth=8,
                    text_threshold=0.45,
                    low_text=0.20,
                    link_threshold=0.25,
                    contrast_ths=0.10,
                    adjust_contrast=0.70,
                )
            except Exception as exc:
                LOGGER.debug("OCR failed for one crop variant: %s", exc)
                continue

            for _bbox, line, conf in lines:
                for form in decoder.ocr_hypothesis_variants(line):
                    candidate = decoder.extract_plate(form)
                    if candidate is None:
                        raw = decoder.extract_civil_plate(form)
                        candidate = decoder.format_with_space(raw) if raw else None
                    if candidate is None:
                        continue

                    sc = decoder.score_candidate(candidate, float(conf))
                    if len(candidate.replace(" ", "")) == 8:
                        sc += 0.02
                    if best_score is None or sc > best_score:
                        best_text = candidate
                        best_conf = float(conf)
                        best_score = sc

        if best_score is not None and best_score < self._min_accept_score:
            return None, None
        return best_text, best_conf


class My_LicensePlate_Model:
    """YOLO-детектор номера + EasyOCR + правила РФ и трекинг по кадрам."""

    # Совместимость: старые имена констант как алиасы на декодер
    RF_ALLOWED_LETTERS = _RussianPlateDecoder.ALLOWED_LETTERS
    LATIN_TO_CYRILLIC = _RussianPlateDecoder.LATIN_TO_CYRILLIC
    RF_PLATE_PATTERNS = _RussianPlateDecoder.PATTERNS_ALL
    RF_CIVIL_PLATE_PATTERN = _RussianPlateDecoder.PATTERN_CIVIL
    RF_TRAILER_PLATE_PATTERN = _RussianPlateDecoder.PATTERN_TRAILER
    RF_MOTO_PLATE_PATTERN = _RussianPlateDecoder.PATTERN_MOTO
    RF_LETTER_FIXUPS = _RussianPlateDecoder.LETTER_FIXUPS
    RF_DIGIT_FIXUPS = _RussianPlateDecoder.DIGIT_FIXUPS
    RF_REGION_CODES = _RussianPlateDecoder.REGION_CODES
    LETTER_SLOT_FALLBACKS = _RussianPlateDecoder.LETTER_SLOT_FALLBACKS
    DIGIT_SLOT_FALLBACKS = _RussianPlateDecoder.DIGIT_SLOT_FALLBACKS

    def __init__(
        self,
        weights_path: str = "models/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
        enable_ocr: bool = True,
        ocr_languages: Sequence[str] | str = ("ru", "en"),
        ocr_gpu: bool = False,
        ocr_min_score: float = 0.25,
        ocr_fast: bool = False,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.enable_ocr = enable_ocr
        self.ocr_languages = self._parse_ocr_languages(ocr_languages)
        self.ocr_gpu = ocr_gpu
        self.ocr_min_score = ocr_min_score
        self.ocr_fast = ocr_fast
        self.ocr_reader = None
        self._tracker = _PlateTracker(
            distance_threshold=90.0,
            max_missed_frames=8,
            max_votes=14,
        )
        self._ocr = _PlateCropOcr(reader=None, min_accept_score=ocr_min_score, ocr_fast=ocr_fast)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {self.weights_path}. Train model first."
            )

        LOGGER.info("Initializing model from %s", self.weights_path)
        self.model = YOLO(str(self.weights_path))

        if self.enable_ocr:
            self.ocr_reader = self._init_easyocr_reader()
            self._ocr = _PlateCropOcr(
                reader=self.ocr_reader,
                min_accept_score=self.ocr_min_score,
                ocr_fast=self.ocr_fast,
            )

    @staticmethod
    def _parse_ocr_languages(ocr_languages: Sequence[str] | str) -> list[str]:
        if isinstance(ocr_languages, str):
            parts = [p.strip() for p in ocr_languages.split(",")]
        else:
            parts = [str(x).strip() for x in ocr_languages]
        return [p for p in parts if p]

    def _init_easyocr_reader(self):
        if easyocr is None:
            LOGGER.warning("easyocr is not installed; OCR is disabled")
            return None
        try:
            reader = easyocr.Reader(self.ocr_languages, gpu=self.ocr_gpu, verbose=False)
            LOGGER.info("OCR initialized | languages=%s gpu=%s", self.ocr_languages, self.ocr_gpu)
            return reader
        except Exception as exc:
            LOGGER.exception("Failed to initialize OCR reader: %s", exc)
            return None

    @staticmethod
    def _expand_crop(frame: np.ndarray, bbox: list[float], padding_ratio: float = 0.08) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)
        left = max(int(x1) - pad_x, 0)
        top = max(int(y1) - pad_y, 0)
        right = min(int(x2) + pad_x, w)
        bottom = min(int(y2) + pad_y, h)
        return frame[top:bottom, left:right]

    def detect_plates(self, frame: np.ndarray) -> list[dict]:
        out: list[dict] = []
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            if not results:
                return out

            result = results[0]
            boxes = result.boxes
            if boxes is None:
                return out

            names = result.names
            matched: set[int] = set()

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                det_conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())

                crop = self._expand_crop(frame, xyxy)
                plate_text, plate_conf = self._ocr.read_plate_text(crop)

                tid = self._tracker.assign_track_id(xyxy, already_used=matched)
                matched.add(tid)
                self._tracker.update_track(tid, xyxy, plate_text, plate_conf)

                stable, stable_conf = self._tracker.consensus_for(tid)
                display_text = stable if stable else plate_text
                display_conf = stable_conf if stable is not None else plate_conf
                # На камере текущий кадр часто «проваливает» OCR; показываем последний удачный текст по треку.
                if not display_text:
                    fallback = self._tracker.last_plate_reading(tid)
                    if fallback:
                        display_text, display_conf = fallback

                out.append(
                    {
                        "bbox": [float(v) for v in xyxy],
                        "confidence": det_conf,
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, "number"),
                        "plate_text": display_text,
                        "plate_text_confidence": display_conf,
                        "track_id": tid,
                    }
                )

            self._tracker.mark_missed_and_prune(matched)
            LOGGER.info("Frame processed | detections=%d", len(out))
            return out

        except Exception as exc:
            LOGGER.exception("Error during detect_plates: %s", exc)
            return []
