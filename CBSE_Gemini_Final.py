# ╔══════════════════════════════════════════════════════════════════╗
# ║       CBSE CLASS 10 ENGLISH — RAG ANSWER KEY GENERATOR          ║
# ║       Gemini API | RAG for Literature Only | VS Code Ready      ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# SETUP (run once in terminal):
#   python -m venv D:\cbse_venv
#   D:\cbse_venv\Scripts\activate
#   pip install pdfplumber sentence-transformers chromadb google-generativeai
#         reportlab fuzzywuzzy python-Levenshtein pytesseract pdf2image pillow tqdm
#
# Then open this file in VS Code and select D:\cbse_venv as the interpreter.


# ════════════════════════════════════════════════════════════════════
# CELL 1 — INSTALL & CONFIG
# ════════════════════════════════════════════════════════════════════

# ---------- VS CODE (run these in terminal, NOT here) ---------------
# python -m venv D:\cbse_venv
# D:\cbse_venv\Scripts\activate
# pip install pdfplumber sentence-transformers chromadb google-generativeai
#     reportlab fuzzywuzzy python-Levenshtein pytesseract pdf2image pillow tqdm
# Also install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
# After install, set TESSERACT_CMD below to the install path
# --------------------------------------------------------------------

import os

# ── YOUR GEMINI API KEY ─────────────────────────────────────────────
# Get free key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "your key"

# ── PATHS  (change these to match your local machine) ───────────────
BASE      = r"D:\CBSE_Project"              # Project root on D drive
TEXTBOOKS = os.path.join(BASE, "textbooks")          # Folder with 29 textbook PDFs
PAPERS    = os.path.join(BASE, "question_papers")    # Folder with question paper
DB_PATH   = os.path.join(BASE, "chroma_db")          # ChromaDB persistent storage
OUTPUT    = os.path.join(BASE, "outputs")             # Where ANSWER_KEY.pdf is saved

# ── TESSERACT PATH (for OCR on scanned PDFs) ────────────────────────
# Windows default install path — change if yours is different
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── QUESTION PAPER FILENAME ─────────────────────────────────────────
QUESTION_PAPER = "sample_paper.pdf"

# ── MODEL SETTINGS ──────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"   # Paid key — no rate limits
EMBED_MODEL  = "all-MiniLM-L6-v2"  # Free local embeddings (no API needed)
CHUNK_SIZE   = 3000
TOP_K        = 5

print("✅ Config ready")
print(f"   Base     : {BASE}")
print(f"   Textbooks: {TEXTBOOKS}")
print(f"   Output   : {OUTPUT}")


# ════════════════════════════════════════════════════════════════════
# CELL 2 — IMPORTS
# ════════════════════════════════════════════════════════════════════

import re, time, warnings
from pathlib import Path
from typing  import List, Dict, Optional, Tuple
warnings.filterwarnings('ignore')

import pdfplumber
import pytesseract
from PIL        import Image
from pdf2image  import convert_from_path

import google.generativeai as genai

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from fuzzywuzzy import fuzz, process
from tqdm.auto  import tqdm

from reportlab.lib.pagesizes  import A4
from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units      import inch
from reportlab.platypus       import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib            import colors
from reportlab.lib.enums      import TA_JUSTIFY, TA_CENTER

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

print("✅ All imports done")
print(f"   Gemini model: {GEMINI_MODEL}")


# ════════════════════════════════════════════════════════════════════
# CELL 3 — ALL CLASSES
# ════════════════════════════════════════════════════════════════════

# ── Chapter name mapping (filename → chapter title) ─────────────────
FILENAME_TO_CHAPTER = {
    "jewe201": "A Letter to God",
    "jewe202": "Nelson Mandela: Long Walk to Freedom",
    "jewe203": "Two Stories About Flying",
    "jewe204": "From the Diary of Anne Frank",
    "jewe205": "Glimpses of India",
    "jewe206": "Mijbil the Otter",
    "jewe207": "Madam Rides the Bus",
    "jewe208": "The Sermon at Benares",
    "jewe209": "The Proposal",
    "jefp1ps": "Footprints Without Feet",
    "jewe2ps": "First Flight Workbook",
    "a_letter_to_god":                    "A Letter to God",
    "nelson_mandela_long_walk_to_freedom": "Nelson Mandela: Long Walk to Freedom",
    "two_stories_about_flying":            "Two Stories About Flying",
    "from_the_diary_of_anne_frank":        "From the Diary of Anne Frank",
    "glimpses_of_india":                   "Glimpses of India",
    "mijbil_the_otter":                    "Mijbil the Otter",
    "madam_rides_the_bus":                 "Madam Rides the Bus",
    "the_sermon_at_benares":               "The Sermon at Benares",
    "the_proposal":                        "The Proposal",
    "the_necklace":                        "The Necklace",
    "bholi":                               "Bholi",
    "the_midnight_visitor":                "The Midnight Visitor",
    "a_question_of_trust":                 "A Question of Trust",
    "footprints_without_feet":             "Footprints Without Feet",
    "the_thiefs_story":                    "The Thief's Story",
    "the_book_that_saved_the_earth":       "The Book That Saved the Earth",
    "a_triumph_of_surgery":                "A Triumph of Surgery",
    "the_making_of_a_scientist":           "The Making of a Scientist",
    "the_hack_driver":                     "The Hack Driver",
}

def chapter_from_filename(pdf_path: str) -> str:
    name = os.path.basename(pdf_path).replace('.pdf', '').replace('-', '_').strip()
    key  = name.lower().replace(' ', '_')
    if key in FILENAME_TO_CHAPTER:
        return FILENAME_TO_CHAPTER[key]
    for k, chapter in FILENAME_TO_CHAPTER.items():
        if key.startswith(k) or k in key:
            return chapter
    return name.replace('_', ' ').title()


# ────────────────────────────────────────────────────────────────────
# CLASS 1: PDFExtractor — supports normal + scanned PDFs (OCR)
# ────────────────────────────────────────────────────────────────────
class PDFExtractor:
    """
    Extracts text from PDFs.
    - Normal PDFs  → pdfplumber (fast)
    - Scanned PDFs → Tesseract OCR (auto-detected)
    """
    MIN_TEXT = 100  # chars; below this = scanned PDF → use OCR

    def extract(self, path: str) -> str:
        text = self._text(path)
        if len(text.strip()) < self.MIN_TEXT:
            print(f"  [OCR] {os.path.basename(path)}")
            text = self._ocr(path)
        return text.strip()

    def _text(self, path: str) -> str:
        out = ""
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        out += t + "\n\n"
        except Exception as e:
            print(f"  [WARN] pdfplumber: {e}")
        return out

    def _ocr(self, path: str) -> str:
        out = ""
        try:
            imgs = convert_from_path(path, dpi=300, thread_count=2)
            for img in imgs:
                out += pytesseract.image_to_string(img, lang='eng',
                           config='--psm 6 --oem 3') + "\n\n"
        except Exception as e:
            print(f"  [ERROR] OCR: {e}")
        return out

    def chunk(self, text: str) -> List[str]:
        paras = [p.strip() for p in re.split(r'\n\s*\n', text)
                 if p.strip() and len(p.strip()) > 50]
        chunks, current = [], ""
        for para in paras:
            if len(current) + len(para) > CHUNK_SIZE and current:
                chunks.append(current.strip())
                sents   = current.split('. ')
                overlap = '. '.join(sents[-2:]) if len(sents) > 2 else ""
                current = (overlap + " " + para).strip() if overlap else para
            else:
                current += ("\n\n" + para) if current else para
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def chapter_name(self, filename: str) -> str:
        """Fallback chapter name from filename (used when folder-based mapping fails)."""
        name = Path(filename).stem
        name = re.sub(r'^(chapter|ch|lesson|unit)[\s_-]*\d*[\s_-]*', '', name, flags=re.I)
        return re.sub(r'[\s_-]+', ' ', name).strip().title()


# ────────────────────────────────────────────────────────────────────
# CLASS 2: VectorDB — ChromaDB with local sentence-transformer embeddings
# (Used ONLY for literature questions — RAG is skipped for other types)
# ────────────────────────────────────────────────────────────────────
class VectorDB:
    """
    Persistent vector store.
    Ingest textbooks once, reuse across runs (saved to chroma_db folder).
    RAG retrieval is called ONLY for literature questions (Q6-Q11).
    """
    def __init__(self):
        print("📥 Loading embedding model (first time ~1 min)...")
        self.model      = SentenceTransformer(EMBED_MODEL)
        self.client     = chromadb.PersistentClient(
            path=DB_PATH, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name="cbse_final", metadata={"hnsw:space": "cosine"})
        print(f"✅ VectorDB ready | Chunks: {self.collection.count()}")

    def embed(self, text: str) -> List[float]:
        return self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True).tolist()

    def is_ingested(self, source: str) -> bool:
        try:
            r = self.collection.get(where={"source": {"$eq": source}}, limit=1)
            return len(r['ids']) > 0
        except Exception:
            return False

    def ingest(self, textbooks_path: str, extractor: PDFExtractor) -> int:
        """Walk textbooks folder, ingest all PDFs. Skips already-ingested files."""
        print("\n" + "="*60)
        print("INGESTING TEXTBOOKS (runs once, then cached)")
        print("="*60)

        pdf_files = []
        for root, _, files in os.walk(textbooks_path):
            for f in sorted(files):
                if f.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, f))

        print(f"Found {len(pdf_files)} PDFs")
        total = 0

        for pdf_path in pdf_files:
            fname   = os.path.basename(pdf_path)
            chapter = chapter_from_filename(pdf_path)

            if self.is_ingested(fname):
                print(f"  ⏭️  Already ingested: {chapter}")
                continue

            print(f"  📚 {chapter:<45}", end="", flush=True)
            text = extractor.extract(pdf_path)

            if not text or len(text) < 100:
                print(" → SKIPPED (empty)")
                continue

            chunks = extractor.chunk(text)
            print(f" → {len(chunks)} chunks", end="", flush=True)

            for i in range(0, len(chunks), 50):  # batch of 50
                batch = chunks[i:i+50]
                self.collection.add(
                    embeddings=[self.embed(c) for c in batch],
                    documents=batch,
                    metadatas=[{"chapter": chapter, "source": fname} for _ in batch],
                    ids=[f"{Path(pdf_path).stem}_{i+j}" for j in range(len(batch))]
                )
                total += len(batch)
            print(" ✅")

        print(f"\n✅ Done | Added: {total} | Total: {self.collection.count()}")
        return self.collection.count()

    def get_all_chapters(self) -> List[str]:
        try:
            data = self.collection.get()
            return list(set(m['chapter'] for m in data['metadatas']))
        except Exception:
            return []


# ────────────────────────────────────────────────────────────────────
# CLASS 3: QuestionExtractor
# ────────────────────────────────────────────────────────────────────
class QuestionExtractor:
    def __init__(self, extractor: PDFExtractor):
        self.extractor = extractor
        self.passages  = {}

    def extract(self, path: str) -> List[Dict]:
        print("\n" + "="*60)
        print("EXTRACTING QUESTIONS FROM QUESTION PAPER")
        print("="*60)

        text = self.extractor.extract(path)
        if not text:
            print("❌ Could not extract text from question paper!")
            return []

        self._extract_passages(text)
        questions, seen = [], set()

        def add(q):
            if q and q.get('number') not in seen and q.get('question'):
                questions.append(q); seen.add(q['number'])

        # ── Section A: Q1 (8 sub-questions, saffron passage) ────────
        # Q1 ends where Q2 passage starts — use large block to capture all 8
        ans_pos = [m.start() for m in re.finditer(r"Answer the following questions", text, re.I)]
        q2_start = self._find(text, ["2. Read the following passage", "2.\tRead the following"])

        if len(ans_pos) >= 1:
            # Q1 block: from "Answer the following questions" to start of Q2 passage (or +6000)
            q1_end = q2_start if q2_start > ans_pos[0] else ans_pos[0] + 6000
            for q in self._extract_subqs(text[ans_pos[0]:q1_end], "1", "comprehension", 40):
                add(q)

        # ── Section A: Q2 (9 sub-questions, silk passage) ────────────
        if len(ans_pos) >= 2:
            # Q2 block: from second "Answer the following questions" to start of Section B
            sec_b = self._find(text, ["SECTION B", "Section B", "Grammar"])
            q2_end = sec_b if sec_b > ans_pos[1] else ans_pos[1] + 7000
            for q in self._extract_subqs(text[ans_pos[1]:q2_end], "2", "comprehension", 40):
                add(q)

        # ── Section B: Q3 (12 grammar tasks, complete any 10) ────────
        q3s = self._find(text, ["3. Complete any ten", "3. Attempt any ten",
                                 "Complete any ten", "Complete any TEN"])
        if q3s >= 0:
            # Q3 ends at Writing section header
            q3e = self._find(text[q3s:], ["Writing\n", "4. Attempt", "4.\tAttempt"])
            q3e = (q3s + q3e) if q3e > 0 else q3s + 8000
            for q in self._extract_subqs(text[q3s:q3e], "3", "grammar", 25):
                add(q)

        # ── Section B: Q4, Q5 (Writing — both options a and b) ───────
        for qnum in [4, 5]:
            qs = self._find(text, [f"{qnum}. Attempt any one", f"{qnum}. Answer any one",
                                    f"{qnum}.\tAttempt"])
            if qs >= 0:
                # Large block to capture full writing task text
                block = text[qs:qs+3000]
                ma = re.search(r'\(a\)\s+(.+?)(?:\bOR\b)', block, re.DOTALL | re.I)
                mb = re.search(r'\bOR\b\s+\(b\)\s+(.+?)(?=\n\s*\d+\.|\Z)', block, re.DOTALL | re.I)
                if ma: add({"number": f"{qnum}(a)", "marks": 5, "type": "writing",
                             "question": self._clean(ma.group(1))[:800],
                             "chapter": None, "word_limit": 120})
                if mb: add({"number": f"{qnum}(b)", "marks": 5, "type": "writing",
                             "question": self._clean(mb.group(1))[:800],
                             "chapter": None, "word_limit": 120})

        # ── Section C: Q6, Q7 (Extract-based — both options a and b) ─
        for qnum in [6, 7]:
            qs = self._find(text, [f"{qnum}. Read the following extract",
                                    f"{qnum}. Read the extract",
                                    f"{qnum}.\tRead the following"])
            if qs < 0:
                continue

            # Find the NEXT question boundary (next top-level question number)
            next_q = re.search(rf'\n\s*{qnum+1}\s*[.\)]', text[qs:])
            qend   = (qs + next_q.start()) if next_q else qs + 6000
            block  = text[qs:qend]

            # Split on OR (which appears between option (a) and option (b))
            # OR appears as standalone word on its own line in CBSE papers
            or_split = re.split(r'\n\s*OR\s*\n', block, maxsplit=1, flags=re.I)

            if len(or_split) == 2:
                a_block_raw = or_split[0]
                b_block_raw = or_split[1]
            else:
                # Fallback: split on " OR " with surrounding whitespace
                or_split = re.split(r'\s{2,}OR\s{2,}', block, maxsplit=1)
                if len(or_split) == 2:
                    a_block_raw = or_split[0]
                    b_block_raw = or_split[1]
                else:
                    print(f"  ⚠️  Q{qnum}: Could not split a/b blocks")
                    continue

            # Strip the (a) / (b) prefix from each block
            ma = re.search(r'\(a\)\s*(.+)', a_block_raw, re.DOTALL)
            mb = re.search(r'\(b\)\s*(.+)', b_block_raw, re.DOTALL)

            if ma:
                a_block   = ma.group(1).strip()
                a_passage = self._extract_passage_from_block(a_block)
                a_chapter = self._extract_chapter_from_passage(a_block)
                print(f"  Q{qnum}a passage: {a_passage[:60]}... chapter={a_chapter}")
                for q in self._extract_subqs(a_block, f"{qnum}a", "literature", 40):
                    q['extract'] = a_passage
                    q['chapter'] = q.get('chapter') or a_chapter
                    add(q)

            if mb:
                b_block   = mb.group(1).strip()
                b_passage = self._extract_passage_from_block(b_block)
                b_chapter = self._extract_chapter_from_passage(b_block)
                print(f"  Q{qnum}b passage: {b_passage[:60]}... chapter={b_chapter}")
                for q in self._extract_subqs(b_block, f"{qnum}b", "literature", 40):
                    q['extract'] = b_passage
                    q['chapter'] = q.get('chapter') or b_chapter
                    add(q)

        # ── Section C: Q8 (5 questions, answer any four, 3 marks each) ─
        q8s = self._find(text, ["8. Answer any four", "8. Attempt any four",
                                  "Answer any four", "Answer any FOUR"])
        if q8s >= 0:
            for q in self._extract_subqs(text[q8s:q8s+3000], "8", "literature", 50):
                q['marks'] = 3  # always 3 marks each for Q8
                add(q)

        # ── Section C: Q9 (3 questions, answer any two, 3 marks each) ─
        q9s = self._find(text, ["9. Answer any two", "9. Attempt any two",
                                  "Answer any two", "Answer any TWO"])
        if q9s >= 0:
            for q in self._extract_subqs(text[q9s:q9s+2000], "9", "literature", 50):
                q['marks'] = 3  # always 3 marks each for Q9
                add(q)

        # ── Section C: Q10, Q11 (Long answer, 6 marks, both options) ─
        for qnum in [10, 11]:
            qs = self._find(text, [f"{qnum}. Answer any one", f"{qnum}. Attempt any one",
                                    f"{qnum}.\tAnswer any one"])
            if qs >= 0:
                block = text[qs:qs+2500]
                ma = re.search(r'\(a\)\s+(.+?)(?:\bOR\b)', block, re.DOTALL | re.I)
                mb = re.search(r'\bOR\b\s+\(b\)\s+(.+?)(?=\n\s*\d+\.|\Z)', block, re.DOTALL | re.I)
                if ma: add({"number": f"{qnum}(a)", "marks": 6, "type": "literature",
                             "question": self._clean(ma.group(1))[:800],
                             "chapter": self._extract_chapter(ma.group(1)), "word_limit": 120})
                if mb: add({"number": f"{qnum}(b)", "marks": 6, "type": "literature",
                             "question": self._clean(mb.group(1))[:800],
                             "chapter": self._extract_chapter(mb.group(1)), "word_limit": 120})

        print(f"✅ Extracted {len(questions)} questions")
        return questions

    # ── Helpers ──────────────────────────────────────────────────────
    def _find(self, text: str, patterns: List[str]) -> int:
        for p in patterns:
            i = text.find(p)
            if i >= 0: return i
        return -1

    def _extract_passages(self, text: str):
        p1 = list(re.finditer(r'\(1\)\s+[A-Z]', text))
        ap = [m.start() for m in re.finditer(r"Answer the following questions", text, re.I)]
        if len(p1) >= 1 and len(ap) >= 1:
            self.passages['1'] = text[p1[0].start():ap[0]].strip()
        if len(p1) >= 2 and len(ap) >= 2:
            self.passages['2'] = text[p1[1].start():ap[1]].strip()
        print(f"  Passages found: {len(self.passages)}")

    def _extract_subqs(self, block: str, parent: str, qtype: str, wlimit: int) -> List[Dict]:
        questions = []
        pat = re.compile(r'\(([ivxIVX]+)\)\s+(.+?)(?=\n\s*\((?:[ivxIVX]+|[a-d])\)|\Z)', re.DOTALL)
        for m in pat.finditer(block):
            roman, qtext = m.groups()
            qtext = self._clean(qtext)
            if len(qtext) < 10: continue
            questions.append({
                "number":     f"{parent}({roman.lower()})",
                "marks":      self._extract_marks(qtext),
                "type":       qtype,
                "question":   qtext[:700],
                "chapter":    self._extract_chapter(qtext),
                "word_limit": self._extract_wlimit(qtext) or wlimit,
            })
        return questions

    def _extract_passage_from_block(self, block: str) -> str:
        """Extract the prose/poem passage from a Q6/Q7 option block,
        stopping before the sub-questions start."""
        m = re.search(r'\n\s*\(i\)', block)
        if m:
            passage = block[:m.start()].strip()
        else:
            passage = block[:800].strip()
        return self._clean(passage)[:1500]

    def _extract_chapter_from_passage(self, block: str) -> Optional[str]:
        """Extract chapter name from the source attribution in the block,
        e.g. '(A Letter to God)' or '(The Thief's Story)'"""
        # Look for chapter name in parentheses — typically at end of extract
        matches = re.findall(r'\(([A-Z][A-Za-z\s\'–\-]{4,60})\)', block)
        for m in matches:
            # Skip option labels like (A), (B), MCQ options, etc.
            if re.match(r'^[A-D]$', m.strip()):
                continue
            if len(m.strip()) > 5:
                return m.strip()
        return None

    def _extract_marks(self, text: str) -> int:
        m = re.search(r'\b(\d{1,2})\s*$', text.strip())
        return int(m.group(1)) if m and int(m.group(1)) <= 10 else 1

    def _extract_wlimit(self, text: str) -> Optional[int]:
        m = re.search(r'(?:in|about)\s+(\d+)\s*(?:[-–]\s*\d+)?\s*words', text, re.I)
        return int(m.group(1)) if m else None

    def _extract_chapter(self, text: str) -> Optional[str]:
        m = re.search(r'\(([A-Z][^()]{5,60})\)', text)
        return m.group(1).strip() if m and not re.match(r'^[A-D]$', m.group(1)) else None

    def _clean(self, text: str) -> str:
        text = re.sub(r'Page \d+|P\.T\.O\.|2/1/1|#|\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+\d{1,2}\s*$', '', text)
        return text.strip()


# ────────────────────────────────────────────────────────────────────
# CLASS 4: RAGRetriever
# Called ONLY for literature questions (Q6–Q11)
# ────────────────────────────────────────────────────────────────────
class RAGRetriever:
    """
    Retrieves relevant textbook chunks.
    Strategy:
      1. Chapter-filtered search (if chapter hint available, fuzz match ≥ 60)
      2. Global search (fallback)
      3. Any chunks (last resort)
    """
    def __init__(self, db: VectorDB):
        self.db       = db
        self.chapters = db.get_all_chapters()
        print(f"✅ Retriever ready | {len(self.chapters)} chapters")

    def retrieve(self, question: str, chapter_hint: Optional[str] = None) -> Tuple[str, str]:
        q_emb = self.db.embed(question)
        n     = min(TOP_K, self.db.collection.count())

        # 1. Chapter-filtered
        if chapter_hint and self.chapters:
            matched = process.extractOne(chapter_hint, self.chapters, scorer=fuzz.token_sort_ratio)
            if matched and matched[1] >= 60:
                try:
                    r = self.db.collection.query(
                        query_embeddings=[q_emb], n_results=n,
                        where={"chapter": {"$eq": matched[0]}})
                    if r['documents'][0]:
                        return "\n\n".join(r['documents'][0]), f"Ch:{matched[0][:30]}"
                except Exception:
                    pass

        # 2. Global
        try:
            r = self.db.collection.query(query_embeddings=[q_emb], n_results=n)
            if r['documents'][0]:
                return "\n\n".join(r['documents'][0]), "Global"
        except Exception:
            pass

        # 3. Fallback
        try:
            d = self.db.collection.get(limit=TOP_K)
            if d['documents']:
                return "\n\n".join(d['documents']), "Fallback"
        except Exception:
            pass

        return "", "None"


# ────────────────────────────────────────────────────────────────────
# CLASS 5: AnswerGenerator (Gemini only)
#
# KEY DESIGN DECISION:
#   • Comprehension (Q1, Q2) → Gemini with passage context (NO RAG)
#   • Grammar (Q3)           → Gemini with question only (NO RAG)
#   • Writing (Q4, Q5)       → Gemini with question only (NO RAG)
#   • Literature (Q6–Q11)    → Gemini WITH RAG context from textbooks
#
# This is smarter: RAG is only useful when we need textbook quotes/evidence.
# For comprehension the passage IS the context. For grammar/writing, no context needed.
# ────────────────────────────────────────────────────────────────────
class AnswerGenerator:
    SYSTEM = (
        "You are a CBSE Class 10 English expert. "
        "Write precise, exam-appropriate answers strictly following CBSE marking scheme. "
        "STRICT RULES: "
        "1. Never start with preamble like 'Here is...', 'Sure...', 'Certainly...', 'Here's an analytical paragraph...'. "
        "2. Never add word counts at the end like '[43 words]' or '(45 words)'. "
        "3. Never add mark indicators like '(1)' in the answer. "
        "4. Start the answer directly. "
        "5. Answer ONLY the specific question asked — do not repeat answers from other questions."
    )

    def __init__(self, passages: Dict):
        self.model    = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=self.SYSTEM
        )
        self.passages = passages
        self.calls    = 0

    def generate(self, q: Dict, rag_context: str = "") -> str:
        """
        Generate an answer.
        rag_context is only passed (non-empty) for literature questions.
        """
        qtext  = q['question']
        wlimit = q.get('word_limit', max(q['marks'] * 20, 40))

        # Detect sub-type
        is_mcq   = bool(re.search(r'\(A\).*?\(B\).*?\(C\).*?\(D\)', qtext, re.DOTALL))
        is_tf    = bool(re.search(r'true or false|state whether', qtext, re.I))
        is_err   = bool(re.search(r'error.*correction|find.*error|identify.*error', qtext, re.I))
        is_fill  = bool(re.search(r'fill in|complete the.*blank|from the brackets|blank with', qtext, re.I))
        is_40w   = bool(re.search(r'in about 40|40[\-\s]?50 words?|40 words?', qtext, re.I))
        is_order = bool(re.search(r'rearrange|reorder|correct order', qtext, re.I))

        # For comprehension: use actual passage, not RAG
        if q['type'] == 'comprehension':
            key = '1' if q['number'].startswith('1') else '2'
            context = self.passages.get(key, "")[:3000]
        elif q['type'] == 'literature':
            extract = q.get('extract', '')
            if extract:
                # Q6/Q7: use the extract from question paper as PRIMARY context
                # RAG context is secondary — appended only if extract is short
                if len(extract.split()) > 30:
                    context = extract  # extract alone is enough
                else:
                    context = extract + "\n\n" + rag_context[:1000]
            else:
                context = rag_context  # Q8–Q11: use RAG
        else:
            context = ""  # Grammar & Writing: no context needed

        prompt = self._build_prompt(q, context, wlimit,
                                    is_mcq, is_tf, is_err, is_fill, is_40w, is_order)

        # Call Gemini with retry
        for attempt in range(3):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=1000,
                    )
                )
                self.calls += 1
                answer = response.text.strip()

                # Post-process
                return self._postprocess(answer, wlimit, is_mcq, is_tf, is_err, is_fill, is_40w)

            except Exception as e:
                err = str(e).lower()
                if '429' in err or 'quota' in err or 'rate' in err or 'resource' in err:
                    wait = 5  # short wait even for paid keys (transient errors)
                    print(f"\n  [Transient error] Waiting {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                elif '400' in err or 'api_key' in err:
                    print(f"\n  [ERROR] Invalid Gemini API key — check GEMINI_API_KEY in Cell 1")
                    return "[Auth failed — check API key]"
                else:
                    wait = (attempt + 1) * 10
                    print(f"\n  [Error] {e} | Wait {wait}s...", end=" ", flush=True)
                    time.sleep(wait)

        return "[Generation failed — will retry]"

    def _build_prompt(self, q, context, wlimit,
                      is_mcq, is_tf, is_err, is_fill, is_40w, is_order) -> str:
        qtext = q['question']
        has_extract = bool(q.get('extract'))
        ctx_label   = "EXTRACT FROM QUESTION PAPER" if has_extract else "PASSAGE/CONTEXT"

        if is_mcq:
            ctx_block = f"\n{ctx_label}:\n{context}\n" if context else ""
            return f"""Read the extract/passage carefully and select the BEST answer.
{ctx_block}
QUESTION:
{qtext}

Carefully read EACH option against the extract. Choose the most accurate one.
Reply with ONLY: (A) or (B) or (C) or (D)
ANSWER:"""

        elif is_tf:
            return f"""Read the passage carefully word by word, then answer True or False.

PASSAGE:
{context}

QUESTION:
{qtext}

IMPORTANT: The statement must be EXACTLY supported by the passage to be True.
If the passage says something DIFFERENT (even slightly), the answer is False.
For example: if the passage says 'domesticated' but the statement says 'used in kitchens' — that is False.

Reply with ONLY: True  OR  False
ANSWER:"""

        elif is_err:
            return f"""Grammar error correction.

{qtext}

Reply in EXACTLY this format:
Error: [wrong word or phrase]
Correction: [correct word or phrase]

Nothing else.
ANSWER:"""

        elif is_fill:
            ctx_block = f"\n{ctx_label}:\n{context[:1500]}\n" if context else ""
            opts = re.search(r'\(([a-zA-Z]+/[a-zA-Z]+(?:/[a-zA-Z]+)*)\)', qtext)
            opts_line = f"\nChoose from: {opts.group(0)}" if opts else ""
            return f"""Fill in the blank with the correct word.{opts_line}
{ctx_block}
QUESTION: {qtext}

Reply with ONLY the single correct word. No explanation.
ANSWER:"""

        elif is_40w:
            ctx_block = f"\n{ctx_label}:\n{context[:2000]}\n" if context else ""
            return f"""Write a CBSE answer in 40 to 50 words. No preamble. No word count at the end.
{ctx_block}
QUESTION:
{qtext}

Write 40-50 words. Start the answer directly.
ANSWER:"""

        elif is_order:
            return f"""Rearrange into the correct order.

{qtext}

Give the correctly ordered sequence only. No preamble.
ANSWER:"""

        elif q['type'] == 'writing':
            return f"""Write a CBSE Class 10 English answer for this writing task. Start directly with the answer format (letter/paragraph). No preamble like "Here is..." or "Sure...".

QUESTION:
{qtext}

INSTRUCTIONS:
- Write approximately {wlimit} words
- Use proper format (letter/notice/paragraph as needed)
- Formal, exam-appropriate language
- Follow CBSE conventions
- Start directly with the answer, no introduction sentence
ANSWER:"""

        elif q['type'] == 'literature' and q['marks'] >= 6:
            return f"""Write a CBSE Class 10 long answer worth {q['marks']} marks. Start directly with the answer. No preamble. No word count at the end.

TEXTBOOK CONTEXT:
{context[:2500]}

QUESTION:
{qtext}

INSTRUCTIONS:
- Write approximately {wlimit} words
- Structure: Introduction → Analysis with evidence → Conclusion
- Quote or refer to the text where relevant
- Answer ONLY this specific question: {qtext[:100]}
ANSWER:"""

        elif q['type'] == 'literature' and q.get('extract'):
            # Q6/Q7 extract-based subquestion — use ONLY the given extract
            chapter_label = f" from '{q.get('chapter')}'" if q.get('chapter') else ""
            return f"""Answer this CBSE Class 10 English question based ONLY on the following extract{chapter_label}.
Start directly. No preamble. No word count.

EXTRACT:
{context[:2000]}

QUESTION:
{qtext}

Use ONLY the above extract to answer. Write in maximum {wlimit} words.
ANSWER:"""

        else:  # literature short answer (Q8/Q9) — uses RAG
            ctx_block = f"\nTEXTBOOK CONTEXT:\n{context[:2000]}\n" if context else ""
            return f"""Write a CBSE Class 10 English answer worth {q['marks']} marks. Start directly. No preamble. No word count at the end.
{ctx_block}
QUESTION:
{qtext}

Answer ONLY this specific question. Write in maximum {wlimit} words.
ANSWER:"""

    def _postprocess(self, answer, wlimit, is_mcq, is_tf, is_err, is_fill, is_40w) -> str:
        # Remove markdown bold/italic
        answer = re.sub(r'\*\*(.+?)\*\*', r'\1', answer)
        answer = re.sub(r'\*(.+?)\*',     r'\1', answer)

        # Remove "Answer:" / "ANSWER:" prefix
        answer = re.sub(r'^(Answer|ANSWER|The answer is):\s*', '', answer, flags=re.I)

        # Remove word counts like [43 words] or (45 words) or [98 words]
        answer = re.sub(r'\[?\(?\d+\s+words?\)?\]?', '', answer, flags=re.I)

        # Remove inline mark markers like (1) or (2) at end of sentences
        answer = re.sub(r'\s*\(\d\)\s*', ' ', answer)

        # Remove preamble lines like "Here's an analytical paragraph..." or
        # "Here is a CBSE answer..." or "Sure, here is..."
        lines = answer.strip().split('\n')
        clean_lines = []
        skip_patterns = re.compile(
            r"^(here'?s|here is|sure|certainly|below is|the following|"
            r"this is|as requested|i have written|i will write|"
            r"analytical paragraph|suitable for cbse)",
            re.I
        )
        for line in lines:
            if skip_patterns.match(line.strip()):
                continue
            clean_lines.append(line)
        answer = '\n'.join(clean_lines).strip()

        # Re-strip any leading/trailing whitespace
        answer = answer.strip()

        if is_mcq:
            m = re.search(r'\(([A-D])\)', answer)
            if m: return f"({m.group(1)})"
            m = re.search(r'\b([A-D])\b', answer)
            return f"({m.group(1)})" if m else answer

        if is_tf:
            return "False" if re.search(r'\bfalse\b', answer, re.I) else "True"

        if is_err:
            em = re.search(r'Error:\s*(.+)',       answer, re.I)
            cm = re.search(r'Correction:\s*(.+)',  answer, re.I)
            if em and cm:
                return f"Error: {em.group(1).strip()}\nCorrection: {cm.group(1).strip()}"
            return answer

        if is_fill:
            return answer.split('\n')[0].strip()

        if is_40w:
            words = answer.split()
            if len(words) > 55:
                answer = ' '.join(words[:50])
                lp = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
                if lp > len(answer) * 0.6: answer = answer[:lp+1]
            return answer.strip()

        # Descriptive: truncate only if WAY over limit
        if wlimit:
            words = answer.split()
            if len(words) > wlimit * 2.5:
                answer = ' '.join(words[:int(wlimit * 2.0)])
                lp = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
                if lp > len(answer) * 0.7: answer = answer[:lp+1]

        return answer.strip()


# ────────────────────────────────────────────────────────────────────
# CLASS 6: Validator
# ────────────────────────────────────────────────────────────────────
class Validator:
    def validate(self, q: Dict, answer: str) -> Tuple[bool, str]:
        if not answer or 'failed' in answer.lower() or 'retry' in answer.lower():
            return False, "failed"

        words  = len(answer.split())
        qtext  = q['question']
        wlimit = q.get('word_limit', 40)

        if bool(re.search(r'\(A\).*?\(B\).*?\(C\).*?\(D\)', qtext, re.DOTALL)):
            return (True, "ok") if re.match(r'^\([A-D]\)$', answer.strip()) else (False, "mcq_format")
        if re.search(r'true or false', qtext, re.I):
            return (True, "ok") if answer.strip() in ['True', 'False'] else (False, "tf_format")
        if re.search(r'fill in|blank', qtext, re.I):
            return (True, "ok") if 1 <= words <= 15 else (False, f"fill({words}w)")
        if re.search(r'error.*correction', qtext, re.I):
            return (True, "ok") if ('Error:' in answer and 'Correction:' in answer) else (False, "err_format")
        if re.search(r'in about 40|40[\-\s]?50 words?', qtext, re.I):
            return (True, "ok") if 35 <= words <= 65 else (False, f"40w({words}w)")
        if words < max(3, int(wlimit * 0.25)):
            return False, f"short({words}w)"
        return True, "ok"


# ────────────────────────────────────────────────────────────────────
# CLASS 7: Exporter — formatted PDF with Section A/B/C headers
# ────────────────────────────────────────────────────────────────────
class Exporter:
    def _section(self, qnum: str) -> Optional[Tuple[str, str]]:
        m = re.match(r'^(\d+)', qnum)
        if not m: return None
        n = m.group(1)
        if n in {'1', '2'}:                         return ('A', 'SECTION A — READING SKILLS')
        if n in {'3', '4', '5'}:                    return ('B', 'SECTION B — GRAMMAR AND CREATIVE WRITING')
        if n in {'6','7','8','9','10','11'}:         return ('C', 'SECTION C — LITERATURE')
        return None

    def export(self, answers: List[Dict], out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        pdf_path = os.path.join(out_dir, "ANSWER_KEY.pdf")

        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
            leftMargin=0.8*inch, rightMargin=0.8*inch,
            topMargin=1*inch,    bottomMargin=0.8*inch)

        title_s   = ParagraphStyle('T',   fontName='Helvetica-Bold', fontSize=18,
                                   alignment=TA_CENTER, spaceAfter=6,
                                   textColor=colors.HexColor('#1a1a2e'))
        sub_s     = ParagraphStyle('Sub', fontName='Helvetica',      fontSize=12,
                                   alignment=TA_CENTER, spaceAfter=20,
                                   textColor=colors.HexColor('#4a4a6a'))
        section_s = ParagraphStyle('Sec', fontName='Helvetica-Bold', fontSize=12,
                                   spaceAfter=8, spaceBefore=18,
                                   textColor=colors.HexColor('#2c3e50'))
        qnum_s    = ParagraphStyle('Q',   fontName='Helvetica-Bold', fontSize=11,
                                   spaceAfter=4, spaceBefore=10,
                                   textColor=colors.HexColor('#c0392b'))
        ans_s     = ParagraphStyle('A',   fontName='Helvetica',      fontSize=11,
                                   leading=17, alignment=TA_JUSTIFY, spaceAfter=10)

        content = []
        content.append(Paragraph("CBSE CLASS 10 — ENGLISH LANGUAGE AND LITERATURE", title_s))
        content.append(Paragraph("Model Answer Key", sub_s))
        content.append(HRFlowable(width="100%", thickness=2,
                                  color=colors.HexColor('#3498db'), spaceAfter=20))

        current_section = None

        for a in answers:
            sec = self._section(a['number'])
            if sec and sec[0] != current_section:
                content.append(Spacer(1, 0.15*inch))
                content.append(Paragraph(sec[1], section_s))
                content.append(HRFlowable(width="100%", thickness=0.5,
                                          color=colors.grey, spaceAfter=10))
                current_section = sec[0]

            content.append(Paragraph(
                f"Q. {a['number']}  [{a['marks']} mark{'s' if a['marks']>1 else ''}]", qnum_s))

            safe = (a['answer']
                    .replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'))
            for line in safe.split('\n'):
                if line.strip():
                    content.append(Paragraph(line.strip(), ans_s))

        doc.build(content)
        print(f"✅ PDF saved: {pdf_path}")
        return pdf_path


print("✅ All classes ready")


# ════════════════════════════════════════════════════════════════════
# CELL 4 — RUN FULL PIPELINE
# ════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  CBSE RAG ANSWER KEY GENERATOR")
print("  Gemini API | RAG only for Literature")
print("="*60)

os.makedirs(OUTPUT, exist_ok=True)

# ── Step 1: Setup ────────────────────────────────────────────────────
pdf_ext = PDFExtractor()
db      = VectorDB()

# ── Step 2: Ingest textbooks ─────────────────────────────────────────
if db.collection.count() == 0:
    db.ingest(TEXTBOOKS, pdf_ext)
else:
    print(f"⏭️  Textbooks already ingested ({db.collection.count()} chunks)")

# ── Step 3: Extract questions ─────────────────────────────────────────
paper_path = os.path.join(PAPERS, QUESTION_PAPER)
print(f"\nQuestion paper: {paper_path}")

qextractor = QuestionExtractor(pdf_ext)
questions  = qextractor.extract(paper_path)

if not questions:
    print("❌ No questions extracted.")
    print("   Check: 1) File path is correct  2) PDF is readable  3) Tesseract is installed")
    raise SystemExit("Stopping — no questions found.")

# ── Step 4: Generate answers ──────────────────────────────────────────
retriever = RAGRetriever(db)
generator = AnswerGenerator(qextractor.passages)
validator = Validator()
exporter  = Exporter()

print(f"\n{'='*60}")
print(f"GENERATING ANSWERS FOR {len(questions)} QUESTIONS")
print(f"  • Comprehension → Gemini + passage (no RAG)")
print(f"  • Grammar       → Gemini only      (no RAG)")
print(f"  • Writing       → Gemini only      (no RAG)")
print(f"  • Literature    → Gemini + RAG     (textbook context)")
print(f"  Paid API key — no rate limits, running at full speed!")
print(f"{'='*60}")

answers      = []
regen_count  = 0
failed_count = 0

for i, q in enumerate(questions, 1):
    print(f"[{i:2}/{len(questions)}] Q{q['number']:8} ({q['type']:13} {q['marks']}m) → ",
          end="", flush=True)

    # RAG retrieval ONLY for literature questions
    if q['type'] == 'literature':
        rag_context, src = retriever.retrieve(q['question'], q.get('chapter'))
    else:
        rag_context, src = "", "no-rag"

    answer        = generator.generate(q, rag_context)
    valid, status = validator.validate(q, answer)

    # Immediate retry if invalid
    if not valid:
        print(f"regen({status}) → ", end="", flush=True)
        answer        = generator.generate(q, rag_context)
        valid, status = validator.validate(q, answer)
        regen_count  += 1

    if 'failed' in answer.lower():
        failed_count += 1
        print(f"❌ FAILED | {src}")
    else:
        print(f"✅ {len(answer.split())}w | {src}")

    answers.append({
        "number":   q['number'],
        "marks":    q['marks'],
        "type":     q['type'],
        "question": q['question'],
        "answer":   answer,
        "valid":    status,
    })

# ── Step 5: Retry failures ────────────────────────────────────────────
if failed_count > 0:
    print(f"\n{'='*60}")
    print(f"RETRYING {failed_count} FAILED QUESTIONS")
    print(f"{'='*60}")

    for i, a in enumerate(answers):
        if 'failed' in a['answer'].lower():
            q = next((qq for qq in questions if qq['number'] == a['number']), None)
            if not q: continue
            print(f"  Retrying Q{a['number']}... ", end="", flush=True)
            rag_context = retriever.retrieve(q['question'], q.get('chapter'))[0] \
                          if q['type'] == 'literature' else ""
            new_ans = generator.generate(q, rag_context)
            if 'failed' not in new_ans.lower():
                answers[i]['answer'] = new_ans
                answers[i]['valid']  = 'ok'
                failed_count        -= 1
                print(f"✅ {len(new_ans.split())}w")
            else:
                print("❌ still failed")

# ── Step 6: Export PDF ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("EXPORTING ANSWER KEY")
print(f"{'='*60}")

pdf_file    = exporter.export(answers, OUTPUT)
valid_count = sum(1 for a in answers if a['valid'] == 'ok')
accuracy    = round(valid_count * 100 / len(answers)) if answers else 0

print(f"\n{'='*60}")
print("  ✅ COMPLETE!")
print(f"{'='*60}")
print(f"  Questions   : {len(answers)}")
print(f"  Valid       : {valid_count}/{len(answers)}")
print(f"  Accuracy    : {accuracy}%")
print(f"  Regenerated : {regen_count}")
print(f"  Failed      : {failed_count}")
print(f"  Gemini calls: {generator.calls}")
print(f"  Answer Key  : {pdf_file}")
print(f"{'='*60}")
