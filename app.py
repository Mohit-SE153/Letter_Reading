# Letter_Processing_MultiPagePDF.py

import streamlit as st
from PIL import Image
from openai import OpenAI
import base64
import io
import json
import sqlite3
from datetime import datetime
import pytz
import re
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from dateutil.parser import parse
import os
from dotenv import load_dotenv
import fitz

try:
    from pdf2image import convert_from_bytes
    from pdf2image.exceptions import PDFInfoNotInstalledError
except ImportError:
    convert_from_bytes = None

# ─── IMPORTANT: Move DB_FILE to the VERY TOP ───────────────────────────────
# ─── IMPORTANT: Move DB_FILE to the VERY TOP ───────────────────────────────
DB_FILE = "my_data.db"

# ─── Safe one-time database initialization ─────────────────────────────────────
def init_database():
    # Open a fresh connection for initialization
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 1. Create cases table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            Case_ID TEXT PRIMARY KEY,
            Member_Name TEXT,
            NI_Number TEXT,
            Date_of_Birth TEXT,
            Old_Address TEXT,
            New_Address TEXT,
            Correspondence_Address TEXT,
            Pension_Account_Number TEXT,
            Case_Type TEXT,
            Case_Subtype TEXT,
            Case_Status TEXT,
            Pending_Reason TEXT,
            Expected_Document TEXT
        )
    ''')

    # 2. Create Supporting_Documents table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Supporting_Documents (
            Case_Type TEXT,
            Documents_Required TEXT,
            PRIMARY KEY (Case_Type, Documents_Required)
        )
    ''')

    conn.commit()

    # 3. Check if Supporting_Documents has any rows
    cursor.execute("SELECT COUNT(*) FROM Supporting_Documents")
    count = cursor.fetchone()[0]

    if count == 0:
        # Insert default rows only if table is empty (runs only once)
        supporting_data = [
            ("Address Change", "Council Tax Bill"),
            ("Address Change", "Driving licence"),
            ("Address Change", "Passport"),
            ("Lump Sum", "Passport"),
            ("Lump Sum", "Driving licence"),
            ("Death Case", "Death Certificate"),
            ("Payment Change", "Passport"),
            ("Payment Change", "Driving licence"),
            ("Retirement", "Passport"),
            ("Retirement", "Driving licence"),
            ("Retirement", "Birth Certificate"),
            ("Beneficiary Update", "Passport"),
            ("Beneficiary Update", "Driving licence"),
            ("Payment Setup", "Passport"),
            ("Payment Setup", "Driving Licence"),
            ("Divorce", "Marriage Certificate"),
        ]

        cursor.executemany(
            "INSERT OR IGNORE INTO Supporting_Documents (Case_Type, Documents_Required) VALUES (?, ?)",
            supporting_data
        )
        conn.commit()
        st.info("Supporting_Documents table was empty → inserted default rows.")

    # else:
    #     st.info("Supporting_Documents table already has data → skipping insert.")

    # Close the connection when done
    conn.close()

# Run initialization once when the app loads
init_database()

# ─── IMPORTANT: Replace with your actual OpenAI API key ───
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in the .env file or in Streamlit Cloud secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ─── IMPORTANT: Replace with your actual OpenAI API key ───
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in the .env file or in Streamlit Cloud secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ─── Make the app use more of the screen width ─────────────────────────────
st.set_page_config(
    page_title="AI Letter Processor",
    layout="wide",               # ← This is the key line: makes layout wide instead of centered/compact
    initial_sidebar_state="collapsed"  # Optional: hides sidebar if any exists (you don't have one)
)

# ─── Case Type Normalization ────────────────────────────────────────────────
CASE_TYPE_NORMALIZER = {
    "death notification": "Death Case",
    "death notice": "Death Case",
    "notification of death": "Death Case",
    "bereavement notification": "Death Case",
    "bereavement notice": "Death Case",
    "death case": "Death Case",
    "bereavement": "Death Case",
    "pension notification": "Death Case",
    "notification": "Death Case",
    "address change": "Address Change",
    "change of address": "Address Change",
    "address update": "Address Change",
    "lump sum": "Lump Sum",
    "lump sum payment": "Lump Sum",
    "beneficiary update": "Beneficiary Update",
    "beneficiary change": "Beneficiary Update",
    "nomination update": "Beneficiary Update",
    "retirement": "Retirement",
    "retirement application": "Retirement",
    "state pension": "Retirement",
    "pension application": "Retirement",
    "divorce": "Divorce",
    "pension sharing": "Divorce",
    "divorce proceedings": "Divorce",
    "pension sharing order": "Divorce",  # Added for your output example
    # Add more as needed
}

def normalize_date(date_str):
    if not date_str or date_str in ("null", None, "", "None"):
        return None
    date_str = str(date_str).strip().replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
    formats = [
        "%d %B %Y", "%d %b %Y", "%d/%m/%Y", "%d-%m-%Y",
        "%Y-%m-%d", "%B %d, %Y", "%d %B, %Y", "%d %b, %Y"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%d %b %Y")
        except ValueError:
            continue
    return date_str

def normalize_case_type(raw_type: str) -> str:
    if not raw_type:
        return raw_type
    cleaned = raw_type.strip().lower()
    return CASE_TYPE_NORMALIZER.get(cleaned, raw_type.strip())


def normalize_nino(nino_str):
    if not nino_str:
        return None
    cleaned = ''.join(c.upper() for c in str(nino_str) if c.isalnum())
    if cleaned.startswith('APT-'):
        return cleaned
    return cleaned


def get_pdf_pages(uploaded_file):
    if not uploaded_file:
        return []

    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    if uploaded_file.type.startswith("image/"):
        try:
            img = Image.open(io.BytesIO(file_bytes))
            return [img]
        except Exception as e:
            st.error(f"Cannot open image: {e}")
            return []

    elif uploaded_file.type == "application/pdf":
        try:
            # Open PDF from bytes using PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=180)  # Same DPI as before
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                pages.append(img)
            doc.close()
            if not pages:
                st.error("No pages found in PDF.")
                return []
            return pages

        except Exception as e:
            st.error(f"PDF processing failed: {str(e)}")
            return []

    else:
        st.error("Unsupported file type.")
        return []


def agent_one_extract(pages):
    if not pages:
        return None

    with st.spinner("Analyzing all pages with GPT-4o…"):
        try:
            image_urls = []
            for page in pages:
                buffered = io.BytesIO()
                page.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_urls.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

            extraction_prompt = """
You are an expert at reading official UK pension, HMRC, DWP letters and forms.

Analyze the entire multi-page document.

Return **valid JSON only** — no extra text, no markdown, no explanations.

Exact structure:

{
  "all_text": "full text with [Page X] markers",
  "persons": [
    {
      "name": "string or null",
      "role": "string or null",
      "dob": "string or null",
      "nino": "string or null",
      "addresses": [{"address": "string", "type": "current|new|old|previous|correspondence|unknown"}],
      "references": [string],
      "signed_date": "string or null"
    }
  ],
  "case_type": "string 2-4 words",
  "case_subtype": "string 2-4 words",
  "provided_documents": [string]   // only documents explicitly said to be enclosed/attached NOW
}

Rules:
- "persons" MUST be a list of objects (dictionaries) — never strings or null
- Main person (usually deceased/member) should come FIRST — prioritize person with NINO
- Use null for missing values
- "provided_documents": only items clearly stated as currently attached
"""

            messages = [
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": [{"type": "text", "text": "Analyze document."}, *image_urls]}
            ]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1800,
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.split("```json", 1)[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```", 2)[1].strip()

            return json.loads(content)

        except Exception as e:
            st.error(f"GPT-4o error: {str(e)}")
            return None


# ─── Required docs lookup ──────────────────────────────────────────────────

def get_required_docs(db_file, case_type):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT Documents_Required FROM Supporting_Documents WHERE Case_Type = ?", (case_type,))
        docs = [row[0] for row in cursor.fetchall()]
        conn.close()
        return docs
    except Exception as e:
        st.error(f"Error reading Supporting_Documents: {e}")
        return []


def documents_match(provided_docs, required_docs):
    if not required_docs:
        return True
    for expected in required_docs:
        exp_low = expected.strip().lower()
        for doc in provided_docs:
            doc_low = doc.strip().lower()
            if exp_low in doc_low or doc_low in exp_low:
                return True
    return False


# ─── Database functions ─────────────────────────────────────────────────────

def agent_two_check_update(db_file, nino, case_type, provided_docs):
    nino = normalize_nino(nino)
    if not nino or not case_type:
        return 'no_pending'

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        st.info(f"Looking for pending case: NINO={nino}, Type={case_type}")

        cursor.execute("""
            SELECT Case_ID, Expected_Document
            FROM cases
            WHERE NI_Number = ? AND Case_Type = ? AND Case_Status = 'Pending Info'
        """, (nino, case_type))

        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return 'no_pending'

        required = get_required_docs(db_file, case_type)
        matched = documents_match(provided_docs, required)

        if matched:
            for case_id, _ in rows:
                cursor.execute("""
                    UPDATE cases
                    SET Case_Status = 'Active',
                        Pending_Reason = NULL,
                        Expected_Document = NULL
                    WHERE Case_ID = ?
                """, (case_id,))
            conn.commit()
            st.success(f"Updated {len(rows)} pending case(s) to Active")
            conn.close()
            return 'updated'
        else:
            conn.close()
            return 'pending_no_update'

    except Exception as e:
        st.error(f"Update failed: {e}")
        return 'no_pending'


def agent_three_add_new(db_file, result, normalized_case_type):
    try:
        if not result.get("persons"):
            raise ValueError("No persons extracted")

        main_person = None

        # ─── Divorce special handling ──────────────────────────────────────
        if "divorce" in normalized_case_type.lower() or "sharing" in normalized_case_type.lower():
            # Prefer recipient/writer in divorce cases
            for p in result["persons"]:
                if isinstance(p, dict):
                    role = p.get("role", "").lower()
                    if any(kw in role for kw in ["recipient", "applicant", "writer", "ex-spouse", "petitioner"]):
                        main_person = p
                        break

        # ─── Prefer person with NINO (fallback or normal cases) ───────────
        if not main_person:
            for p in result["persons"]:
                if isinstance(p, dict) and p.get("nino"):
                    main_person = p
                    break

        # Ultimate fallback: first person
        if not main_person and result["persons"]:
            main_person = result["persons"][0]

        if not isinstance(main_person, dict):
            raise ValueError("No valid person dictionary found")

        st.info(f"Selected main person: {main_person.get('name', '—')} ({main_person.get('role', 'unknown')})")

        name = main_person.get("name")
        dob = normalize_date(main_person.get("dob"))
        nino = normalize_nino(main_person.get("nino"))
        pension_acc = ", ".join(main_person.get("references", []))

        # ─── Address logic – conditional on case type ──────────────────────
        addresses = main_person.get("addresses", [])
        old_address = new_address = correspondence_address = None

        is_address_change = "address change" in normalized_case_type.lower()

        if is_address_change:
            # Address Change: old = previous/old, new = new/current, correspondence = new
            for a in addresses:
                addr_type = a.get("type", "").lower()
                if addr_type in ["old", "previous"]:
                    old_address = a.get("address")
                if addr_type in ["new", "current"]:
                    new_address = a.get("address")
            correspondence_address = new_address if new_address else old_address
        else:
            # Non-Address Change: old = current/first address, new = None, correspondence = old
            for a in addresses:
                addr_type = a.get("type", "").lower()
                if addr_type in ["current", "correspondence", "old"]:
                    old_address = a.get("address")
                    break
            if not old_address and addresses:
                old_address = addresses[0].get("address")
            correspondence_address = old_address

        provided_docs = result.get("provided_documents", [])
        all_text_lower = result.get("all_text", "").lower()

        # ─── FUTURE SUBMISSION CHECK ───────────────────────────────────────
        future_keywords = [
            "to follow", "to follow shortly", "will follow", "following shortly",
            "will send", "will send later", "to be provided", "proof to follow",
            "documents will be", "later", "soon", "shortly", "enclosed will be",
            "following soon", "will provide", "to be sent", "awaiting"
        ]
        is_future_submission = any(kw in all_text_lower for kw in future_keywords)

        required_docs = get_required_docs(db_file, normalized_case_type)

        if is_future_submission:
            case_status = 'Pending Info'
            pending_reason = 'Documents mentioned as "to follow" or similar – awaiting submission'
            expected_document = ', '.join(required_docs) if required_docs else '(requirements not defined)'
            st.info("Future submission detected → forcing Pending Info")
        else:
            match_found = documents_match(provided_docs, required_docs)
            if match_found:
                case_status = 'Active'
                pending_reason = ''
                expected_document = ''
            else:
                case_status = 'Pending Info'
                pending_reason = 'Missing required documents'
                expected_document = ', '.join(required_docs) if required_docs else ''

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # ─── Prevent duplicates for same NINO + name + case_type ──────────
        cursor.execute("""
            SELECT Case_ID FROM cases 
            WHERE NI_Number = ? AND Member_Name = ? AND Case_Type = ?
        """, (nino, name, normalized_case_type))

        if cursor.fetchone():
            st.warning(f"Duplicate case already exists for {name} (NINO: {nino}, Type: {normalized_case_type}) → skipping creation")
            conn.close()
            return 'duplicate_skipped'

        year = datetime.now(pytz.timezone('Asia/Kolkata')).year
        prefix = f"APT-{year}-"

        cursor.execute("SELECT Case_ID FROM cases WHERE Case_ID LIKE ?", (f"{prefix}%",))
        existing = {int(r[0].rsplit("-", 1)[-1]) for r in cursor.fetchall() if r[0].rsplit("-", 1)[-1].isdigit()}

        num = 1
        while num in existing:
            num += 1
        case_id = f"{prefix}{num:03d}"

        cursor.execute("""
        INSERT INTO cases
        (Case_ID, Member_Name, NI_Number, Date_of_Birth, 
         Old_Address, New_Address, Correspondence_Address,
         Pension_Account_Number, Case_Type, Case_Subtype, 
         Case_Status, Pending_Reason, Expected_Document)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (case_id, name, nino, dob,
              old_address, new_address, correspondence_address,
              pension_acc, normalized_case_type, result.get("case_subtype", ""),
              case_status, pending_reason, expected_document))

        conn.commit()
        conn.close()
        return case_status

    except Exception as e:
        st.error(f"Failed to create new case: {e}")
        return None

# ─── Custom styling ─────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.8rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2.5rem;
    }
    .instructions {
        margin: 0 auto 2.5rem auto;
        font-size: 1.1rem;
        line-height: 1.6;
        padding: 0 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">AI-Powered Letter & Document Processor</div>', unsafe_allow_html=True)
#st.markdown('<div class="subtitle">Extract key information from UK pension, HMRC and DWP documents quickly and accurately</div>', unsafe_allow_html=True)

# ─── Instructions ──────────────────────────────────────────────────────────
st.markdown("""
<div class="instructions">
Upload a **PDF** (multi-page supported) or image (**JPG** / **PNG**) containing a letter or official document.

**What happens next:**
- The system analyzes **all pages** using AI
- Key information is automatically extracted (names, NINO, dates, addresses, case type, attached documents…)
- You will see a preview of the document + the extracted data
- Results can be downloaded as JSON

**Supported formats:** PDF, JPG, JPEG, PNG  
**Best results:** Clear scans/photos, good lighting, no heavy shadows
</div>
""", unsafe_allow_html=True)

# ─── Dashboard Data Functions ──────────────────────────────────────────────

@st.cache_data(ttl=30)
def get_dashboard_stats():
    try:
        conn = sqlite3.connect(DB_FILE)
        query = """
        SELECT 
            COUNT(DISTINCT Case_ID) AS total_cases,
            SUM(CASE WHEN Case_Status = 'Active' THEN 1 ELSE 0 END) AS active_cases,
            SUM(CASE WHEN Case_Status = 'Completed' THEN 1 ELSE 0 END) AS complete_cases,
            SUM(CASE WHEN Case_Status = 'Pending Info' THEN 1 ELSE 0 END) AS pending_cases
        FROM cases
        """
        stats = pd.read_sql_query(query, conn).iloc[0].to_dict()
        conn.close()
        return stats
    except Exception as e:
        st.error(f"Could not load dashboard stats: {str(e)}")
        return {"total_cases": 0, "active_cases": 0, "complete_cases": 0, "pending_cases": 0}


@st.cache_data(ttl=30)
def get_case_type_distribution():
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("""
            SELECT Case_Type, COUNT(DISTINCT Case_ID) as count
            FROM cases
            GROUP BY Case_Type
            ORDER BY count DESC
        """, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Case type query failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def get_top_ninos():
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("""
            SELECT 
                NI_Number,
                Member_Name,
                COUNT(DISTINCT Case_ID) as case_count
            FROM cases
            WHERE NI_Number IS NOT NULL AND NI_Number != ''
            GROUP BY NI_Number, Member_Name
            ORDER BY case_count DESC
            LIMIT 3
        """, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Top NINOs query failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_all_cases():
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM cases", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Failed to load cases: {e}")
        return pd.DataFrame()


def calculate_age(dob):
    if pd.isna(dob) or not dob:
        return None
    try:
        dob_date = parse(str(dob))
        today = datetime.today()
        return (today - dob_date).days // 365
    except:
        return None


# Load all cases once for the tabs that need data
all_cases_df = load_all_cases()

# Compute Age column once (used in Dashboard and Member Insights)
if not all_cases_df.empty and 'Date_of_Birth' in all_cases_df.columns:
    all_cases_df['Age'] = all_cases_df['Date_of_Birth'].apply(calculate_age)

# ─── Tabs Layout ───────────────────────────────────────────────────────────
tab_names = ["Upload & Process", "Dashboard Overview", "Case Explorer", "Member Insights"]

# Initialize session state for active tab (only once)
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = tab_names[0]

# Horizontal radio buttons acting as persistent tabs
active_tab = st.radio(
    label="Navigation",
    options=tab_names,
    horizontal=True,
    index=tab_names.index(st.session_state.active_tab),
    key="main_tab_radio",                     # ← stable key is very important
    label_visibility="collapsed"
)

# Update session state so it survives script reruns
st.session_state.active_tab = active_tab


# ── Tab 1: Upload & Process ────────────────────────────────────────────────
if active_tab == "Upload & Process":
    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload a document…",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="PDFs can have multiple pages — all pages will be analyzed."
    )

    if uploaded_file is not None:
        pages = get_pdf_pages(uploaded_file)

        if pages:
            result = agent_one_extract(pages)

            if result:
                # Display extraction results HERE in Tab 1
                st.subheader("📄 Extracted Information")

                persons = result.get("persons", [])
                if persons:
                    if not isinstance(persons, list):
                        st.error("'persons' is not a list – invalid format from AI")
                    else:
                        for idx, p in enumerate(persons, 1):
                            if isinstance(p, dict):
                                role = p.get('role', 'unknown')
                                st.markdown(f"**Person {idx} ({role})**:")
                                st.markdown(f" - **Name**: {p.get('name', '—')}")
                                st.markdown(f" - **NINO**: {p.get('nino', '—')}")
                                st.markdown(f" - **DOB**: {normalize_date(p.get('dob')) or '—'}")

                                addrs = p.get("addresses", [])
                                if addrs:
                                    st.markdown(" - **Addresses**:")
                                    for a in addrs:
                                        if isinstance(a, dict):
                                            typ = a.get("type", "unknown").title()
                                            st.markdown(f"   - **{typ}**: {a.get('address', '—')}")
                                        else:
                                            st.markdown(f"   - {a}")
                            else:
                                st.warning(f"Person {idx} is not a dictionary → skipping details")
                                st.markdown(f"**Person {idx} (invalid format)**: {p}")

                case_type_raw = result.get("case_type", "—")
                case_type = normalize_case_type(case_type_raw)

                st.markdown(f"**Raw Case Type**: {case_type_raw}")
                st.markdown(f"**Normalized Case Type**: {case_type}")
                st.markdown(f"**Case Subtype**: {result.get('case_subtype', '—')}")

                # Extract NINO logic
                nino_raw = None
                selected_person_name = "—"
                selected_person_role = "unknown"

                case_subject = result.get("case_subject", {})
                if isinstance(case_subject, dict) and case_subject.get("nino"):
                    nino_raw = case_subject.get("nino")
                    selected_person_name = case_subject.get("name", "—")
                    selected_person_role = case_subject.get("role", "unknown")
                    st.info(f"GPT selected case subject: {selected_person_name} ({selected_person_role})")

                if not nino_raw and persons and isinstance(persons, list):
                    for p in persons:
                        if isinstance(p, dict) and p.get("nino"):
                            nino_raw = p.get("nino")
                            selected_person_name = p.get("name", "—")
                            selected_person_role = p.get("role", "unknown")
                            break

                    if not nino_raw and persons:
                        first = persons[0]
                        if isinstance(first, dict):
                            nino_raw = first.get("nino")
                            selected_person_name = first.get("name", "—")
                            selected_person_role = first.get("role", "unknown")

                nino = normalize_nino(nino_raw)

                st.info(f"Final NINO selected: {nino or 'None'} from {selected_person_name} ({selected_person_role})")

                if not nino:
                    all_text = result.get("all_text", "").lower()
                    patterns = [
                        r'(?:national\s*insurance\s*(?:number|no|num)|nino|ni\s*(?:number|no))\s*[:\-]?\s*([\w\s\d\-/]+)',
                        r'apt-\d{4}-\d{3}',
                    ]
                    for pat in patterns:
                        match = re.search(pat, all_text, re.IGNORECASE | re.DOTALL)
                        if match:
                            raw = match.group(1).strip().upper().replace(" ", "").replace("-", "").replace("/", "")
                            if len(raw) >= 8 or raw.startswith('APT-'):
                                nino = raw
                                st.info(f"Fallback regex recovered NINO: {nino}")
                                break

                # Database operations
                if nino and case_type:
                    action = agent_two_check_update(DB_FILE, nino, case_type, result.get("provided_documents", []))
                    if action == 'updated':
                        st.success("✅ Pending case updated to **Active**")
                    elif action == 'pending_no_update':
                        st.warning("⚠️ Pending case found but documents missing → still **Pending**")
                    elif action == 'no_pending':
                        status = agent_three_add_new(DB_FILE, result, case_type)
                        if status == 'Active':
                            st.success("✅ **New Active case** created successfully")
                        elif status == 'Pending Info':
                            st.warning("⚠️ **New Pending Info case** created (missing documents or to follow)")
                        elif status == 'duplicate_skipped':
                            st.info("ℹ️ Duplicate case skipped — already exists in database")
                        else:
                            st.error("❌ Failed to create case")
                else:
                    st.warning("⚠️ Cannot update/create case: missing NINO or case type")

                st.success("🎉 Document processed successfully! View extraction details in the Dashboard Overview tab.")


# ── Tab 2: Dashboard Overview ──────────────────────────────────────────────
elif active_tab == "Dashboard Overview":
    st.subheader("Dashboard Overview")

    stats = get_dashboard_stats()

    # Calculate Total Members (unique NINOs)
    if not all_cases_df.empty and 'NI_Number' in all_cases_df.columns:
        total_members = all_cases_df['NI_Number'].dropna().nunique()
    else:
        total_members = 0

    # ── Top layer: 4 main metrics ──────────────────────────────────────────
    cols = st.columns(4)

    # Total Cases
    with cols[0]:
        st.markdown(
            f"""
            <div style="
                background: transparent;
                padding: 8px 8px;
                text-align: left;
                min-height: 140px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                color: inherit;
            ">
                <div style="
                    font-size: 17px;
                    color: inherit;
                    opacity: 0.8;
                    margin-bottom: 8px;
                    line-height: 1.3;
                    font-weight: 300;
                ">
                    Total Cases
                </div>
                <div style="
                    font-size: 32px;
                    font-weight: 500;
                    color: inherit;
                    line-height: 1.2;
                ">
                    {f"{stats['total_cases']:,}"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Active Cases
    with cols[1]:
        st.markdown(
            f"""
            <div style="
                background: transparent;
                padding: 8px 8px;
                text-align: left;
                min-height: 140px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                color: inherit;
            ">
                <div style="
                    font-size: 17px;
                    color: inherit;
                    opacity: 0.8;
                    margin-bottom: 8px;
                    line-height: 1.3;
                    font-weight: 300;
                ">
                    Active Cases
                </div>
                <div style="
                    font-size: 32px;
                    font-weight: 500;
                    color: inherit;
                    line-height: 1.2;
                ">
                    {f"{stats['active_cases']:,}"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Complete Cases
    with cols[2]:
        st.markdown(
            f"""
            <div style="
                background: transparent;
                padding: 8px 8px;
                text-align: left;
                min-height: 140px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                color: inherit;
            ">
                <div style="
                    font-size: 17px;
                    color: inherit;
                    opacity: 0.8;
                    margin-bottom: 8px;
                    line-height: 1.3;
                    font-weight: 300;
                ">
                    Complete Cases
                </div>
                <div style="
                    font-size: 32px;
                    font-weight: 500;
                    color: inherit;
                    line-height: 1.2;
                ">
                    {f"{stats['complete_cases']:,}"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Pending Cases
    with cols[3]:
        st.markdown(
            f"""
            <div style="
                background: transparent;
                padding: 8px 8px;
                text-align: left;
                min-height: 140px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                color: inherit;
            ">
                <div style="
                    font-size: 17px;
                    color: inherit;
                    opacity: 0.8;
                    margin-bottom: 8px;
                    line-height: 1.3;
                    font-weight: 300;
                ">
                    Pending Cases
                </div>
                <div style="
                    font-size: 32px;
                    font-weight: 500;
                    color: inherit;
                    line-height: 1.2;
                ">
                    {f"{stats['pending_cases']:,}"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── Bottom layer: 4 additional KPIs ────────────────────────────────────
    if not all_cases_df.empty and 'Case_Type' in all_cases_df.columns and 'Case_Status' in all_cases_df.columns:
        # Calculate metrics
        avg_age = all_cases_df['Age'].mean().round(1) if 'Age' in all_cases_df.columns else "N/A"
        pending_rate = (stats['pending_cases'] / stats['total_cases'] * 100) if stats['total_cases'] > 0 else 0

        # Pending reason — show all tied at max count in alphabetical order with line breaks
        pending_cases = all_cases_df[all_cases_df['Case_Status'] == 'Pending Info']
        if not pending_cases.empty and 'Pending_Reason' in pending_cases.columns:
            reason_counts = pending_cases['Pending_Reason'].value_counts()
            if not reason_counts.empty:
                max_count = reason_counts.max()
                top_reasons = reason_counts[reason_counts == max_count]
                # Sort alphabetically
                display_parts = [f"{reason} ({count})" for reason, count in sorted(top_reasons.items())]
                pending_reason_text = "<br>".join(display_parts)  # Use HTML line break
            else:
                pending_reason_text = "N/A"
        else:
            pending_reason_text = "N/A"

        cols2 = st.columns(4)

        # Total Members
        with cols2[0]:
            st.markdown(
                f"""
                <div style="
                    background: transparent;
                    padding: 8px 8px;
                    text-align: left;
                    min-height: 140px;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                    color: inherit;
                ">
                    <div style="
                        font-size: 17px;
                        color: inherit;
                        opacity: 0.7;
                        margin-bottom: 8px;
                        line-height: 1.2;
                        font-weight: 300;
                    ">
                        Total Members
                    </div>
                    <div style="
                        font-size: 27px;
                        font-weight: 500;
                        color: inherit;
                        line-height: 1.3;
                    ">
                        {f"{total_members:,}"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Pending Rate
        with cols2[1]:
            st.markdown(
                f"""
                <div style="
                    background: transparent;
                    padding: 8px 8px;
                    text-align: left;
                    min-height: 140px;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                    color: inherit;
                ">
                    <div style="
                        font-size: 17px;
                        color: inherit;
                        opacity: 0.7;
                        margin-bottom: 8px;
                        line-height: 1.2;
                        font-weight: 300;
                    ">
                        Pending Rate
                    </div>
                    <div style="
                        font-size: 27px;
                        font-weight: 500;
                        color: inherit;
                        line-height: 1.3;
                    ">
                        {f"{pending_rate:.1f}%"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Most Common Pending Reason
        with cols2[2]:
            st.markdown(
                f"""
                <div style="
                    background: transparent;
                    padding: 8px 8px;
                    text-align: left;
                    min-height: 140px;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                    color: inherit;
                ">
                    <div style="
                        font-size: 17px;
                        color: inherit;
                        opacity: 0.7;
                        margin-bottom: 8px;
                        line-height: 1.2;
                        font-weight: 300;
                    ">
                        Most Common Pending Reason
                    </div>
                    <div style="
                        font-size: 20px;
                        font-weight: 500;
                        color: inherit;
                        line-height: 1.3;
                        word-break: break-word;
                    ">
                        {pending_reason_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Avg Member Age
        with cols2[3]:
            st.markdown(
                f"""
                <div style="
                    background: transparent;
                    padding: 8px 8px;
                    text-align: left;
                    min-height: 140px;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                    color: inherit;
                ">
                    <div style="
                        font-size: 17px;
                        color: inherit;
                        opacity: 0.7;
                        margin-bottom: 8px;
                        line-height: 1.2;
                        font-weight: 300;
                    ">
                        Avg Member Age
                    </div>
                    <div style="
                        font-size: 27px;
                        font-weight: 500;
                        color: inherit;
                        line-height: 1.3;
                    ">
                        {f"{avg_age} years" if avg_age != "N/A" else "N/A"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.info("No data for additional KPIs yet.")

    # Charts with safeguards
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        df_types = get_case_type_distribution()
        if not df_types.empty:
            # Rename columns for better tooltip display
            df_types_display = df_types.rename(columns={
                'Case_Type': 'Case Type',
                'count': 'No. of Cases'
            })

            fig_bar = px.bar(
                df_types_display,
                x="Case Type",
                y="No. of Cases",
                title="Number of Cases by Case Type",
                text_auto=True,
                color="No. of Cases",
                color_continuous_scale="Blues"
            )

            fig_bar.update_layout(
                xaxis_title="Case Type",
                yaxis_title="Number of Cases",
                xaxis_tickangle=-45
            )

            # Customize hover template
            fig_bar.update_traces(
                hovertemplate='<b>Case Type:</b> %{x}<br><b>No. of Cases:</b> %{y}<extra></extra>'
            )

            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No case type data.")

    with col_chart2:
        df_nino = get_top_ninos()
        if not df_nino.empty:
            # Rename columns for better tooltip display
            df_nino_display = df_nino.rename(columns={
                'NI_Number': 'NINO',
                'Member_Name': 'Member',
                'case_count': 'No of Cases'
            })

            fig_pie = px.pie(
                df_nino_display,
                values="No of Cases",
                names="NINO",
                title="Top 3 NINOs by Most Cases",
                hover_data=['Member'],  # Only include Member in hover_data
                hole=0.4
            )

            # Customize hover template to show in correct order: NINO, Member, No of Cases
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>NINO:</b> %{label}<br><b>Member:</b> %{customdata[0]}<br><b>No of Cases:</b> %{value}<extra></extra>'
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        else:
            st.info("No NINO data.")

    # More Insights with safeguards
    col_v1, col_v2, col_v3 = st.columns(3)

    with col_v1:
        if not all_cases_df.empty and 'Case_Status' in all_cases_df.columns:
            df_status = all_cases_df['Case_Status'].value_counts().reset_index(name='count')

            df_status_display = df_status.rename(columns={
                'Case_Status': 'Case Status',
                'count': 'No. of Cases'
            })

            fig_status = px.pie(
                df_status_display,
                values='No. of Cases',
                names='Case Status',
                title="Status Breakdown"
            )

            fig_status.update_traces(
                hovertemplate='<b>Case Status:</b> %{label}<br><b>No. of Cases:</b> %{value}<extra></extra>'
            )

            st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.info("No status data.")

    with col_v2:
        if not all_cases_df.empty and 'Case_ID' in all_cases_df.columns:
            df_temp = all_cases_df.copy()

            df_temp['Year'] = df_temp['Case_ID'].str.extract(r'APT-(\d{4})-\d{3}')[0]

            df_yearly = df_temp.groupby('Year')['Case_ID'].nunique().reset_index()
            df_yearly.columns = ['Year', 'No. of Cases']

            df_yearly = df_yearly.sort_values('Year')

            if not df_yearly.empty:
                fig_bar = px.bar(
                    df_yearly,
                    x='Year',
                    y='No. of Cases',
                    title="Cases By Year",
                    text_auto=True
                )

                fig_bar.update_layout(
                    xaxis_title="Year",
                    yaxis_title="No. of Cases",
                    xaxis_type='category'
                )

                fig_bar.update_traces(
                    textposition = 'outside',
                    hovertemplate='<b>Year:</b> %{x}<br><b>No. of Cases:</b> %{y}<extra></extra>'
                )

                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No timeline data.")
        else:
            st.info("No Case_ID data.")

    with col_v3:
        if not all_cases_df.empty and 'Case_Status' in all_cases_df.columns and 'Expected_Document' in all_cases_df.columns:
            df_pending_docs = all_cases_df[all_cases_df['Case_Status'] == 'Pending Info'][
                'Expected_Document'].value_counts().reset_index(name='count')

            if not df_pending_docs.empty:
                df_pending_docs_display = df_pending_docs.rename(columns={
                    'Expected_Document': 'Expected Document',
                    'count': 'No. of Cases'
                })

                fig_docs = px.bar(
                    df_pending_docs_display,
                    x='Expected Document',
                    y='No. of Cases',
                    title="Top Missing Documents",
                    text_auto=True
                )

                fig_docs.update_layout(
                    xaxis_title="Expected Document",
                    yaxis_title="No. of Cases"
                )

                fig_docs.update_traces(
                    textposition='outside',
                    hovertemplate='<b>Expected Document:</b> %{x}<br><b>No. of Cases:</b> %{y}<extra></extra>'
                )

                st.plotly_chart(fig_docs, use_container_width=True)
            else:
                st.info("No pending documents.")
        else:
            st.info("No pending document data.")

    st.caption(f"Data last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ── Tab 3: Case Explorer ───────────────────────────────────────────────────
elif active_tab == "Case Explorer":
    st.subheader("Case Explorer - All Cases")

    if not all_cases_df.empty:
        if 'Case_Status' in all_cases_df.columns:
            st.dataframe(
                all_cases_df.style.apply(
                    lambda row: ['background: lightcoral' if row['Case_Status'] == 'Pending Info' else 'background: lightgreen' if row['Case_Status'] == 'Completed' else '' for _ in row],
                    axis=1
                ),
                use_container_width=True
            )
        else:
            st.dataframe(all_cases_df, use_container_width=True)

        st.subheader("Bulk Actions")
        csv = all_cases_df.to_csv(index=False).encode('utf-8')
        st.download_button("Export All Cases (CSV)", csv, "all_cases.csv", "text/csv")

        st.subheader("High Priority Pending")

        if 'Case_Status' in all_cases_df.columns and 'Case_Type' in all_cases_df.columns:
            pending_death = all_cases_df[
                (all_cases_df['Case_Status'] == 'Pending Info') &
                (all_cases_df['Case_Type'].str.contains("Death", case=False, na=False))
            ]

            if not pending_death.empty:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #0068c9;
                        color: white;
                        padding: 16px;
                        border-radius: 8px;
                        margin-bottom: 16px;
                        font-weight: 500;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                    ">
                        <strong>{len(pending_death)} High-Priority Death Cases Pending</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.dataframe(
                    pending_death.style.set_properties(**{
                        'background-color': 'rgba(255,105,100,0.05)',
                        'color': 'white'
                    })
                )
            else:
                st.info("No high-priority pending Death cases.")
        else:
            st.info("No case status/type data.")
    else:
        st.info("No cases have been saved yet. Upload documents to populate the database.")


# ── Tab 4: Member Insights ─────────────────────────────────────────────────
elif active_tab == "Member Insights":
    st.subheader("Member Insights - All Cases")

    if not all_cases_df.empty and 'NI_Number' in all_cases_df.columns:
        all_ninos = sorted(all_cases_df['NI_Number'].dropna().unique())

        # Removed the separate text_input filter
        # The multiselect itself now handles search/filter when you type in it

        selected_ninos = st.multiselect(
            "Select NI Number(s)  –  start typing to search/filter",
            options=all_ninos,
            default=[],
            placeholder="Type to search and select one or more NI Numbers...",
            key="member_ninos_multiselect_key"   # stable key prevents reset
        )

        if selected_ninos:
            member_df = all_cases_df[all_cases_df['NI_Number'].isin(selected_ninos)]

            if not member_df.empty:
                unique_members = member_df['NI_Number'].nunique()
                total_cases = len(member_df)
                st.write(f"**Selected Members**: {unique_members}")
                st.write(f"**Total Cases**: {total_cases}")

                # Show all columns from the dataframe
                styled_df = member_df.style.set_properties(**{
                    'text-align': 'left'
                }).set_table_styles([
                    {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'left')]}
                ])

                st.dataframe(styled_df, use_container_width=True)

                if unique_members > 1:
                    st.caption("Showing cases for all selected NI Numbers.")
            else:
                st.info("No cases found for the selected NI Number(s).")
        else:
            st.info("Select one or more NI Numbers above (start typing in the box to filter the list).")
    else:
        st.info("No NI_Number column found in the database.")