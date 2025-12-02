# A Specialized Mathematics Tutor Agent for Students of Grades 7 & 8  
(Using Google ADK + Gemini Models)

---

## 1. Overview

This project implements an intelligent mathematics tutoring agent for **CBSE Grades 7 and 8**, built using the **Google ADK** and **Gemini LLMs**.  
It provides scaffolded, Socratic-style tutoring and can optionally interpret student-uploaded solution images.

⚠️ **Important:**  
This project **must be run locally only**.  
**Public deployment is strictly prohibited**, because it requires users to load their own copyrighted NCERT textbook files.  
The author does **not** include or distribute *any* NCERT textbook content in this repository.

---

## 2. Local-Only Usage Requirement (Read Before Running)

Because NCERT textbooks are copyrighted, this project is designed for:

- **Local testing only**  
- **Educational / evaluation use only**  
- **Non-commercial use**  
- **With NCERT textbook PDFs supplied manually by the evaluator/user**

This repository does **not** contain:
- NCERT PDFs  
- Extracted pages  
- Images from textbooks  
- Scanned or OCR text from NCERT content  

All textbook-related functionality works **only when the user supplies their own legally obtained files locally**.

---

## 3. How to Download NCERT Textbook PDFs (Required for Textbook-Based Features)

Before using the textbook-lookup features, evaluators must manually download NCERT PDFs from the official NCERT website:

### Official Source (NCERT)
- Main Textbook Page: https://ncert.nic.in/textbook.php  
- Grade 7 Mathematics: https://ncert.nic.in/textbook.php?gegp1=0-8  
- Grade 8 Mathematics: https://ncert.nic.in/textbook.php?hegp1=0-7  

### Important Notes
- For Grade 7, only **Part 1 (Ganita Prakash)** is required.  
- Do **not** use Part 2 (Ganita Prakash-II).  
- After downloading, store the files in the following folders:

`data/bookchapters/grade7/`
`data/bookchapters/grade8/`

### Rename the files exactly as follows:

| NCERT File | Rename to |
|------------|-----------|
| Prelims | `preamble.pdf` |
| Chapter 1 | `ch_1.pdf` |
| Chapter 2 | `ch_2.pdf` |
| Chapter 3 | `ch_3.pdf` |
| ... | continue similarly |

This naming scheme is required for the `load_context` tool.

💡 **Evaluator convenience note**  
You do **not** need to download full Grade 7 *and* Grade 8 textbooks.  
**Just one chapter from either grade is sufficient to evaluate the project.**

For example, for Grade 8 you may download only:

Grade 8 → Mathematics → Chapter 7 → save as ch_7.pdf

This alone is enough to test the full tutor-agent pipeline.

💡 **Canonical test case (optional)**  
Evaluators may use the following scenario for a quick functional test:

- **Grade:** 8  
- **Chapter:** 7  
- **Chapter Page Number:** 12  
- **Book Page Number:** 170  
- **Exercise Number:** 2  

Providing page/exercise numbers does **not** violate NCERT copyright because no copyrighted content is reproduced.

### After evaluation, all NCERT files should be deleted from the evaluator’s machine.

---

## 4. Legal Notice (Copyright & Use Restrictions)

The following copyright notice is posted on the official NCERT website:

> **NCERT TEXTBOOK COPYRIGHT NOTICE**  
> NCERT textbooks are copyrighted. While they may be downloaded and used as textbooks or for reference, **republication, redistribution, hosting on websites, packaging into software, or electronic circulation by any individual or agency is strictly prohibited.**  
> No website or online service is permitted to host these online textbooks.  
> Links may be provided only with written permission from NCERT.

### Compliance in This Project

- ✔ This repository **does NOT include** any NCERT textbooks.  
- ✔ This project does **NOT** republish or redistribute NCERT content.  
- ✔ Users must download textbooks themselves from NCERT’s official site.  
- ✔ This software must **NOT** be deployed publicly in any form.  
- ✔ Intended **only for local evaluation and academic use**.

By running this project, users agree to comply with NCERT’s copyright policy.

---

## 5. API Key Setup

To run this project, you must set a Google API key.

1. Create a `.env` file in the project root.  
2. Add the following line:
GOOGLE_API_KEY=your_api_key_here

---

## 6. System Requirements / System Tested On

This project is lightweight and can run on most modern systems without specialized hardware.  
Below are the system specifications used during development, followed by general requirements for reproducibility.

### 6.1 System Tested On (Developer Machine)
- **OS:** Ubuntu 24.04 LTS  
- **Python Environment Manager:** Miniconda3  
- **Python Version:** 3.11  
- **CPU:** AMD Ryzen™ AI 7 PRO 360 w/ Radeon™ 880M × 16  
- **RAM:** 64 GB  
- **Disk Space Used:** ~300 MB (including all textbook PDFs)  
- **GPU:** Not required  
- **Internet:** Required for Gemini API calls  

### 6.2 General Requirements (For Anyone Running This Project)
- **OS:** Linux, macOS, or Windows  
- **Python Version:** 3.11 recommended  
- **CPU:** Any modern x86-64 processor  
- **RAM:** 4 GB minimum (8 GB recommended)  
- **Disk:** ~300 MB (project files + local textbook PDFs)  
- **GPU:** Not required  
- **Internet:** Required for Gemini API calls  

---

## 7. .gitignore Notice (Important)

To avoid unintentional copyright violations, ensure your `.gitignore` contains:

`data/bookchapters/`

This prevents accidental uploading of textbook PDFs.

---

## 8. Evaluation Instructions (For Reviewers & Judges)

1. Clone the repository.  
2. Create a Conda environment and install dependencies.  
3. Download **at least one** NCERT chapter PDF from the official NCERT website.  
   You do **not** need the full textbooks.  
4. Rename and place the file(s) following the instructions in Section 3 above.
5. (Optional) For fastest evaluation, download only **Grade 8 → Chapter 7**.
6. Run the Streamlit interface from the repository root:

   ```bash
   streamlit run streamlit_app.py
7. After evaluation, delete all NCERT textbook files from your machine.  

---

## 9. Public Deployment Prohibited

This project **must NOT be deployed** to:

- Streamlit Cloud  
- HuggingFace Spaces  
- GitHub Pages  
- Any public or private server  

Reason:
- It requires access to NCERT textbook pages, which cannot be redistributed or served publicly.  
- Local-only usage is required for copyright compliance.

---

## 10. Contact

For evaluation or academic inquiries, please contact the project author.  
No public deployment, redistribution, or production use is permitted.

---
