# Demo Script and Viva Preparation (Major Project)

## 7-minute demo
1. **Intro (30 s):** "It reads a resume PDF, extracts skills and structure with NLP, recommends careers, matches job descriptions and uses a GenAI API for guidance."
2. **Structure (30 s):** `src/` modules, `data/`, `.env.example`, `.gitignore`, `private/`.
3. **Validation (30 s):** `python main.py analyze --resume /tmp/notes.txt` → friendly error (try a renamed text file too).
4. **Analyze (60 s):** `python main.py analyze --resume data/sample_resumes/aarav_mehta.pdf`.
5. **Recommend (60 s):** `python main.py recommend --resume data/sample_resumes/priya_nair.pdf`; explain the 3 score parts.
6. **Match (90 s):** `python main.py match --resume data/sample_resumes/aarav_mehta.pdf --jd-file data/sample_job_descriptions/data_scientist_jd.txt`; point out "Tableau or Power BI".
7. **GenAI (90 s):** with your key in `.env`: `python main.py improve --resume ... --task project`, then `python main.py chat --resume ...`. Show the privacy consent question. Then remove the key to show the offline fallback.
8. **Evaluation + tests (30 s):** `python main.py evaluate`, `python -m unittest discover tests`.

## Likely questions
**What is TF-IDF?** A weight that is high for words frequent in a document but rare across documents. **Cosine similarity?** The cosine of the angle between two vectors: 1 = same direction, 0 = nothing in common.
**What is content-based filtering?** Recommending by comparing item content (role descriptions) with the user's profile (resume), without using other users' data.
**Why a skill dictionary instead of an ML model?** It is transparent, needs no labelled data and is easy to extend; the drawback is that unknown skills are missed.
**How do you avoid matching "go" or "r" in ordinary words?** The skills are named "golang", "r programming", "c programming"; ordinary words never match.
**How are "X or Y" requirements handled?** They form one group, satisfied by either skill.
**How is personal data protected?** Redaction before any AI call, user consent, no storage of resume text, key in `.env`, `private/` git-ignored.
**What stops the AI from inventing achievements?** The system and task prompts forbid it and require placeholders such as [add measurable result]; output must still be reviewed by the user.
**What is prompt injection?** Text inside the resume trying to give the AI instructions. The resume is wrapped in tags and declared as data; this reduces but does not eliminate the risk.
**What happens when the API fails?** Retries with exponential backoff for rate limits/network/5xx, no retry for a wrong key, then an offline fallback.
**Why did skill-only beat the combined score in evaluation?** The synthetic resumes were generated from the role skill lists, which favours that component; weights were fixed in advance and deliberately not tuned on this data.
**What bug did testing find?** Phone numbers like "+91 90000 02824" were not being redacted; fixed with a better detector and tests.
**Is the AI part tested?** With mocked providers only; capture and paste real outputs into `docs/sample_prompts_and_outputs.md` before submission.

## Upload to GitHub
```bash
git init && git add . && git status     # confirm .env is NOT listed
git commit -m "Career guidance and resume intelligence system"
git branch -M main
git remote add origin https://github.com/<your-username>/career-guidance-resume-intelligence.git
git push -u origin main
```
## Before submitting
- [ ] Replace `<your-username>` and the title-page placeholders in the report.
- [ ] Add your API key to `.env`, run the AI commands, paste real responses into `docs/sample_prompts_and_outputs.md` and the report.
- [ ] Test with your own resume (keep it in `private/`) and take your own terminal screenshots.
- [ ] Install PyMuPDF (`pip install -r requirements.txt`) and re-run the tests on your machine.
