"""AI-Powered Career Guidance and Resume Intelligence System - command-line interface.

Examples (sample files are included in data/):
    python main.py build-model
    python main.py analyze   --resume data/sample_resumes/aarav_mehta.pdf
    python main.py recommend --resume data/sample_resumes/aarav_mehta.pdf --top 5
    python main.py match     --resume data/sample_resumes/aarav_mehta.pdf --jd-file data/sample_job_descriptions/data_scientist_jd.txt
    python main.py match     --resume my_resume.pdf --jd-stdin
    python main.py improve   --resume data/sample_resumes/aarav_mehta.pdf --task review
    python main.py chat      --resume data/sample_resumes/aarav_mehta.pdf
    python main.py evaluate
"""
import argparse
import json
import sys

from src import config
from src.career_assistant import improve_resume, learning_roadmap, professional_summary, recommend_skills, rewrite_project
from src.chatbot import run_chatbot
from src.evaluation import run_evaluation
from src.jd_matcher import match_resume_to_jd, read_jd_file, read_jd_terminal, skill_gap_report
from src.llm_client import LLMError, ask, llm_status, provider
from src.pdf_parser import ResumeFileError, extract_text
from src.privacy import ask_consent
from src.recommender import build_model, format_recommendations, load_model, recommend
from src.resume_parser import parse_resume
from src.skill_extractor import group_by_category


def load_resume(path_arg):
    """Validate + read + parse the PDF. Asks for the path in the terminal if --resume was not given."""
    path = path_arg or input("Enter the path to your resume PDF: ").strip()
    try:
        return parse_resume(extract_text(path))
    except ResumeFileError as exc:
        sys.exit(f"Error: {exc}")


def print_analysis(r: dict) -> None:
    edu = r["education"]
    print("\n" + "=" * 60 + "\n  RESUME ANALYSIS\n" + "=" * 60)
    print(f"Candidate     : {r['name'] or 'not detected'}   (shown only on your screen)")
    print(f"Sections found: {', '.join(r['sections_found'])}")
    print(f"Education     : {edu['highest_level']}" + (f" | fields: {', '.join(edu['fields'])}" if edu["fields"] else ""))
    print(f"Experience    : about {r['experience_years']} year(s) across {len(r['experience'])} role(s)")
    print(f"\nSkills identified ({len(r['skills'])}):")
    for cat, skills in group_by_category(r["skills"]).items():
        print(f"  {cat:<30} {', '.join(skills)}")
    print(f"\nCertifications ({len(r['certifications'])}):" + ("" if r["certifications"] else " none detected"))
    for c in r["certifications"]:
        print(f"  - {c}")
    print(f"\nProjects ({len(r['projects'])}):")
    for p in r["projects"]:
        print(f"  - {p['title']}  [{', '.join(p['skills']) or 'no skills detected'}]")
    print(f"\nExperience entries:")
    for j in r["experience"]:
        print(f"  - {j['title_and_company']}")
    if r["interests"]:
        print(f"\nProfessional interests: {', '.join(r['interests'])}")


def resolve_role(resume, role_name):
    """Return the recommendation record for the requested role (or the best-matching role if none given)."""
    ranked = recommend(resume, top_k=0)
    if role_name:
        for r in ranked:
            if role_name.lower() in r["role"].lower():
                return r
        sys.exit(f"Unknown role '{role_name}'. Available roles: " + ", ".join(r["role"] for r in ranked))
    return ranked[0]


def ai_allowed(args) -> bool:
    """AI is used only if enabled, a key is configured, and the user consents."""
    if getattr(args, "no_ai", False):
        return False
    ok, message = llm_status()
    if not ok:
        print(f"[{message}] Running in offline mode.\n")
        return False
    print(message)
    if not ask_consent(provider(), getattr(args, "yes", False)):
        print("Consent not given: running in offline mode.\n")
        return False
    return True


def cmd_improve(args) -> None:
    resume = load_resume(args.resume)
    target = resolve_role(resume, args.role)
    use_ai = ai_allowed(args)
    task = args.task
    if task == "review":
        text, src = improve_resume(resume, target["role"], use_ai)
    elif task == "summary":
        text, src = professional_summary(resume, target["role"], use_ai)
    elif task == "project":
        text, src = rewrite_project(resume, args.project_index, use_ai)
    elif task == "skills":
        text, src = recommend_skills(resume, target["role"], target["missing_required"], target["missing_preferred"], use_ai)
    else:
        text, src = learning_roadmap(resume, target["role"], target["missing_required"] + target["missing_preferred"], args.weeks, use_ai)
    print(f"\n--- {task.upper()} for target role: {target['role']} ({'AI-generated' if src == 'ai' else 'offline'}) ---\n{text}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-Powered Career Guidance and Resume Intelligence System")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-model", help="build and save the TF-IDF career recommendation model")
    sub.add_parser("evaluate", help="evaluate the recommender and skill extractor")
    sub.add_parser("check-ai", help="test your AI provider, key and model settings")
    for name, h in [("analyze", "extract structured information from a resume PDF"), ("recommend", "recommend suitable career paths"),
                    ("match", "compare a resume with a job description"), ("improve", "AI resume improvement and guidance"),
                    ("chat", "interactive career guidance chatbot")]:
        p = sub.add_parser(name, help=h)
        p.add_argument("--resume", help="path to the resume PDF (asked interactively if omitted)")
        if name == "analyze":
            p.add_argument("--json", help="save the structured data (without raw text/contact info) to this JSON file")
        if name == "recommend":
            p.add_argument("--top", type=int, default=5)
        if name == "match":
            g = p.add_mutually_exclusive_group()
            g.add_argument("--jd-file", help="job description .txt file")
            g.add_argument("--jd-text", help="job description as text")
            g.add_argument("--jd-stdin", action="store_true", help="type/paste the job description in the terminal")
            p.add_argument("--report", help="save the skill-gap report (Markdown) to this path")
        if name == "improve":
            p.add_argument("--task", choices=["review", "summary", "project", "skills", "roadmap"], default="review")
            p.add_argument("--role", help="target role (default: best recommended role)")
            p.add_argument("--project-index", type=int, default=0, help="which project to rewrite (0 = first)")
            p.add_argument("--weeks", type=int, default=8)
        if name in ("improve", "chat"):
            p.add_argument("--yes", action="store_true", help="skip the privacy consent question")
        if name == "improve":
            p.add_argument("--no-ai", action="store_true", help="use only offline rule-based output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build-model":
        build_model()
    elif args.command == "evaluate":
        run_evaluation()
    elif args.command == "check-ai":
        ok, message = llm_status()
        print(message)
        if ok:
            try:
                print("AI is working. Reply:", ask("Answer in one short sentence.", [{"role": "user", "content": "Say OK and name the model family you are."}], 60))
            except LLMError as exc:
                sys.exit(f"AI check failed: {exc}")
    elif args.command == "analyze":
        resume = load_resume(args.resume)
        print_analysis(resume)
        if args.json:
            safe = {k: v for k, v in resume.items() if k not in ("raw_text", "body_text", "name", "has_email", "has_phone")}
            with open(args.json, "w") as f:
                json.dump(safe, f, indent=2)
            print(f"\nStructured data saved to {args.json}")
    elif args.command == "recommend":
        print(format_recommendations(recommend(load_resume(args.resume), load_model(), top_k=args.top)))
    elif args.command == "match":
        resume = load_resume(args.resume)
        try:
            jd = read_jd_file(args.jd_file) if args.jd_file else args.jd_text if args.jd_text else read_jd_terminal() if args.jd_stdin else None
            if jd is None:
                sys.exit("Provide the job description with --jd-file, --jd-text or --jd-stdin.")
            report = skill_gap_report(match_resume_to_jd(resume, jd))
        except (FileNotFoundError, ValueError) as exc:
            sys.exit(f"Error: {exc}")
        print("\n" + report)
        if args.report:
            config.OUT_DIR.mkdir(exist_ok=True)
            with open(args.report, "w") as f:
                f.write(report)
            print(f"\nReport saved to {args.report}")
    elif args.command == "improve":
        cmd_improve(args)
    elif args.command == "chat":
        run_chatbot(load_resume(args.resume), assume_yes=args.yes)


if __name__ == "__main__":
    main()
