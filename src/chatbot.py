"""Interactive terminal career chatbot that uses the (redacted) resume as context."""
from src import config
from src.career_assistant import SYSTEM_RULES, _context, offline_summary
from src.llm_client import LLMError, ask, llm_status, provider
from src.privacy import ask_consent
from src.recommender import format_recommendations, recommend

HELP = """Commands:  /roles  career recommendations | /skills  skills you have | /clear  reset conversation | /help | /exit
Or just type a question, e.g. "What should I learn next?" or "How do I prepare for a data science interview?" """


def _offline_answer(question: str, resume: dict) -> str:
    q = question.lower()
    if any(k in q for k in ("role", "career", "job", "suit")):
        return format_recommendations(recommend(resume, top_k=3))
    if any(k in q for k in ("skill", "learn", "missing", "next")):
        top = recommend(resume, top_k=1)[0]
        return f"For {top['role']} you could learn next: {', '.join(top['missing_required'][:5]) or 'advanced topics'}."
    if "summary" in q:
        return offline_summary(resume)
    return "Offline mode can answer questions about roles, skills and summaries. Add an API key in .env for full AI answers."


def run_chatbot(resume: dict, assume_yes: bool = False, input_fn=input) -> None:
    ok, message = llm_status()
    use_ai = ok
    print(f"\n{'AI mode: ' + message if ok else 'Offline mode (' + message + ')'}")
    if use_ai and not ask_consent(provider(), assume_yes):
        print("Consent not given: continuing in offline mode.")
        use_ai = False
    system = SYSTEM_RULES + "\nYou are chatting with the candidate. " + "Answer using the resume context below.\n" + _context(resume)
    print(HELP)
    history = []
    while True:
        try:
            question = input_fn("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if not question:
            continue
        low = question.lower()
        if low in ("/exit", "/quit", "exit", "quit"):
            print("Goodbye!")
            return
        if low == "/help":
            print(HELP); continue
        if low == "/clear":
            history.clear(); print("Conversation cleared."); continue
        if low == "/roles":
            print(format_recommendations(recommend(resume, top_k=3))); continue
        if low == "/skills":
            print("Skills detected: " + ", ".join(resume["skills"])); continue
        if len(question) > config.MAX_QUESTION_CHARS:
            print(f"Please keep questions under {config.MAX_QUESTION_CHARS} characters."); continue
        if not use_ai:
            print(f"\nAssistant: {_offline_answer(question, resume)}"); continue
        history.append({"role": "user", "content": question})
        try:
            answer = ask(system, history[-12:], max_tokens=600)
        except LLMError as exc:
            history.pop()
            print(f"[AI error: {exc}]")
            continue
        history.append({"role": "assistant", "content": answer})
        print(f"\nAssistant: {answer}")
