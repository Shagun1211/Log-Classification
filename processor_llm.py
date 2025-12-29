from dotenv import load_dotenv
from groq import Groq
import re
from functools import lru_cache

load_dotenv()
groq = Groq()

@lru_cache(maxsize=256)
def classify_with_llm(log_msg):
    prompt = f"""
Classify the log message into ONE category ONLY:
- Workflow Error
- Deprecation Warning
- Unclassified

Rules:
- Return ONLY the category
- Wrap it in <category></category>
- No explanation

Log message:
{log_msg}
""".strip()

    try:
        chat_completion = groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-70b-versatile",  
            temperature=0.0,
            max_tokens=50
        )
    except Exception as e:
        return "Unclassified"

    content = chat_completion.choices[0].message.content.strip()

    match = re.search(
        r"<category>\s*(.*?)\s*</category>",
        content,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not match:
        return "Unclassified"

    category = match.group(1).strip()

    if category not in {"Workflow Error", "Deprecation Warning", "Unclassified"}:
        return "Unclassified"

    return category
