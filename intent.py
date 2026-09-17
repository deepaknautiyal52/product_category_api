import re

# -------------------------
# Intent Detection
# -------------------------

GREETINGS = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how r u",
    "how are you doing",
    "what's up",
    "whats up",
    "sup"
]

THANKS = [
    "thanks",
    "thank you",
    "thanks a lot",
    "thank you so much",
    "appreciate it"
]

GOODBYE = [
    "bye",
    "goodbye",
    "see you",
    "take care"
]

HELP = [
    "help",
    "what can you do",
    "how can you help",
    "options"
]


def detect_intent(prompt):

    prompt = prompt.lower().strip()

    prompt = re.sub(r"\s+", " ", prompt)

    if prompt in GREETINGS:
        return "greeting"

    if prompt in THANKS:
        return "thanks"

    if prompt in GOODBYE:
        return "goodbye"

    if prompt in HELP:
        return "help"

    return "prediction"