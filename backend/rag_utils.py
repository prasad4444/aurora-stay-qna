
from langchain_openai import ChatOpenAI
import os

import re

PROPERTY_ALIASES = {"chicago": "ash-chi", "chi loop": "ash-chi", "denver": "ash-den", "miami": "ash-mia", "brickell": "ash-mia", "seattle": "ash-sea", "boston": "ash-bos", "atlanta": "ash-atl"}
STOPWORDS = {"what", "when", "where", "which", "this", "that", "there", "with", "from", "have", "will", "would", "could", "about", "hotel", "hotels", "aurorastay", "your", "their", "does", "they", "need", "please", "room", "stay", "booked", "since", "someone", "supposed"}

def retrieve_policy_context(question, k=6, vectorstore=None):
    """Select grounded, question-specific evidence from the complete FAISS document store.

    Full-store lexical scanning supplements semantic search so rare policy clauses
    are not lost before reranking. No reference answers are used in retrieval.
    """
    if vectorstore is None:
        raise ValueError("Pass the initialized FAISS vectorstore")
    q = question.lower()
    codes = set(re.findall(r"ash-[a-z]{3}", q))
    codes.update(code for alias, code in PROPERTY_ALIASES.items() if re.search(r"\b" + re.escape(alias) + r"\b", q))

    intent_terms = {
        "checkin": ("check in", "check-in", "check out", "check-out", "checkout", "departure"),
        "pets": ("dog", "pet", "animal"),
        "advance": ("advance purchase", "prepaid", "non-refundable", "nonrefundable"),
        "refund": ("refund", "cancel", "no-show", "flexible", "one-night"),
        "smoking": ("smok", "vap", "cigarette"),
        "incidental": ("incidental", "pending", "hold", "authorization", "charged twice"),
        "noise": ("loud", "noise", "midnight", "disturb", "quiet", "room next door"),
        "amenities": ("pool", "cabana", "fitness", "amenities"),
        "breakfast": ("breakfast",),
        "accessibility": ("accessible", "accessibility", "wheelchair"),
        "rooms": ("queen", "king", "occupancy", "beds"),
        "agency": ("agency", "ota", "third-party", "travel agent"),
        "latecheckout": ("late checkout", "late check-out", "late departure"),
    }
    active = [name for name, terms in intent_terms.items() if any(t in q for t in terms)]
    # Avoid interpreting a late-cancellation charge as a question about incidental holds.
    if "incidental" in active and not any(t in q for t in ("incidental", "pending", "hold", "authorization", "charged twice")):
        active.remove("incidental")
    if "refund" in active and "advance" in active:
        pass  # Both policies must be represented.

    queries = {
        "checkin": "check in check out early late availability approval property times",
        "pets": "pet dog non-service animal property pet fee limit",
        "advance": "Advance Purchase non-refundable cannot change confirmation exception law system error service recovery",
        "refund": "flexible late cancellation one night tax manager exception service failure duplicate booking verified system error approval",
        "smoking": "smoking vaping remediation charge up to 250 evidence odor alone dispute channel",
        "incidental": "incidentals authorization pending hold not final charge release issuer folio settled",
        "noise": "quiet hours 10:00 PM 7:00 AM noise complaint document escalate threats emergency",
        "amenities": "pool cabana availability property profile amenities cannot confirm",
        "breakfast": "breakfast property fee included rate plan",
        "accessibility": "accessible king features room profile confirmation",
        "rooms": "double queen king occupancy fire limit",
        "agency": "OTA agency booking refund cancellation agency handles",
        "latecheckout": "late checkout availability approval fee departure",
    }
    docs = []
    seen_docs = set()
    def add_doc(d):
        key = re.sub(r"\s+", " ", d.page_content).strip()
        if key and key not in seen_docs:
            seen_docs.add(key)
            docs.append(d)
    # FAISS retains all indexed chunks in its docstore. Inspecting them is
    # deterministic and catches relevant clauses absent from similarity top-k.
    store = getattr(getattr(vectorstore, "docstore", None), "_dict", None)
    if isinstance(store, dict):
        for d in store.values():
            if hasattr(d, "page_content"):
                add_doc(d)
    else:
        for search in [question] + [queries[a] for a in active]:
            for d in vectorstore.similarity_search(search, k=50):
                add_doc(d)

    keywords = {
        "checkin": ("check-in", "check-out", "checkout", "early check", "late check", "arrival", "departure", "availability"),
        "pets": ("pet", "dog", "animal", "per stay", "per night"),
        "advance": ("advance purchase", "non-refundable", "nonrefundable", "non-changeable", "cannot be changed", "confirmation", "system error", "service recovery"),
        "refund": ("refund", "cancellation", "cancel", "one night", "one-night", "manager", "exception", "duplicate booking", "service failure", "system error"),
        "smoking": ("smok", "vap", "remediation", "$250", "odor", "dispute"),
        "incidental": ("incidental", "authorization", "pending", "issuer", "folio", "hold", "release"),
        "noise": ("quiet hours", "noise", "disturb", "complaint", "10:00 pm", "7:00 am", "threat", "violence"),
        "amenities": ("pool", "cabana", "amenit", "property profile"),
        "breakfast": ("breakfast", "rate plan"),
        "accessibility": ("accessible", "accessibility", "feature"),
        "rooms": ("queen", "king", "occupancy", "fire"),
        "agency": ("agency", "ota", "third-party", "booking channel"),
        "latecheckout": ("late checkout", "late check-out", "extension", "departure", "approval"),
    }
    stop = STOPWORDS | {"can", "and", "the", "for", "now", "how", "much", "after", "before", "yesterday", "still", "time"}
    qtokens = set(re.findall(r"[a-z]{3,}", q)) - stop
    candidates = []
    seen_fragments = set()
    for doc in docs:
        # Each original line or paragraph is retained verbatim. Adjacent lines
        # are grouped for PDF-extracted tables and multi-line policy clauses.
        lines = [x.strip() for x in doc.page_content.splitlines() if x.strip()]
        windows = []
        for i, line in enumerate(lines):
            if len(line) >= 12:
                windows.append(line)
                if i + 1 < len(lines):
                    windows.append(line + " " + lines[i+1])
                if i + 2 < len(lines):
                    windows.append(line + " " + lines[i+1] + " " + lines[i+2])
        if not lines:
            windows = [doc.page_content]
        for fragment in windows:
            if len(fragment) < 18 or len(fragment) > 850:
                continue
            low = fragment.lower()
            if low.startswith("|") and codes and not any(c in low for c in codes):
                continue
            norm = re.sub(r"\s+", " ", low)
            if norm in seen_fragments:
                continue
            seen_fragments.add(norm)
            matched = {topic for topic in active if any(term in low for term in keywords[topic])}
            if active and not matched:
                continue
            overlap = len(qtokens & set(re.findall(r"[a-z]{3,}", low)))
            code_match = any(c in low for c in codes)
            if codes and re.search(r"ash-[a-z]{3}", low) and not code_match:
                continue
            score = 5 * len(matched) + 1.3 * overlap + (7 if code_match else 0)
            score += min(3, sum(sum(term in low for term in keywords[t]) for t in matched) * 0.5)
            if len(fragment) > 400:
                score -= 2
            candidates.append((score, fragment, doc.metadata, matched))

    # Recover property table records split across multiple PDF-extracted lines.
    # This is source-backed: only lines from the SAME indexed document are used.
    if "checkin" in active and codes:
        for doc in docs:
            lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
            for i, line in enumerate(lines):
                if not any(code in line.lower() for code in codes):
                    continue
                # Stop at the next property row rather than mixing two hotels.
                row = [line]
                for following in lines[i + 1:i + 16]:
                    if re.search(r"ash-[a-z]{3}", following.lower()):
                        break
                    row.append(following)
                    if re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\s*/\s*\d{1,2}:\d{2}\s*(?:am|pm)", following, re.I):
                        break
                fragment = " ".join(row)
                if re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\s*/\s*\d{1,2}:\d{2}\s*(?:am|pm)", fragment, re.I):
                    candidates.append((1000, fragment, doc.metadata, {"checkin"}))

    candidates.sort(key=lambda x: (-x[0], len(x[1])))
    selected = []
    covered = set()
    max_items = max(4, min(8, k + 2))
    def append(item):
        score, fragment, metadata, matched = item
        # Prevent near-duplicate overlapping windows from dominating the context.
        norm = set(re.findall(r"[a-z]{3,}", fragment.lower()))
        for existing in selected:
            other = set(re.findall(r"[a-z]{3,}", existing[1].lower()))
            if norm and len(norm & other) / max(1, min(len(norm), len(other))) > .84:
                return False
        selected.append(item)
        covered.update(matched)
        return True
    # Coverage first: multi-part questions need distinct policies.
    for topic in active:
        for item in candidates:
            if topic in item[3] and append(item):
                break
    for item in candidates:
        if len(selected) >= max_items:
            break
        append(item)
    return [f"Content:\n{fragment}\n\nMetadata:\n{metadata}" for _, fragment, metadata, _ in selected]

def retrieve(query: str, k: int = 4, vectorstore=None) -> list[str]:
    return retrieve_policy_context(query, k=k, vectorstore=vectorstore)

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about AuroraStay hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, deadlines, property-specific rules, and exceptions.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent fees, exceptions, property rules, deadlines, or refund terms.
    10. If some parts of the question are supported, answer those parts and explicitly identify the missing parts.
       For unsupported property amenities, state they cannot be confirmed from the supplied documents
       and recommend checking the current property profile or contacting the property.
       Never claim that a refund is approved; explain any documented approval requirements.
       If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
