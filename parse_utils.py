import re
import spacy

nlp = spacy.load("en_core_web_trf")

def merge_person_entities(doc):
    """
    Combine consecutive PERSON entities to form a full name.
    E.g., if spaCy sees "Sebastian", "Belulok", "Saging" as separate
    PERSON tokens, we join them into one name string.
    """
    names = []
    current_name_parts = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            current_name_parts.append(ent.text)
        else:
            if current_name_parts:
                names.append(" ".join(current_name_parts))
                current_name_parts = []
    # flush leftover
    if current_name_parts:
        names.append(" ".join(current_name_parts))

    if not names:
        return None
    # Return the longest string (the name with the most tokens/characters)
    return max(names, key=len)

def skip_org_keyword(org_text):
    """
    Returns True if the org_text contains 'noise' keywords or
    if it's just a known brand or short reference we want to skip.
    You can customize this list as needed.
    """
    skip_keywords = [
        "foundation", "list", "university", "college", "hubspot",
        "aws", "wordpress", "42kl", "sigma school", "mongodb",
        "adnexioedu", "dean’s list"  # add more as needed
    ]
    lower_org = org_text.lower()
    # If it contains any skip keyword, or is too short (e.g. "AWS")
    if any(kw in lower_org for kw in skip_keywords):
        return True
    # If org is extremely short, skip
    if len(org_text) < 5:
        return True
    return False

def extract_skills_from_text(text):
    """
    Scan the text for known skills/keywords.
    This simple version checks for exact substring matches in a predefined list.
    Feel free to customize or expand for your use case.
    """
    SKILL_LIST = [
        "python", "java", "machine learning", "nlp",
        "docker", "aws", "react", "c++", "sql",
        "deep learning", "pytorch", "tensorflow"
    ]
    found = []
    lower_text = text.lower()
    for skill in SKILL_LIST:
        if skill in lower_text:
            found.append(skill)
    return list(set(found))  # remove duplicates

def parse_resume_spacy(text):
    """
    Parse a resume's text using the en_core_web_trf (transformer) model.

    We'll attempt to extract:
    - Email (via regex)
    - Phone (via regex)
    - Person name (via merging consecutive PERSON entities + fallback)
    - Organization mentions (ORG entity) [with skip logic for known 'noise']
    - Education lines (keywords like 'university', 'bachelor', etc.)
    - (Optional) Skills via 'extract_skills_from_text'

    Returns a dict containing these fields.
    """
    doc = nlp(text)

    # 1. Email regex
    email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails = re.findall(email_regex, text)

    # 2. Phone regex
    phone_regex = r'(\+?\d[\d\-\(\) ]{7,}\d)'
    phones = re.findall(phone_regex, text)

    # 3. Merge PERSON entities
    person_name = merge_person_entities(doc)
    if not person_name:
        # Fallback: maybe use the first line if it looks name-ish
        lines = text.split("\n")
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 60:
                person_name = first_line

    # 4. ORG detection with skip logic
    orgs_all = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    orgs_filtered = [org for org in orgs_all if not skip_org_keyword(org)]

    # 5. Education lines
    edu_keywords = ["bachelor", "master", "university", "college", "b.sc", "m.sc", "diploma"]
    education_lines = []
    for line in text.split("\n"):
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in edu_keywords):
            education_lines.append(line.strip())

    # 6. Skills (optional)
    # Uncomment if you want the parse to also return a 'skills' list
    skills = extract_skills_from_text(text)

    parsed_data = {
        "name": person_name,
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "organizations": orgs_filtered,
        "education": education_lines,
        "skills": skills,  # optional
    }

    return parsed_data