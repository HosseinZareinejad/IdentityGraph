"""
Phase 1 - synthetic multi-platform world generator.

Replaces the old single-table data/datasetCreator.py. Instead of one flat
table of duplicate/noisy records, this builds:

  1. A "real identity" registry (data/real_identity_registry.parquet) -
     stands in for a secondary/official database with ground-truth identity
     attributes (name, birth year, city, education, job). Includes a small
     fraction of "twins": distinct people who coincidentally share name +
     city + birth year, to give the resolution pipeline real hard negatives.

  2. Four platform dumps (data/platform_dumps/{platform}.parquet) - for each
     real identity, 1-4 platform accounts are generated with PER-PLATFORM
     field coverage, tone, and noise (mirroring the proposal's own claim that
     data availability differs sharply by network).

  3. Four per-platform social graphs (data/social_graphs/{platform}_edges.parquet)
     - built with community structure correlated across platforms (same
     underlying friend groups), NOT literal cross-platform edges. This is a
     deliberate design choice: node2vec/graph embeddings trained separately
     per platform live in different vector spaces and are not directly
     comparable, so the graph model (Phase 3) should NOT cosine-compare raw
     embeddings across platforms. It should either (a) use platform-agnostic
     structural features (degree, clustering coefficient, community id via
     Louvain) as comparable inputs to the metadata/fusion model, or (b) use
     the graph for label-propagation refinement once a few seed matches are
     found by other modalities. The community-correlated structure generated
     here supports both.

  4. Evaluation scaffolding: ground_truth_pairs.parquet (true cross-platform
     links) and hard_negative_pairs.parquet (twin pairs) - so Phase 6 doesn't
     need to reconstruct these from entity_id joins by hand.

Note on scope: this generator only creates ONE account per platform per real
identity (no same-platform duplicates). Within-platform deduplication is
already covered by the existing ingestion.py / resolution.py pipeline; this
script's job is specifically the cross-platform linkage problem.
"""
import json
import os
import random
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx
import pandas as pd
from faker import Faker

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.text_utils import transliterate, get_nickname_variants, NICKNAME_MAP  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
NUM_IDENTITIES = 5000
TWIN_FRACTION = 0.04          # fraction of identities that get a "twin"
PLATFORMS = ["twitter", "instagram", "telegram", "linkedin"]
NUM_PLATFORMS_WEIGHTS = {1: 0.10, 2: 0.30, 3: 0.35, 4: 0.25}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLATFORM_DUMP_DIR = os.path.join(BASE_DIR, "platform_dumps")
GRAPH_DIR = os.path.join(BASE_DIR, "social_graphs")

random.seed(SEED)
fake_fa = Faker("fa_IR")
fake_fa.seed_instance(SEED)

EDUCATION_FIELDS = [
    "مهندسی کامپیوتر", "مهندسی برق", "مدیریت بازرگانی", "حقوق", "پزشکی",
    "حسابداری", "روانشناسی", "مهندسی عمران", "علوم کامپیوتر", "اقتصاد",
]
EDUCATION_DEGREES = ["کارشناسی", "کارشناسی ارشد", "دکتری"]
# Interests are the ONE semantic thread that persists across a person's
# accounts on different platforms (expressed formally on LinkedIn, casually on
# Twitter, etc.). Without this, each platform's text is independent noise and
# no text model can ever link accounts semantically - which is exactly what
# the first version of this generator got wrong.
#
# Ordered roughly popular -> niche, and sampled with a Zipf-like weighting so
# common interests genuinely collide across people (many "فوتبال" fans). That
# keeps the signal realistic and noisy rather than an accidental secret key
# that trivially solves the matching problem.
INTERESTS = [
    "فوتبال", "موسیقی", "فیلم", "سفر", "کتاب", "آشپزی", "ورزش", "طبیعت",
    "عکاسی", "کوهنوردی", "برنامه‌نویسی", "بازی‌های ویدیویی", "تاریخ", "هنر",
    "شعر", "خوشنویسی", "نقاشی", "دوچرخه‌سواری", "شنا", "والیبال", "بسکتبال",
    "یوگا", "مدیتیشن", "باغبانی", "حیوانات خانگی", "فناوری", "هوش مصنوعی",
    "اقتصاد", "بورس", "کارآفرینی", "معماری", "طراحی گرافیک", "مد و پوشاک",
    "سینمای ایران", "تئاتر", "موسیقی سنتی", "گیتار", "پیانو", "خوانندگی",
    "سفرهای جاده‌ای", "کمپینگ", "ماهیگیری", "اسکی", "تنیس", "شطرنج",
    "فلسفه", "روانشناسی", "زبان انگلیسی", "ترجمه", "نویسندگی",
    "پادکست", "مستند", "نجوم", "محیط زیست", "کار داوطلبانه",
]
# Exponent 0.5 (not steeper): still skewed toward popular topics, but not so
# concentrated that half the population shares the same top-3 interests, which
# would make interest overlap useless as a discriminator.
INTEREST_WEIGHTS = [1.0 / ((i + 1) ** 0.5) for i in range(len(INTERESTS))]

TITLES = ["دکتر", "مهندس", "آقای", "خانم", "سید"]
EMOJIS = ["\U0001F642", "❤️", "\U0001F525", "\U0001F602", "\U0001F44D",
          "✨", "\U0001F339", "\U0001F60D", "\U0001F97A", "\U0001F4AF", "\U0001F605", "\U0001F90D"]
FINGLISH_WORDS = {
    "سلام": "salam", "خوب": "khoob", "عالی": "ali", "ممنون": "mamnoon",
    "دوست": "dooset", "امروز": "emrooz", "خیلی": "kheili", "عاشق": "asheghe",
    "زندگی": "zendegi", "خوش": "khosh",
}
PHONETIC_PAIRS = [("ز", "ذ"), ("س", "ص"), ("ت", "ط"), ("ق", "غ")]


# ---------------------------------------------------------------------------
# Registry generation (the "real identities")
# ---------------------------------------------------------------------------
def sample_interests(k: int) -> list[str]:
    """k distinct interests, weighted so popular topics genuinely collide."""
    chosen = []
    while len(chosen) < k:
        pick = random.choices(INTERESTS, weights=INTEREST_WEIGHTS, k=1)[0]
        if pick not in chosen:
            chosen.append(pick)
    return chosen


def make_identity(entity_id=None, twin_of=None, name_hint=None, city_hint=None, birth_year_hint=None):
    first_name = fake_fa.first_name()
    last_name = fake_fa.last_name()
    if name_hint:
        first_name, last_name = name_hint
    return {
        "entity_id": entity_id or str(uuid.uuid4()),
        "first_name": first_name,
        "last_name": last_name,
        "full_name": f"{first_name} {last_name}",
        "birth_year": birth_year_hint if birth_year_hint is not None else random.randint(1340, 1400),
        "city": city_hint if city_hint is not None else fake_fa.city(),
        "job": fake_fa.job(),
        "education": f"{random.choice(EDUCATION_DEGREES)} {random.choice(EDUCATION_FIELDS)}",
        "phone_number": fake_fa.phone_number(),
        "email": f"{transliterate(first_name)}.{transliterate(last_name)}{random.randint(1,99)}@gmail.com",
        "twin_of": twin_of,
        # Persistent interests - the semantic thread linking this person's
        # accounts across platforms (each account mentions only a subset)
        "interests": sample_interests(random.randint(2, 3)),
        # Stylometric fingerprint - reused with drift across this person's platform accounts
        "emoji_rate": round(random.uniform(0.0, 0.3), 3),
        "finglish_rate": round(random.uniform(0.0, 0.2), 3),
        "punctuation_style": random.choice(["none", "ellipsis", "exclaim", "mixed"]),
        # Temporal fingerprint - preferred activity hour (0-24) and spread
        "preferred_hour": round(random.uniform(0, 24), 2),
        "hour_std": round(random.uniform(1.5, 4.0), 2),
    }


def build_registry():
    registry = [make_identity() for _ in range(NUM_IDENTITIES)]

    # Hard negatives: "twins" - distinct people sharing name + city + birth_year
    num_twins = int(NUM_IDENTITIES * TWIN_FRACTION)
    anchors = random.sample(registry, num_twins)
    twins = []
    for anchor in anchors:
        twin = make_identity(
            twin_of=anchor["entity_id"],
            name_hint=(anchor["first_name"], anchor["last_name"]),
            city_hint=anchor["city"],
            birth_year_hint=anchor["birth_year"],
        )
        twins.append(twin)
    registry.extend(twins)

    # Community assignment (shared across platforms - drives correlated graph structure)
    all_ids = [r["entity_id"] for r in registry]
    random.shuffle(all_ids)
    community_of = {}
    i, cid = 0, 0
    while i < len(all_ids):
        size = random.randint(8, 25)
        for eid in all_ids[i:i + size]:
            community_of[eid] = cid
        i += size
        cid += 1
    for r in registry:
        r["community_id"] = community_of[r["entity_id"]]

    return registry, num_twins


# ---------------------------------------------------------------------------
# Text generation (stylometric + tonal, per platform)
# ---------------------------------------------------------------------------
def apply_finglish(text, rate):
    words = text.split()
    out = []
    for w in words:
        bare = w.strip(".,!؟،")
        if bare in FINGLISH_WORDS and random.random() < rate:
            out.append(FINGLISH_WORDS[bare])
        else:
            out.append(w)
    return " ".join(out)


def apply_emoji(text, rate):
    if random.random() < rate:
        n = random.randint(1, 2)
        return f"{text} {''.join(random.choice(EMOJIS) for _ in range(n))}"
    return text


def apply_punctuation(text, style):
    if style == "ellipsis":
        return text + " ..."
    if style == "exclaim":
        return text + " !!"
    if style == "mixed":
        return text + random.choice([" ...", " !", " ?", ""])
    return text


def stylize(text, identity, casual=True):
    text = apply_finglish(text, identity["finglish_rate"])
    if casual:
        text = apply_emoji(text, identity["emoji_rate"])
        text = apply_punctuation(text, identity["punctuation_style"])
    return text


# Bio templates keyed by platform, then by how many of the person's interests
# this particular profile happens to mention (0, 1 or 2). More templates than
# the first version - two LinkedIn templates for 5000 people made every
# LinkedIn bio semantically identical, which drowned out the cross-platform
# signal entirely.
BIO_TEMPLATES = {
    "twitter": {
        0: ["فقط یه آدم معمولی", "دنبال آرامش می‌گردم", "{job} | زندگی روزمره",
            "اینجا حرفای خودمو می‌زنم", "روزمرگی"],
        1: ["عاشق {i1}م", "{i1} + قهوه = زندگی خوب", "اینجا بیشتر از {i1} می‌نویسم",
            "{job} | دوستدار {i1}", "{i1} دوست", "زندگی یعنی {i1}"],
        2: ["{i1} و {i2} | زندگی روزمره", "بین {i1} و {i2} در نوسانم",
            "{i1}، {i2} و کمی روزمرگی"],
    },
    "instagram": {
        0: ["لحظه‌های خوب زندگیم", "\U0001F4CD {city}", "{job}", "همینجا"],
        1: ["{i1} \U0001F31F", "\U0001F4CD {city} | {i1}", "{i1} و لحظه‌های خوب",
            "دنیای {i1}"],
        2: ["{i1} • {i2}", "\U0001F4CD {city} | {i1} و {i2}", "{i1} \U0001F90D {i2}"],
    },
    "telegram": {
        0: ["", "", "", "فقط برای دوستان نزدیک", "بیو ندارم"],
        1: ["{i1}", "علاقه‌مند به {i1}", ""],
        2: ["{i1} و {i2}", "{i1} | {i2}", ""],
    },
    "linkedin": {
        0: ["{job} با تجربه در حوزه فعالیت خود. فارغ‌التحصیل {education} از یکی از دانشگاه‌های کشور.",
            "متخصص {job}. {education}. به‌دنبال فرصت‌های همکاری حرفه‌ای هستم.",
            "{education} | {job}"],
        1: ["{job} با تجربه در حوزه فعالیت خود. فارغ‌التحصیل {education}. علاقه‌مند به {i1} و یادگیری مستمر.",
            "متخصص {job}. {education}. فعال در زمینه {i1}.",
            "{education} | {job} | علاقه‌مند به حوزه {i1}",
            "به عنوان {job} فعالیت می‌کنم. در کنار کار حرفه‌ای، به {i1} علاقه دارم.",
            "کارشناس {job}. تمرکز اصلی من روی {i1} است."],
        2: ["کارشناس {job} با پس‌زمینه تحصیلی {education}. حوزه‌های مورد علاقه: {i1}، {i2}.",
            "متخصص {job}. {education}. فعال در زمینه {i1} و {i2}.",
            "{job} | علاقه‌مند به {i1} و {i2} | {education}"],
    },
}


def pick_bio_interests(identity):
    """A profile mentions only a SUBSET of the person's real interests, so the
    cross-platform overlap is partial rather than a giveaway.

    Biased toward interests[0] (the person's "dominant" interest), because in
    reality one main interest tends to recur across someone's profiles while
    secondary ones appear sporadically. Purely uniform subset sampling made
    overlap between two of the same person's accounts too rare to be a signal
    at all.
    """
    ints = identity["interests"]
    if not ints:
        return []
    k = min(random.choice([0, 1, 1, 1, 1, 2, 2]), len(ints))
    if k == 0:
        return []
    if k == 1:
        # dominant interest most of the time, a secondary one occasionally
        return [ints[0]] if random.random() < 0.7 else [random.choice(ints)]
    return [ints[0]] + random.sample(ints[1:], k - 1)


def make_bio(platform, identity):
    picked = pick_bio_interests(identity)
    template = random.choice(BIO_TEMPLATES[platform][len(picked)])
    text = template.format(
        job=identity["job"],
        education=identity["education"],
        city=identity["city"],
        i1=picked[0] if len(picked) >= 1 else "",
        i2=picked[1] if len(picked) >= 2 else "",
    )
    return stylize(text, identity, casual=platform in ("twitter", "instagram")) if text else ""


# Roughly half of a person's posts touch one of their persistent interests;
# the rest is filler. Fully random filler (the first version) meant the
# "posts" vector carried no person-specific semantic signal at all.
INTEREST_POST_TEMPLATES = [
    "امروز درباره {i} خوندم و خیلی برام جالب بود.",
    "{i} همیشه حال آدمو خوب می‌کنه.",
    "یه تجربه جدید در زمینه {i} داشتم.",
    "هیچی مثل {i} آرومم نمی‌کنه.",
    "بحث {i} همیشه برام جذابه.",
    "این روزا بیشتر وقتمو می‌ذارم روی {i}.",
    "کاش وقت بیشتری برای {i} داشتم.",
]
LINKEDIN_POST_TEMPLATES = [
    "تجربه اخیر من در حوزه {i} نکات آموزنده‌ای داشت.",
    "به نظر می‌رسد {i} در سال‌های آینده اهمیت بیشتری پیدا کند.",
    "خوشحالم که در پروژه‌ای مرتبط با {i} مشارکت داشتم.",
    "یادداشتی کوتاه درباره اهمیت {i} در کار حرفه‌ای.",
]

POST_INTEREST_RATE = 0.5


def make_post_text(platform, identity):
    casual = platform in ("twitter", "instagram")
    if identity["interests"] and random.random() < POST_INTEREST_RATE:
        pool = LINKEDIN_POST_TEMPLATES if platform == "linkedin" else INTEREST_POST_TEMPLATES
        # same dominant-interest bias as bios
        topic = identity["interests"][0] if random.random() < 0.6 else random.choice(identity["interests"])
        base = random.choice(pool).format(i=topic)
    else:
        base = fake_fa.sentence()
    return stylize(base, identity, casual=casual)


def sample_post_timestamp(identity, platform):
    hour = random.gauss(identity["preferred_hour"], identity["hour_std"])
    if platform == "linkedin":  # office-hours bias regardless of personal rhythm
        hour = 0.6 * hour + 0.4 * random.uniform(9, 18)
    hour = hour % 24
    base_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 600))
    return base_date.replace(hour=int(hour), minute=random.randint(0, 59)).isoformat()


NUM_POSTS_RANGE = {"twitter": (3, 8), "instagram": (3, 6), "telegram": (0, 2), "linkedin": (0, 2)}


def make_posts(platform, identity):
    lo, hi = NUM_POSTS_RANGE[platform]
    n = random.randint(lo, hi)
    return [
        {"text": make_post_text(platform, identity), "timestamp": sample_post_timestamp(identity, platform)}
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Name noise + username generation
# ---------------------------------------------------------------------------
def apply_name_noise(full_name, first_name, last_name):
    if random.random() >= 0.55:
        return full_name, False
    noise_type = random.choices(
        ["replace_y", "replace_k", "phonetic", "delete_char", "swap_order", "add_title"],
        weights=[15, 15, 15, 20, 20, 15], k=1,
    )[0]
    name = full_name
    if noise_type == "replace_y" and "ی" in name:
        name = name.replace("ی", "ي", 1)
    elif noise_type == "replace_k" and "ک" in name:
        name = name.replace("ک", "ك", 1)
    elif noise_type == "phonetic":
        for a, b in PHONETIC_PAIRS:
            if a in name:
                name = name.replace(a, b, 1)
                break
    elif noise_type == "delete_char" and len(name) > 4:
        idx = random.randint(1, len(name) - 2)
        if name[idx] != " ":
            name = name[:idx] + name[idx + 1:]
    elif noise_type == "swap_order":
        name = f"{last_name} {first_name}"
    elif noise_type == "add_title":
        name = f"{random.choice(TITLES)} {name}"
    return name, (name != full_name)


def make_username(first_name, last_name):
    nicknames = get_nickname_variants(first_name)
    use_nick = bool(nicknames) and random.random() < 0.4
    base_first = transliterate(random.choice(nicknames) if use_nick else first_name)
    base_last = transliterate(last_name)
    pattern = random.choice(["dot", "underscore", "concat", "reverse_dot", "first_only"])
    if pattern == "dot":
        uname = f"{base_first}.{base_last}"
    elif pattern == "underscore":
        uname = f"{base_first}_{base_last}"
    elif pattern == "concat":
        uname = f"{base_first}{base_last}"
    elif pattern == "reverse_dot":
        uname = f"{base_last}.{base_first}"
    else:
        uname = base_first
    if random.random() < 0.45:
        uname += str(random.randint(1, 999))
    return uname


# ---------------------------------------------------------------------------
# Per-platform field coverage (mirrors the proposal's "missing data" challenge)
# ---------------------------------------------------------------------------
CITY_COVERAGE = {"twitter": 0.20, "instagram": 0.30, "telegram": 0.0, "linkedin": 1.0}
BIRTH_YEAR_COVERAGE = {"twitter": 0.05, "instagram": 0.05, "telegram": 0.0, "linkedin": 0.25}
JOB_EDU_COVERAGE = {"twitter": 0.0, "instagram": 0.0, "telegram": 0.0, "linkedin": 1.0}
# A small amount of contact info DOES leak onto public profiles in reality
# (business accounts showing a phone, "business inquiries: x@y.com" in a bio).
# Without any cross-platform overlap here, the deterministic exact-match path
# is unreachable by construction and Phase 3's exact-match branch would never
# be exercised by a single test case. Kept deliberately rare (~1% of pairs).
PHONE_COVERAGE = {"twitter": 0.0, "instagram": 0.05, "telegram": 0.25, "linkedin": 0.0}
EMAIL_COVERAGE = {"twitter": 0.08, "instagram": 0.08, "telegram": 0.05, "linkedin": 0.20}


def noisy_phone(phone):
    if phone.startswith("+98"):
        return "0" + phone[3:]
    if phone.startswith("0") and random.random() < 0.5:
        return "+98" + phone[1:]
    return phone


def make_platform_account(identity, platform):
    record_id = str(uuid.uuid4())
    noisy_name, is_noisy = apply_name_noise(identity["full_name"], identity["first_name"], identity["last_name"])
    username = make_username(identity["first_name"], identity["last_name"])

    record = {
        "record_id": record_id,
        "entity_id": identity["entity_id"],
        "platform": platform,
        "display_name": noisy_name,
        "username": username,
        "bio": make_bio(platform, identity),
        "posts": make_posts(platform, identity),
        "city": identity["city"] if random.random() < CITY_COVERAGE[platform] else None,
        "birth_year": identity["birth_year"] if random.random() < BIRTH_YEAR_COVERAGE[platform] else None,
        "job_title": identity["job"] if random.random() < JOB_EDU_COVERAGE[platform] else None,
        "education": identity["education"] if random.random() < JOB_EDU_COVERAGE[platform] else None,
        "phone_number": noisy_phone(identity["phone_number"]) if random.random() < PHONE_COVERAGE[platform] else None,
        "email": identity["email"] if random.random() < EMAIL_COVERAGE[platform] else None,
        "is_noisy": is_noisy,
    }
    return record


def generate_platform_accounts(registry):
    accounts_by_platform = defaultdict(list)
    for identity in registry:
        num_platforms = random.choices(
            list(NUM_PLATFORMS_WEIGHTS.keys()), weights=list(NUM_PLATFORMS_WEIGHTS.values()), k=1
        )[0]
        chosen = random.sample(PLATFORMS, num_platforms)
        for platform in chosen:
            accounts_by_platform[platform].append(make_platform_account(identity, platform))
    return accounts_by_platform


# ---------------------------------------------------------------------------
# Per-platform, community-correlated social graph
# ---------------------------------------------------------------------------
def build_platform_graph(platform_records, community_of_entity):
    G = nx.Graph()
    record_ids = [r["record_id"] for r in platform_records]
    G.add_nodes_from(record_ids)

    comm_to_records = defaultdict(list)
    for r in platform_records:
        comm_to_records[community_of_entity[r["entity_id"]]].append(r["record_id"])

    for r in platform_records:
        degree = min(60, max(1, int(random.lognormvariate(1.8, 0.9))))
        same_comm_pool = [rid for rid in comm_to_records[community_of_entity[r["entity_id"]]] if rid != r["record_id"]]
        num_same = int(degree * 0.8)
        targets = set()
        if same_comm_pool:
            targets.update(random.sample(same_comm_pool, min(num_same, len(same_comm_pool))))
        remaining = degree - len(targets)
        if remaining > 0 and len(record_ids) > 1:
            pool = [rid for rid in record_ids if rid != r["record_id"] and rid not in targets]
            if pool:
                targets.update(random.sample(pool, min(remaining, len(pool))))
        for t in targets:
            G.add_edge(r["record_id"], t)
    return G


# ---------------------------------------------------------------------------
# Evaluation scaffolding
# ---------------------------------------------------------------------------
def build_ground_truth_pairs(accounts_by_platform):
    accounts_by_entity = defaultdict(list)
    for platform, records in accounts_by_platform.items():
        for r in records:
            accounts_by_entity[r["entity_id"]].append((r["record_id"], platform))

    pairs = []
    for entity_id, accts in accounts_by_entity.items():
        for i in range(len(accts)):
            for j in range(i + 1, len(accts)):
                (rid_a, plat_a), (rid_b, plat_b) = accts[i], accts[j]
                if plat_a == plat_b:
                    continue  # cross-platform only; within-platform dedup is a different problem
                pairs.append({
                    "entity_id": entity_id,
                    "record_id_a": rid_a, "platform_a": plat_a,
                    "record_id_b": rid_b, "platform_b": plat_b,
                })
    return pd.DataFrame(pairs)


def build_hard_negative_pairs(registry, accounts_by_platform):
    accounts_by_entity = defaultdict(list)
    for platform, records in accounts_by_platform.items():
        for r in records:
            accounts_by_entity[r["entity_id"]].append((r["record_id"], platform))

    pairs = []
    twins = [r for r in registry if r["twin_of"]]
    for twin in twins:
        anchor_id = twin["twin_of"]
        twin_id = twin["entity_id"]
        for rid_a, plat_a in accounts_by_entity.get(anchor_id, []):
            for rid_b, plat_b in accounts_by_entity.get(twin_id, []):
                pairs.append({
                    "entity_id_a": anchor_id, "entity_id_b": twin_id,
                    "record_id_a": rid_a, "platform_a": plat_a,
                    "record_id_b": rid_b, "platform_b": plat_b,
                })
    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    os.makedirs(PLATFORM_DUMP_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)

    print("Building real-identity registry (with twin hard negatives)...")
    registry, num_twins = build_registry()
    print(f"  {len(registry)} identities ({NUM_IDENTITIES} base + {num_twins} twins)")

    community_of_entity = {r["entity_id"]: r["community_id"] for r in registry}

    print("Generating per-platform accounts...")
    accounts_by_platform = generate_platform_accounts(registry)
    for platform in PLATFORMS:
        print(f"  {platform}: {len(accounts_by_platform[platform])} accounts")

    print("Building per-platform community-correlated social graphs...")
    for platform in PLATFORMS:
        G = build_platform_graph(accounts_by_platform[platform], community_of_entity)
        edges_df = pd.DataFrame(list(G.edges()), columns=["record_id_a", "record_id_b"])
        edges_df.to_parquet(os.path.join(GRAPH_DIR, f"{platform}_edges.parquet"), index=False)
        avg_deg = (2 * G.number_of_edges() / G.number_of_nodes()) if G.number_of_nodes() else 0
        print(f"  {platform}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, avg degree {avg_deg:.1f}")

    print("Saving platform dumps...")
    for platform in PLATFORMS:
        df = pd.DataFrame(accounts_by_platform[platform])
        # Parquet can't store a list-of-dicts column directly via pyarrow's
        # default inference in all versions -> keep posts as JSON string.
        df["posts"] = df["posts"].apply(lambda p: json.dumps(p, ensure_ascii=False))
        df.to_parquet(os.path.join(PLATFORM_DUMP_DIR, f"{platform}.parquet"), index=False)

    print("Saving real identity registry...")
    registry_df = pd.DataFrame(registry)
    registry_df.to_parquet(os.path.join(BASE_DIR, "real_identity_registry.parquet"), index=False)

    print("Building evaluation scaffolding (ground truth + hard negatives)...")
    gt_pairs = build_ground_truth_pairs(accounts_by_platform)
    gt_pairs.to_parquet(os.path.join(BASE_DIR, "ground_truth_pairs.parquet"), index=False)
    print(f"  {len(gt_pairs)} true cross-platform pairs")

    hn_pairs = build_hard_negative_pairs(registry, accounts_by_platform)
    hn_pairs.to_parquet(os.path.join(BASE_DIR, "hard_negative_pairs.parquet"), index=False)
    print(f"  {len(hn_pairs)} hard-negative (twin) pairs")

    total_accounts = sum(len(v) for v in accounts_by_platform.values())
    print(f"\nDone. {len(registry)} identities -> {total_accounts} platform accounts across {len(PLATFORMS)} platforms.")


if __name__ == "__main__":
    main()
