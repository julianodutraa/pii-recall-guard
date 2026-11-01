"""Synthetic column generator for PII tagging experiments.

Every column is a list of string values (what a catalog scanner would see
after sampling a few rows) plus a ground-truth class. Column names are
deliberately NOT provided: real schemas are full of `col_17`, `attr_x` and
misleading names, so the detector must work from values alone.

The generator intentionally contains collisions that make the task hard in
the same way production data is hard:

* surnames and first names that are also city names (Lima, Santos, Florence),
* birth dates whose year range overlaps generic business dates,
* raw 11 digit national ids that look like long numeric order ids,
* free text comments where only a fraction of rows leak an email or phone.

All identifiers produced here are synthetic. Card numbers pass the Luhn check
and national ids pass a CPF-style checksum only so that the shapes are
realistic; none of them belong to anyone.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass

PII_CLASSES = [
    "email",
    "phone",
    "card_number",
    "national_id",
    "person_name",
    "street_address",
    "ip_address",
    "birth_date",
    "free_text_pii",
]
NON_PII_CLASSES = [
    "order_id",
    "sku",
    "event_timestamp",
    "amount",
    "status",
    "hostname",
    "uuid",
    "city",
    "calendar_date",
    "free_text",
]
CLASSES = PII_CLASSES + NON_PII_CLASSES
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
IS_PII = [c in PII_CLASSES for c in CLASSES]

FIRST = [
    "Ana", "Bruno", "Carla", "Diego", "Elisa", "Felipe", "Gabriela", "Hugo",
    "Isabela", "Joao", "Karina", "Lucas", "Mariana", "Nicolas", "Olivia",
    "Paulo", "Renata", "Sergio", "Tatiana", "Vitor", "Emma", "Liam", "Noah",
    "Sofia", "Mateo", "Yuki", "Priya", "Omar", "Chen", "Florence", "Victoria",
    "Regina", "Austin", "Paris", "Sydney", "Lorena",
]
LAST = [
    "Silva", "Souza", "Oliveira", "Santos", "Pereira", "Costa", "Almeida",
    "Ferreira", "Ribeiro", "Carvalho", "Gomes", "Martins", "Rocha", "Lima",
    "Barbosa", "Smith", "Johnson", "Nguyen", "Kim", "Muller", "Rossi",
    "Garcia", "Patel", "Kowalski", "Campinas", "Franca", "Londrina",
]
CITIES = [
    "Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Porto Alegre",
    "Recife", "Salvador", "Fortaleza", "Campinas", "Santos", "Lima", "Franca",
    "Londrina", "Florianopolis", "Lisbon", "Porto", "Austin", "Denver",
    "Seattle", "Toronto", "Florence", "Victoria", "Regina", "Paris", "Sydney",
    "Lorena", "Buenos Aires", "Santiago", "Bogota", "Madrid",
]
STREETS_BR = ["Rua das Flores", "Av. Paulista", "Rua Augusta", "Rua XV de Novembro",
              "Av. Brasil", "Rua da Consolacao", "Alameda Santos", "Rua Lima Barreto"]
STREETS_EN = ["Main", "Oak", "Maple", "Elm", "Lake", "Hill", "Park", "Cedar"]
STREET_SUFFIX = ["St", "Ave", "Rd", "Blvd", "Lane"]
DOMAINS = ["gmail.com", "outlook.com", "yahoo.com.br", "uol.com.br",
           "example.org", "corp-mail.net", "proton.me"]
HOST_PARTS = ["db", "api", "etl", "kafka", "redis", "spark", "airflow", "ingest", "pg"]
HOST_ENV = ["prod", "stg", "dev", "dr"]
HOST_ZONE = ["eu-west", "us-east", "sa-east", "ap-south"]
HOST_TLD = ["internal", "example.net", "svc.cluster.local", "corp-mail.net"]
STATUS = [
    ["ACTIVE", "INACTIVE", "PENDING"],
    ["queued", "running", "success", "failed", "skipped"],
    ["open", "closed", "on_hold"],
    ["Y", "N"],
]
PRODUCT_A = ["Steel", "Oak", "Copper", "Nylon", "Carbon", "Glass"]
PRODUCT_B = ["Bolt", "Desk", "Cable", "Valve", "Bracket", "Sensor"]
COMMENT_TEMPLATES = [
    "customer asked to reschedule delivery",
    "payment retried after timeout",
    "ticket escalated to tier 2",
    "batch {n} reprocessed after schema change",
    "duplicate order merged",
    "refund approved by finance",
    "address validation failed, kept original",
    "job {n} finished late due to upstream lag",
    "please check invoice {n} before closing",
    "user reported login issue on mobile",
]
LEAK_TEMPLATES = [
    "please call me at {phone}",
    "send the receipt to {email}",
    "contact {name} at {email}",
    "customer {name} asked for callback {phone}",
]


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def luhn_complete(prefix: str) -> str:
    """Append the Luhn check digit to a numeric prefix."""
    total = 0
    for i, ch in enumerate(reversed(prefix)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return prefix + str((10 - total % 10) % 10)


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return len(digits) > 1 and total % 10 == 0


def cpf_check_digits(base9: str) -> str:
    """CPF-style mod 11 check digits for a 9 digit base."""
    nums = [int(c) for c in base9]
    s1 = sum(v * w for v, w in zip(nums, range(10, 1, -1)))
    d1 = (s1 * 10) % 11 % 10
    nums.append(d1)
    s2 = sum(v * w for v, w in zip(nums, range(11, 1, -1)))
    d2 = (s2 * 10) % 11 % 10
    return f"{d1}{d2}"


def _name(rng: random.Random) -> str:
    f, l = rng.choice(FIRST), rng.choice(LAST)
    style = rng.random()
    if style < 0.55:
        return f"{f} {l}"
    if style < 0.75:
        return f"{l.upper()}, {f}"
    if style < 0.9:
        return f"{f} {rng.choice(string.ascii_uppercase)}. {l}"
    return f"{f} {l}".lower()


def _email(rng: random.Random) -> str:
    f, l = rng.choice(FIRST).lower(), rng.choice(LAST).lower()
    style = rng.random()
    if style < 0.4:
        local = f"{f}.{l}"
    elif style < 0.7:
        local = f"{f[0]}{l}{rng.randint(1, 99)}"
    else:
        local = f"{f}_{_digits(rng, 3)}"
    e = f"{local}@{rng.choice(DOMAINS)}"
    return e.upper() if rng.random() < 0.05 else e


def _phone(rng: random.Random) -> str:
    style = rng.random()
    if style < 0.35:
        return f"+55 {rng.randint(11, 99)} 9{_digits(rng, 4)}-{_digits(rng, 4)}"
    if style < 0.65:
        return f"({rng.randint(11, 99)}) 9{_digits(rng, 4)}-{_digits(rng, 4)}"
    if style < 0.85:
        return f"+1 ({_digits(rng, 3)}) 555-{_digits(rng, 4)}"
    return f"{_digits(rng, 3)}.555.{_digits(rng, 4)}"


def _card(rng: random.Random) -> str:
    prefix = rng.choice(["4", "51", "52", "55"])
    num = luhn_complete(prefix + _digits(rng, 15 - len(prefix)))
    if rng.random() < 0.5:
        return " ".join(num[i:i + 4] for i in range(0, 16, 4))
    return num


def _national_id(rng: random.Random) -> str:
    base = _digits(rng, 9)
    full = base + cpf_check_digits(base)
    if rng.random() < 0.6:
        return f"{full[:3]}.{full[3:6]}.{full[6:9]}-{full[9:]}"
    return full


def _address(rng: random.Random) -> str:
    if rng.random() < 0.55:
        s = f"{rng.choice(STREETS_BR)}, {rng.randint(1, 3000)}"
        if rng.random() < 0.3:
            s += f", apto {rng.randint(1, 200)}"
        return s
    return f"{rng.randint(1, 9999)} {rng.choice(STREETS_EN)} {rng.choice(STREET_SUFFIX)}"


def _ip(rng: random.Random) -> str:
    if rng.random() < 0.4:
        return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def _date(rng: random.Random, y0: int, y1: int) -> str:
    y, m, d = rng.randint(y0, y1), rng.randint(1, 12), rng.randint(1, 28)
    if rng.random() < 0.6:
        return f"{y:04d}-{m:02d}-{d:02d}"
    return f"{d:02d}/{m:02d}/{y:04d}"


def _order_id(rng: random.Random, style: float) -> str:
    if style < 0.4:
        return f"ORD-{_digits(rng, 8)}"
    if style < 0.75:
        return str(rng.randint(10**5, 10**9))
    return str(rng.randint(10**10, 10**11 - 1))  # 11 digits, collides with raw ids


def _sku(rng: random.Random) -> str:
    return f"SKU-{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}-{_digits(rng, 4)}"


def _timestamp(rng: random.Random, style: float) -> str:
    if style < 0.7:
        return (f"20{rng.randint(20, 26)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
                f"T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z")
    return str(rng.randint(1_600_000_000_000, 1_790_000_000_000))


def _amount(rng: random.Random) -> str:
    v = rng.lognormvariate(4, 1.5) * (-1 if rng.random() < 0.05 else 1)
    return f"{v:.2f}"


def _hostname(rng: random.Random) -> str:
    return (f"{rng.choice(HOST_PARTS)}-{rng.choice(HOST_ENV)}-{rng.randint(1, 20):02d}"
            f".{rng.choice(HOST_ZONE)}.{rng.choice(HOST_TLD)}")


def _uuid(rng: random.Random) -> str:
    h = "".join(rng.choice("0123456789abcdef") for _ in range(32))
    return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-{h[16:20]}-{h[20:]}"


def _comment(rng: random.Random) -> str:
    return rng.choice(COMMENT_TEMPLATES).format(n=rng.randint(100, 99999))


def _leak(rng: random.Random) -> str:
    return rng.choice(LEAK_TEMPLATES).format(phone=_phone(rng), email=_email(rng), name=_name(rng))


JUNK = ["", "", "N/A", "null", "unknown", "0", "-"]
DEFAULT_LEAK_RANGE = (0.08, 0.35)


@dataclass
class Column:
    values: list[str]
    label: str
    seed: int = 0
    leak_range: tuple[float, float] = DEFAULT_LEAK_RANGE

    @property
    def is_pii(self) -> bool:
        return self.label in PII_CLASSES

    def resample(self, n_rows: int) -> "Column":
        """Draw a larger sample from the same column population.

        The first len(self.values) rows are identical to the original sample,
        so this models "read more rows of the same table", not a new table.
        """
        return make_column(self.label, self.seed, n_rows, self.leak_range)


def make_column(label: str, seed: int, n_rows: int | None = None,
                leak_range: tuple[float, float] = DEFAULT_LEAK_RANGE) -> Column:
    """Sample one column of `label`.

    Column level parameters (sample size, format style, dirt rate, leak rate)
    come from one random stream and row values from another, both derived
    from `seed`, so a column can be re-read with more rows consistently.
    """
    prng = random.Random(f"params-{seed}")
    n = prng.randint(5, 25) if n_rows is None else n_rows
    style = prng.random()
    status_set = prng.choice(STATUS)
    leak_rate = prng.uniform(*leak_range)
    dirt_rate = prng.uniform(0.05, 0.4) if prng.random() < 0.3 else 0.0
    rng = random.Random(f"rows-{seed}")
    values: list[str] = []
    for _ in range(n):
        if label == "email":
            v = _email(rng)
        elif label == "phone":
            v = _phone(rng)
        elif label == "card_number":
            v = _card(rng)
        elif label == "national_id":
            v = _national_id(rng)
        elif label == "person_name":
            v = _name(rng)
        elif label == "street_address":
            v = _address(rng)
        elif label == "ip_address":
            v = _ip(rng)
        elif label == "birth_date":
            v = _date(rng, 1940, 2008)
        elif label == "free_text_pii":
            v = _leak(rng) if rng.random() < leak_rate else _comment(rng)
        elif label == "order_id":
            v = _order_id(rng, style)
        elif label == "sku":
            v = _sku(rng)
        elif label == "event_timestamp":
            v = _timestamp(rng, style)
        elif label == "amount":
            v = _amount(rng)
        elif label == "status":
            v = rng.choice(status_set)
        elif label == "hostname":
            v = _hostname(rng)
        elif label == "uuid":
            v = _uuid(rng)
        elif label == "city":
            v = rng.choice(CITIES)
        elif label == "calendar_date":
            v = _date(rng, 1995, 2026)
        elif label == "free_text":
            v = _comment(rng)
        else:
            raise ValueError(label)
        # dirt: nulls and junk tokens, as in a real sample of a production table
        junk = rng.choice(JUNK)
        if rng.random() < dirt_rate:
            v = junk
        values.append(v)
    return Column(values=values, label=label, seed=seed, leak_range=leak_range)


def make_dataset(n_per_class: int, seed: int,
                 leak_range: tuple[float, float] = DEFAULT_LEAK_RANGE) -> list[Column]:
    """Balanced dataset with `n_per_class` columns for each of the 19 classes."""
    rng = random.Random(seed)
    cols = [make_column(c, rng.getrandbits(48), None, leak_range)
            for c in CLASSES for _ in range(n_per_class)]
    rng.shuffle(cols)
    return cols


def has_visible_leak(values: list[str]) -> bool:
    """True if any sampled row of a free text column shows an email or phone."""
    from .features import _EMAIL, _PHONEISH
    return any(_EMAIL.search(v) or _PHONEISH.search(v) for v in values)
