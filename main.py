#!/usr/bin/env python3
"""A scripted Werewolf player for make.fun's werewolf-agent-01.

No model calls: every decision is computed from the game's view alone by the
strategy code below. The program long-polls the env at MAKEFUN_ENV_URL with
POST /observe and answers each waiting decision with POST /act until the env
reports the trial done.

Determinism: every answer is a pure function of the brief's trial seed
(MAKEFUN_TRIAL_SEED) and the views the env served. There is no wall clock and
no unseeded randomness anywhere in what is sent; variation in phrasing and
tie-breaks comes from a hash of the seed and the game state, so the replay
sends exactly what the live run sent.

Strategy, in short:
  * A state tracker rebuilds each game from its public log: speeches, votes,
    deaths, revealed roles, and every role claim and check report other
    players make (parsed from their statements). Claimed seer checks are
    tested against revealed roles; a claimant caught in a false check is a
    proven liar, and a claimant whose checks hold up becomes the village's
    trusted seer.
  * Werewolf: match the partner's kill pick (a kill needs unanimity), hunt
    the seer and the doctor, speak and vote like a villager, bus a partner
    only when they are doomed, and counterclaim seer when named by one.
  * Seer: check the unknown and the influential, reveal with a full check
    history when it can swing the day, and lead the vote onto a found wolf.
  * Doctor: guard the revealed seer and the village's leaders; claim only
    when about to be exiled.
  * Villager: follow a credible seer, call out proven liars, and vote the
    best-supported suspicion, coordinating with the votes already cast.
"""

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
TRIAL_SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "")

MAX_OBSERVE_ROUNDS = 20000


# --------------------------------------------------------------------------
# Env transport
# --------------------------------------------------------------------------

def post(path, body, timeout=60):
    """POST one JSON object to the env; return (status, parsed body)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        ENV_URL + path,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as err:
        try:
            raw = err.read().decode("utf-8")
            return err.code, json.loads(raw) if raw else {}
        except Exception:
            return err.code, {}
    except Exception:
        return 0, {}


# --------------------------------------------------------------------------
# Deterministic helpers
# --------------------------------------------------------------------------

def h_int(*parts):
    """A stable hash of the parts, for phrasing variety and tie-breaks."""
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:12], 16)


def choose(parts, variants):
    """Pick one variant deterministically from the given key parts."""
    if not variants:
        return ""
    return variants[h_int(*parts) % len(variants)]


def fold_text(text, max_chars=500):
    """Fold a statement to one line the env's way and keep it in the cap."""
    text = re.sub(r"[\t\n\v\f\r  ]+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    text = re.sub(r" +", " ", text).strip()
    if len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0].strip()
        text = cut if cut else text[:max_chars]
    return text


# --------------------------------------------------------------------------
# Game state, rebuilt from one view
# --------------------------------------------------------------------------

class State(object):
    """Everything one view tells us about one game, public and private."""

    def __init__(self, view):
        self.view = view
        self.day = view.get("day") or 1
        self.phase = view.get("phase") or ""
        you = view.get("you") or {}
        self.me = you.get("seat")
        self.my_name = you.get("name") or ""
        self.role = you.get("role") or "villager"
        self.alive_me = bool(you.get("alive"))
        partner = you.get("partner") or {}
        self.partner = partner.get("seat")
        self.partner_alive = bool(partner.get("alive")) if partner else False

        self.players = {}
        for p in view.get("players") or []:
            self.players[p.get("seat")] = {
                "name": p.get("name") or "",
                "alive": bool(p.get("alive")),
                "role": p.get("role"),
            }
        self.alive_seats = [s for s in sorted(self.players) if self.players[s]["alive"]]
        self.name_to_seat = {}
        for s in sorted(self.players):
            nm = self.players[s]["name"].lower()
            if nm:
                self.name_to_seat[nm] = s

        self.seer_results = {}
        self.seer_checks = []
        for r in view.get("seerResults") or []:
            self.seer_results[r.get("seat")] = bool(r.get("werewolf"))
            self.seer_checks.append((r.get("night"), r.get("seat"), bool(r.get("werewolf"))))
        lp = view.get("lastProtect")
        self.last_protect = lp if isinstance(lp, dict) else None
        self.wolf_votes = view.get("wolfVotes") or []

        self.speeches = []          # {seat, day, round, text}
        self.speeches_by = {}
        self.votes_by_day = {}      # day -> [(voter, target)]
        self.exiles = {}            # seat -> {role, day}
        self.deaths = {}            # seat -> {night, role}
        self.no_death_nights = set()
        for e in view.get("log") or []:
            kind = e.get("kind")
            d = e.get("day")
            if kind == "speech":
                self.speeches.append({
                    "seat": e.get("seat"), "day": d,
                    "round": e.get("round"), "text": e.get("text") or "",
                })
                self.speeches_by[e.get("seat")] = self.speeches_by.get(e.get("seat"), 0) + 1
            elif kind == "vote":
                self.votes_by_day.setdefault(d, []).append((e.get("seat"), e.get("target")))
            elif kind == "exile":
                if e.get("seat") is not None:
                    self.exiles[e.get("seat")] = {"role": e.get("role"), "day": d}
            elif kind == "dawn":
                if e.get("died") is None:
                    self.no_death_nights.add(d)
                else:
                    self.deaths[e.get("died")] = {"night": d, "role": e.get("role")}
        self.votes_today = self.votes_by_day.get(self.day, [])
        self.my_speeches_today = sum(
            1 for s in self.speeches if s["seat"] == self.me and s["day"] == self.day
        )
        self.speeches_today = [s for s in self.speeches if s["day"] == self.day]

        # What the table has claimed, parsed from its statements.
        self.seer_claims = {}       # seat -> {day, checks: {target: {verdict, day}}}
        self.doctor_claims = {}     # seat -> day first claimed
        self.accusations = []       # (accuser, target, day)
        self.defenses = []          # (defender, target, day)
        for sp in self.speeches:
            self._read_speech(sp)

    # -- small accessors ---------------------------------------------------

    def name(self, seat):
        p = self.players.get(seat) or {}
        return p.get("name") or ("seat %s" % seat)

    def ref(self, seat):
        return "seat %s (%s)" % (seat, self.name(seat))

    def is_alive(self, seat):
        p = self.players.get(seat)
        return bool(p and p["alive"])

    def revealed_role(self, seat):
        p = self.players.get(seat)
        return p.get("role") if p else None

    def alive_during(self, seat, night):
        """Whether a seat was alive at some point on the given night."""
        if seat in self.deaths:
            return night <= self.deaths[seat]["night"]
        if seat in self.exiles:
            return night <= self.exiles[seat]["day"]
        return True

    def tally_today(self):
        tally = {}
        for _voter, target in self.votes_today:
            if target is not None:
                tally[target] = tally.get(target, 0) + 1
        return tally

    # -- statement parsing ---------------------------------------------------

    def _normalise(self, text):
        """Lowercase a statement and rewrite names and refs as 'seat N'."""
        t = " " + text.lower() + " "
        for nm in sorted(self.name_to_seat, key=len, reverse=True):
            t = re.sub(r"\b%s\b" % re.escape(nm), " seat %d " % self.name_to_seat[nm], t)
        t = re.sub(r"player\s*#?\s*(\d{1,2})", r"seat \1", t)
        t = re.sub(r"#\s*(\d{1,2})\b", r"seat \1", t)
        return t

    def _read_speech(self, sp):
        speaker = sp["seat"]
        day = sp["day"]
        if speaker is None or speaker == self.me:
            return
        t = self._normalise(sp["text"])
        sentences = [s for s in re.split(r"[.!?;\n]+", t) if s.strip()]
        claimed_here = False
        for s in sentences:
            negated = ("not the seer" in s or "n't the seer" in s or "not a seer" in s
                       or "not the doctor" in s or "n't the doctor" in s
                       or "if i were" in s or "if i was" in s)
            if not negated:
                if re.search(r"\bi(?:'m| am) (?:the |a |your )?seer\b", s) or \
                   re.search(r"\bi claim (?:to be )?(?:the )?seer\b", s) or \
                   re.search(r"\bseer here\b", s) or \
                   re.search(r"\bas (?:the|your) seer\b", s):
                    if speaker not in self.seer_claims:
                        self.seer_claims[speaker] = {"day": day, "checks": {}}
                    claimed_here = True
                if re.search(r"\bi(?:'m| am) (?:the |a )?doctor\b", s) or \
                   re.search(r"\bdoctor here\b", s) or \
                   re.search(r"\bi claim (?:to be )?(?:the )?doctor\b", s) or \
                   re.search(r"\bi (?:protected|saved|healed)\b", s):
                    if speaker not in self.doctor_claims:
                        self.doctor_claims[speaker] = day
            is_claimant = claimed_here or speaker in self.seer_claims
            for target, verdict in self._verdicts(s):
                if target == speaker or target not in self.players:
                    continue
                if is_claimant:
                    checks = self.seer_claims[speaker]["checks"]
                    if target not in checks:
                        checks[target] = {"verdict": verdict, "day": day}
                elif verdict == "wolf":
                    self.accusations.append((speaker, target, day))
                else:
                    self.defenses.append((speaker, target, day))
            if not is_claimant:
                for m in re.finditer(r"(?:i suspect|i'm voting|i vote|vote for|lynch|"
                                     r"i am voting|my vote is)\s+(?:for\s+)?(?:seat\s*)?(\d{1,2})", s):
                    tgt = int(m.group(1))
                    if tgt != speaker and tgt in self.players:
                        self.accusations.append((speaker, tgt, day))
                m = re.search(r"seat\s*(\d{1,2})\s+is (?:very |so |really )?"
                              r"(?:suspicious|sus|shady|off|quiet)", s)
                if m:
                    tgt = int(m.group(1))
                    if tgt != speaker and tgt in self.players:
                        self.accusations.append((speaker, tgt, day))

    @staticmethod
    def _verdicts(sentence):
        """(seat, 'wolf'|'clean') pairs stated in one normalised sentence."""
        refs = [int(x) for x in re.findall(r"seat\s*(\d{1,2})", sentence)]
        if not refs and ("checked" in sentence or "inspected" in sentence
                         or "visited" in sentence):
            refs = [int(x) for x in re.findall(
                r"(?:checked|inspected|visited)\s+(?:seat\s*)?(\d{1,2})", sentence)]
        if not refs:
            return []
        s = sentence
        clean = ("not a wolf" in s or "not a werewolf" in s or "n't a wolf" in s
                 or "not the wolf" in s or "clean" in s or "innocent" in s
                 or re.search(r"\bvillager\b", s) is not None)
        wolf = ("werewolf" in s or re.search(r"\bwolf\b", s) is not None
                or "wolves" in s)
        out = []
        if len(refs) == 1:
            if clean:
                out.append((refs[0], "clean"))
            elif wolf:
                out.append((refs[0], "wolf"))
            return out
        for m in re.finditer(r"seat\s*(\d{1,2})\s+is\s+(?:a\s+|the\s+)?(werewolf|wolf|clean|innocent|villager)\b", s):
            out.append((int(m.group(1)), "clean" if m.group(2) in ("clean", "innocent", "villager") else "wolf"))
        for m in re.finditer(r"seat\s*(\d{1,2})\s+is\s+not\s+(?:a\s+)?(werewolf|wolf)\b", s):
            out.append((int(m.group(1)), "clean"))
        return out


# --------------------------------------------------------------------------
# Analysis: who to believe, who is confirmed, who looks like a wolf
# --------------------------------------------------------------------------

class Info(object):
    """The village's best current picture of one game, from a State."""

    def __init__(self, st):
        self.st = st
        self.claim_eval = {}        # claimant seat -> {false, strong, day}
        for seat, claim in st.seer_claims.items():
            ev = {"false": False, "strong": 0, "day": claim["day"]}
            for target, chk in claim["checks"].items():
                role = st.revealed_role(target)
                if role is None:
                    continue
                said_wolf = chk["verdict"] == "wolf"
                if said_wolf != (role == "werewolf"):
                    ev["false"] = True
                else:
                    died_day = None
                    if target in st.deaths:
                        died_day = st.deaths[target]["night"]
                    elif target in st.exiles:
                        died_day = st.exiles[target]["day"]
                    if died_day is not None and chk["day"] < died_day:
                        ev["strong"] += 1   # named the truth before it was public
            self.claim_eval[seat] = ev
        self.credible_seer = self._credible_seer()
        self.confirmed_wolves = []
        self.confirmed_village = set()
        self.liars = []
        self._confirmed()
        self.suspicion, self.reasons = self._suspicion()
        self.pressure = self._pressure()

    def _credible_seer(self):
        st = self.st
        # A claimant whose role is already revealed settles it: a revealed
        # seer told the truth, and their checks stand.
        for seat in st.seer_claims:
            if st.revealed_role(seat) == "seer":
                return seat
        if st.role == "seer":
            return st.me
        living = [s for s in st.seer_claims
                  if st.is_alive(s) and not self.claim_eval[s]["false"]]
        if not living:
            return None
        if len(living) == 1:
            return living[0]

        def rank(s):
            ev = self.claim_eval[s]
            named_by_rival = 0
            for other in living:
                if other != s:
                    chk = st.seer_claims[other]["checks"].get(s)
                    if chk and chk["verdict"] == "wolf":
                        named_by_rival = 1
            return (ev["strong"], -named_by_rival, -ev["day"], -s)

        return max(living, key=rank)

    def _confirmed(self):
        st = self.st
        wolves = []
        village = set()
        cs = self.credible_seer
        if cs is not None and cs != st.me:
            for target, chk in st.seer_claims[cs]["checks"].items():
                if not st.is_alive(target):
                    continue
                if chk["verdict"] == "wolf":
                    wolves.append(target)
                else:
                    village.add(target)
        if st.role == "seer":
            for target, is_wolf in st.seer_results.items():
                if not st.is_alive(target):
                    continue
                if is_wolf:
                    if target not in wolves:
                        wolves.append(target)
                else:
                    village.add(target)
        self.confirmed_wolves = sorted(wolves)
        self.confirmed_village = village
        self.liars = sorted(
            s for s in st.seer_claims
            if self.claim_eval[s]["false"] and st.is_alive(s) and s != cs
        )

    def _suspicion(self):
        st = self.st
        score = {s: 0.0 for s in st.players if s != st.me}
        reasons = {s: [] for s in score}

        def add(seat, pts, why):
            if seat in score and st.is_alive(seat):
                score[seat] += pts
                if why:
                    reasons[seat].append(why)

        # Accusations from distinct players.
        by_target = {}
        for accuser, target, _day in st.accusations:
            by_target.setdefault(target, set()).add(accuser)
        for target, accusers in by_target.items():
            real = [a for a in accusers if a != st.me]
            if real:
                add(target, min(0.7 * len(real), 2.1),
                    "accused by %s" % ", ".join(st.ref(a) for a in sorted(real)[:2]))

        # Voting history against what the exiles turned out to be.
        for day, votes in st.votes_by_day.items():
            if day == st.day:
                continue
            exiled = [s for s, ex in st.exiles.items() if ex["day"] == day]
            if not exiled:
                continue
            exiled = exiled[0]
            ex_role = st.exiles[exiled]["role"]
            for voter, target in votes:
                if voter == st.me or target is None:
                    continue
                if target == exiled and ex_role != "werewolf":
                    add(voter, 1.2, "voted to exile %s, who was %s"
                        % (st.ref(exiled), ex_role))
                if target != exiled and ex_role == "werewolf" and target != voter:
                    add(voter, 0.6, "voted elsewhere the day %s, a wolf, went out"
                        % st.ref(exiled))

        # Defending or accusing the wrong people, in words.
        for defender, target, _day in st.defenses:
            if st.revealed_role(target) == "werewolf":
                add(defender, 1.0, "defended %s, who was a wolf" % st.ref(target))
        for accuser, target, _day in st.accusations:
            if target in self.confirmed_village:
                add(accuser, 0.8, "accused %s, who checks out clean" % st.ref(target))

        # A rival seer claimant the village should not believe.
        for seat in st.seer_claims:
            if seat != self.credible_seer and st.is_alive(seat) and seat != st.me:
                if len(st.seer_claims) > 1 or self.claim_eval[seat]["false"]:
                    add(seat, 2.0, "their seer claim does not hold up")

        # Quiet seats hide wolves; loud accurate seats rarely are.
        if st.day >= 2:
            for s in score:
                if not st.is_alive(s):
                    continue
                n = st.speeches_by.get(s, 0)
                if n == 0:
                    add(s, 1.0, "has said nothing all game")
                elif n == 1:
                    add(s, 0.4, "has barely spoken")
        for accuser, target, _day in st.accusations:
            if st.revealed_role(target) == "werewolf":
                add(accuser, -1.2, "")
        for s in self.confirmed_village:
            if s in score:
                score[s] = -10.0
                reasons[s] = ["checked clean by the seer"]
        for s in self.confirmed_wolves:
            if s in score:
                score[s] = 50.0
                reasons[s] = ["named a werewolf by the seer"]
        for s in self.liars:
            if s in score and score[s] < 40.0:
                score[s] = 40.0
                reasons[s] = ["caught in a false seer check"]
        return score, reasons

    def _pressure(self):
        """How hard the table (not me) is pushing each seat today and lately."""
        st = self.st
        pressure = {s: 0.0 for s in st.players}
        for accuser, target, day in st.accusations:
            if accuser != st.me and target in pressure:
                pressure[target] += 1.0 if day == st.day else 0.4
        for target, n in st.tally_today().items():
            if target in pressure:
                pressure[target] += 1.5 * n
        return pressure

    def top_suspect(self, candidates):
        """The highest-suspicion seat among candidates, deterministically."""
        best = None
        best_key = None
        for s in candidates:
            key = (self.suspicion.get(s, 0.0), -h_int(TRIAL_SEED, "sus", s) % 997)
            if best_key is None or key > best_key:
                best_key = key
                best = s
        return best


# --------------------------------------------------------------------------
# Speech
# --------------------------------------------------------------------------

def my_claimed(st, word):
    """Whether my own earlier statements claimed the given role."""
    needle = "i am the " + word
    return any(needle in s["text"].lower()
               for s in st.speeches if s["seat"] == st.me)


def check_phrase(st, seat, is_wolf):
    return "%s: %s" % (st.ref(seat), "a werewolf" if is_wolf else "not a werewolf")


def real_history(st):
    """My seer checks as a spoken list, night by night."""
    parts = []
    for night, seat, is_wolf in sorted(st.seer_checks):
        if seat is not None:
            parts.append("Night %d I checked %s" % (night, check_phrase(st, seat, is_wolf)))
    return ". ".join(parts)


def fake_history(st, wolf_target):
    """A werewolf's invented seer history, consistent with public facts.

    'Clean' checks go on players whose revealed roles agree, or on quiet
    living villagers; the one 'wolf' check goes on the target. Every claimed
    check names a player who was alive on that night.
    """
    nights = list(range(1, st.day + 1))
    cleans_pool = []
    for s in sorted(st.players):
        if s in (st.me, st.partner, wolf_target):
            continue
        role = st.revealed_role(s)
        if role is not None and role != "werewolf":
            cleans_pool.append(s)       # revealed village: a safe 'clean'
    for s in sorted(st.players):
        if s in (st.me, st.partner, wolf_target) or s in cleans_pool:
            continue
        if st.revealed_role(s) is None and s != info_seer_seat(st):
            cleans_pool.append(s)       # living and unrevealed
    lines = []
    wolf_night = None
    for n in reversed(nights):
        if st.alive_during(wolf_target, n):
            wolf_night = n
            break
    used = set()
    for n in nights:
        if n == wolf_night:
            lines.append("Night %d I checked %s" % (n, check_phrase(st, wolf_target, True)))
            used.add(wolf_target)
            continue
        pick = None
        for cand in cleans_pool:
            if cand not in used and st.alive_during(cand, n):
                pick = cand
                break
        if pick is None:
            continue
        used.add(pick)
        lines.append("Night %d I checked %s" % (n, check_phrase(st, pick, False)))
    return ". ".join(lines)


def info_seer_seat(st):
    """The seat the table most believes is seer, or None (helper, no Info)."""
    best = None
    for seat in st.seer_claims:
        if st.is_alive(seat):
            best = seat
    return best


def cover_reason(st, target):
    """A mundane, invented-but-plausible reason, safe for a hidden seer."""
    return choose((TRIAL_SEED, "cover", st.day, target), [
        "their story shifts a little each time they tell it",
        "they were quick to agree with everyone and commit to nothing",
        "their vote yesterday sat wrong with me",
        "they push whoever is already losing, and never lead",
    ])


def reason_for(st, info, target):
    rs = [r for r in info.reasons.get(target, []) if r]
    if rs:
        if st.role == "seer" and not my_claimed(st, "seer") and "seer" in rs[0]:
            return cover_reason(st, target)   # never leak my own checks
        return rs[0]
    return choose((TRIAL_SEED, "reason", st.day, target), [
        "they echo everyone and commit to nothing",
        "their votes do not match their words",
        "they went quiet once the talk turned serious",
        "they push whoever is already losing, and never lead",
    ])


def villager_speech(st, info):
    day = st.day
    me = st.me
    key = (TRIAL_SEED, st.day, st.my_speeches_today, me)
    if info.confirmed_wolves and info.credible_seer != me:
        t = info.confirmed_wolves[0]
        cs = info.credible_seer
        return choose(key + ("endorse",), [
            "%s says %s is a werewolf, and nothing I have seen contradicts it. I am voting %s."
            % (st.ref(cs), st.ref(t), st.ref(t)),
            "The seer's check is clear: %s is a werewolf. Let us not split the vote; I am on %s."
            % (st.ref(t), st.ref(t)),
        ])
    if info.liars:
        t = info.liars[0]
        bad = None
        for tgt, chk in st.seer_claims[t]["checks"].items():
            role = st.revealed_role(tgt)
            if role is not None and (chk["verdict"] == "wolf") != (role == "werewolf"):
                bad = (tgt, role)
        detail = ("their check on %s was wrong, %s turned out to be %s"
                  % (st.ref(bad[0]), st.name(bad[0]), bad[1])) if bad else "their checks do not survive what we know"
        return choose(key + ("liar",), [
            "%s claimed seer, but %s. That is a caught lie, and I am voting %s."
            % (st.ref(t), detail, st.ref(t)),
            "We caught %s in a false check: %s. A real seer does not miss like that. My vote is %s."
            % (st.ref(t), detail, st.ref(t)),
        ])
    claimants = [s for s in st.seer_claims if st.is_alive(s)]
    if len(claimants) >= 2 and info.credible_seer is not None \
            and info.credible_seer != me:
        cs = info.credible_seer
        other = [s for s in claimants if s != cs][0]
        return choose(key + ("weigh",), [
            "Two seer claims, and they cannot both be true. %s's checks fit what we know; %s's story is thinner. I trust %s more today."
            % (st.ref(cs), st.ref(other), st.name(cs)),
            "On the seer claims: I lean to %s. Their history lines up with the reveals; %s has not earned that trust yet."
            % (st.ref(cs), st.ref(other)),
        ])
    cands = [s for s in st.alive_seats if s != me and s not in info.confirmed_village]
    t = info.top_suspect(cands) if cands else None
    if t is None:
        return None
    if day == 1 and st.my_speeches_today == 0 and not st.speeches_today:
        return choose(key + ("open",), [
            "Day one, and there is little to go on. I am a villager; I will judge by votes and by who just echoes. %s, I am listening to you closely."
            % st.ref(t),
            "Not much is solid yet. I learn more from votes than from speeches, so I am watching how everyone lines up behind %s and the rest."
            % st.ref(t),
        ])
    if st.my_speeches_today >= 1:
        leader = max((s for s in st.alive_seats if s != me),
                     key=lambda s: info.pressure.get(s, 0.0), default=None)
        if leader is not None and info.pressure.get(leader, 0.0) >= 2 and leader in info.confirmed_village:
            return ("Careful: the push on %s worries me, they check out clean. "
                    "I would rather vote %s." % (st.ref(leader), st.ref(t)))
        return choose(key + ("close",), [
            "I have said my piece: %s is my vote today, because %s."
            % (st.ref(t), reason_for(st, info, t)),
            "Settled on my side: I am voting %s. Splitting helps only the wolves."
            % st.ref(t),
        ])
    return choose(key + ("accuse",), [
        "I am looking hardest at %s: %s. Unless something better comes up, that is my vote."
        % (st.ref(t), reason_for(st, info, t)),
        "My read today: %s, because %s. I am open to being argued out of it, but not by volume alone."
        % (st.ref(t), reason_for(st, info, t)),
        "%s is where my vote is leaning: %s. What do the rest of you see?"
        % (st.ref(t), reason_for(st, info, t)),
    ])


def seer_speech(st, info):
    day = st.day
    me = st.me
    key = (TRIAL_SEED, day, st.my_speeches_today, me)
    found_wolves = [s for s, w in st.seer_results.items() if w and st.is_alive(s)]
    found_clean = [s for s, w in st.seer_results.items() if not w and st.is_alive(s)]
    already_out = my_claimed(st, "seer")
    pressure_me = info.pressure.get(me, 0.0)
    fakes = [s for s in st.seer_claims if s != me and st.is_alive(s)]
    fake_lies = False
    fake_named_me = False
    for f in fakes:
        for tgt, chk in st.seer_claims[f]["checks"].items():
            if tgt == me and chk["verdict"] == "wolf":
                fake_named_me = True
            if tgt in st.seer_results and (chk["verdict"] == "wolf") != st.seer_results[tgt]:
                fake_lies = True
    reveal = already_out
    if not already_out:
        if found_wolves and day >= 2:
            reveal = True
        if found_wolves and pressure_me >= 2:
            reveal = True
        if fake_named_me or fake_lies:
            reveal = True
        if day >= 4 and len(st.seer_results) >= 2:
            reveal = True
        if pressure_me >= 3 and st.seer_results:
            reveal = True
        if len(st.alive_seats) <= 4 and st.seer_results:
            reveal = True
    hist = real_history(st)
    if reveal and hist:
        if found_wolves:
            t = found_wolves[0]
            tail = choose(key + ("revtail",), [
                "%s is a werewolf; vote there today, and the doctor should keep me breathing tonight."
                % st.ref(t),
                "Vote %s today. Every name above is exactly what I was shown." % st.ref(t),
            ])
            return "I am the seer. %s. %s" % (hist, tail)
        cleans = ", ".join(st.ref(s) for s in found_clean)
        verb = "is" if len(found_clean) == 1 else "are"
        return ("I am the seer. %s. No living wolf in my checks, but %s %s clear; "
                "do not waste votes there." % (hist, cleans or "nobody yet", verb))
    if already_out and found_wolves:
        t = found_wolves[0]
        return ("As I said, I am the seer: %s is a werewolf. My checks: %s. Vote %s."
                % (st.ref(t), hist, st.ref(t)))
    if already_out:
        return ("I am the seer, and my checks stand: %s. Beyond them, my best read is %s."
                % (hist, st.ref(info.top_suspect(
                    [s for s in st.alive_seats if s != me and s not in info.confirmed_village]) or me)))
    # Stay hidden: talk like a villager, with two quiet exceptions.
    if day == 1 and found_wolves:
        t = found_wolves[0]
        return choose(key + ("soft",), [
            "Nothing provable from me yet, but %s does not sit right: %s. I am watching them."
            % (st.ref(t), cover_reason(st, t)),
            "Early days. If I had to point now, it would be at %s: %s."
            % (st.ref(t), cover_reason(st, t)),
        ])
    pressured_clean = [s for s in found_clean if info.pressure.get(s, 0.0) >= 2]
    if pressured_clean:
        c = pressured_clean[0]
        alt = info.top_suspect([s for s in st.alive_seats
                                 if s != me and s not in info.confirmed_village and s != c])
        return ("I would not vote %s today; nothing about them reads wolf to me. "
                "Look at %s instead." % (st.ref(c), st.ref(alt or c)))
    return villager_speech(st, info)


def doctor_speech(st, info):
    me = st.me
    key = (TRIAL_SEED, st.day, st.my_speeches_today, me)
    pressure_me = info.pressure.get(me, 0.0)
    if pressure_me >= 3 and not my_claimed(st, "doctor"):
        lp = st.last_protect
        line = ""
        if lp and lp.get("seat"):
            if lp.get("night") == st.day:
                line = " Last night I protected %s." % st.ref(lp["seat"])
            else:
                line = " On night %d I protected %s." % (lp["night"], st.ref(lp["seat"]))
        return ("Before this goes further: I am the doctor.%s Exiling me leaves "
                "the village unguarded tonight, so look again at who benefits from that."
                % line)
    lp = st.last_protect
    if (lp and lp.get("seat") and lp.get("night") == st.day
            and st.day in st.no_death_nights
            and info.pressure.get(lp["seat"], 0.0) >= 2):
        alt = info.top_suspect([s for s in st.alive_seats
                                 if s != me and s not in info.confirmed_village
                                 and s != lp["seat"]])
        if alt is not None:
            return ("No one died last night, and I do not buy the case on %s. "
                    "The quiet night smells like a failed kill, not a guilty %s. "
                    "I would look at %s." % (st.ref(lp["seat"]), st.name(lp["seat"]), st.ref(alt)))
    return villager_speech(st, info)


def wolf_speech(st, info):
    me = st.me
    partner = st.partner
    day = st.day
    key = (TRIAL_SEED, day, st.my_speeches_today, me)
    cs = info.credible_seer
    named = set(info.confirmed_wolves)
    if me in named and not my_claimed(st, "seer") and cs is not None and st.is_alive(cs):
        hist = fake_history(st, cs)
        return ("I am the seer, and I kept quiet because a loud seer dies. %s. "
                "The claim from %s is the fake one; I checked them myself."
                % (hist, st.ref(cs)))
    if my_claimed(st, "seer"):
        target = cs if (cs is not None and cs != me and st.is_alive(cs)) else None
        if target is not None:
            return ("My checks stand, and %s is still the fake seer at this table. "
                    "Vote %s and the village keeps its real seer." % (st.ref(target), st.ref(target)))
    if partner is not None and partner in named and st.is_alive(partner):
        if len(st.alive_seats) <= 5 and not my_claimed(st, "seer") and cs is not None and st.is_alive(cs):
            hist = fake_history(st, cs)
            return ("I am the seer. %s. %s's claim is a wolf's trick to burn the "
                    "village's trust; do not let it work." % (hist, st.ref(cs)))
        return choose(key + ("bus",), [
            "The seer's check on %s is hard to argue with. I did not see it coming, "
            "but I will not stand in the village's way today." % st.ref(partner),
            "If %s is really a wolf, they fooled me too. The check is the check; "
            "I vote with the evidence." % st.ref(partner),
        ])
    # Blend in: push whoever the village already leans on, never the partner.
    cands = [s for s in st.alive_seats
             if s != me and s != partner and s not in info.confirmed_village]
    if not cands:
        return None
    t = max(cands, key=lambda s: (info.pressure.get(s, 0.0),
                                  info.suspicion.get(s, 0.0) if s in info.suspicion else 0.0,
                                  -s))
    if day == 1 and st.my_speeches_today == 0 and not st.speeches_today:
        return choose(key + ("wopen",), [
            "Day one and mostly guesswork. I am a villager; %s feels a touch rehearsed to me, but I am listening."
            % st.ref(t),
            "Little to go on yet. I will judge by the votes; for now %s is the seat I understand least."
            % st.ref(t),
        ])
    if st.my_speeches_today >= 1:
        return choose(key + ("wclose",), [
            "I am settled enough: %s is my vote. Let us see the flip and learn from it."
            % st.ref(t),
            "Still %s for me. Splitting the vote only helps the wolves." % st.ref(t),
        ])
    base = choose(key + ("waccuse",), [
        "I keep coming back to %s: %s. That is where my vote goes unless the talk moves me."
        % (st.ref(t), reason_for(st, info, t)),
        "Put me down as suspicious of %s: %s. Happy to hear a better case."
        % (st.ref(t), reason_for(st, info, t)),
    ])
    if (partner is not None and st.is_alive(partner)
            and 0 < info.pressure.get(partner, 0.0) < 2
            and partner not in named):
        base += " For what it is worth, %s has read fine to me so far." % st.ref(partner)
    return base


def speak_text(st, info, max_chars):
    """The statement for this speak decision, or None to pass."""
    if st.role == "seer":
        txt = seer_speech(st, info)
    elif st.role == "doctor":
        txt = doctor_speech(st, info)
    elif st.role == "werewolf":
        txt = wolf_speech(st, info)
    else:
        txt = villager_speech(st, info)
    if not txt:
        return None
    txt = fold_text(txt, max_chars or 500)
    if not txt:
        return None
    mine_today = [s["text"] for s in st.speeches
                  if s["seat"] == st.me and s["day"] == st.day]
    if txt in mine_today:
        return None
    return txt


# --------------------------------------------------------------------------
# Night and vote decisions
# --------------------------------------------------------------------------

def decide_inspect(st, info, options):
    """Seer: check the unknown, the influential and the contested."""
    if not options:
        return None
    fresh = [s for s in options if s not in st.seer_results]
    pool = fresh if fresh else list(options)

    def score(s):
        v = 0.0
        if s in st.seer_claims:
            v += 3.0            # settle a rival claim
        v += min(st.speeches_by.get(s, 0), 4) * 0.4
        if info.pressure.get(s, 0.0) >= 2:
            v += 1.5
        if st.speeches_by.get(s, 0) == 0:
            v += 1.0            # a silent seat may be a hiding wolf
        for _day, votes in st.votes_by_day.items():
            for voter, target in votes:
                if voter == s and target in st.exiles \
                        and st.exiles[target]["role"] != "werewolf":
                    v += 1.0
        return (v, -h_int(TRIAL_SEED, "insp", st.day, s) % 997)

    return max(pool, key=score)


def decide_protect(st, info, options):
    """Doctor: guard the seer first, then the village's leaders."""
    if not options:
        return None
    doctor_dead = any(st.revealed_role(s) == "doctor" for s in st.players)

    def score(s):
        v = 0.0
        if s == info.credible_seer:
            v += 6.0
        elif s in st.seer_claims:
            v += 2.0
        if s in info.confirmed_village:
            v += 2.0
        if st.speeches_by.get(s, 0) >= 3:
            v += 1.0
        for accuser, target, _day in st.accusations:
            if accuser == s and st.revealed_role(target) == "werewolf":
                v += 2.0        # led a wolf hunt; wolves will want revenge
        v += min(info.pressure.get(s, 0.0), 3.0) * 0.3
        return (v, -h_int(TRIAL_SEED, "prot", st.day, s) % 997)

    if doctor_dead:
        return None             # unreachable in practice; options come from the env
    return max(options, key=score)


def decide_kill(st, info, options):
    """Werewolf: match the partner's pick; else hunt the seer and doctor."""
    if not options:
        return None
    if st.partner is not None:
        for v in reversed(st.wolf_votes):
            if v.get("night") == st.day and v.get("seat") == st.partner \
                    and v.get("target") in options:
                return v["target"]      # a kill needs both wolves on one seat
    doctor_dead = any(st.revealed_role(s) == "doctor" for s in st.players)
    top_suspect = None
    best_pressure = 0.0
    for s in options:
        if info.pressure.get(s, 0.0) > best_pressure:
            best_pressure = info.pressure.get(s, 0.0)
            top_suspect = s

    def score(s):
        v = 0.0
        if s in st.seer_claims:
            claim_day = st.seer_claims[s]["day"]
            if s == info.credible_seer:
                v += 6.0
            else:
                v += 2.5
            if not doctor_dead:
                if claim_day >= st.day - 1:
                    v -= 2.5    # freshly out: the doctor is probably on them
                if claim_day <= st.day - 2 and (st.day - 1) in st.no_death_nights:
                    v += 3.0    # a saved night means tonight they are exposed
        if s in st.doctor_claims:
            v += 4.0
        if s in info.confirmed_village and st.speeches_by.get(s, 0) >= 2:
            v += 3.0            # a trusted, vocal villager coordinates votes
        for accuser, target, _day in st.accusations:
            if accuser == s and target in (st.me, st.partner):
                v += 1.5        # silence a voice hunting us
        if s == top_suspect and best_pressure >= 2:
            v -= 3.0            # let the village waste tomorrow's exile there
        if st.speeches_by.get(s, 0) == 0 and st.day >= 2:
            v += 0.7            # a silent seat might be a hiding seer
        return (v, -h_int(TRIAL_SEED, "kill", st.day, s) % 997)

    return max(options, key=score)


def decide_vote(st, info, options, game):
    if not options:
        return None
    if st.role == "werewolf":
        return wolf_vote(st, info, options)
    return village_vote(st, info, options, game)


def village_vote(st, info, options, game):
    tally = st.tally_today()
    cw = [s for s in info.confirmed_wolves if s in options]
    if cw:
        return max(cw, key=lambda s: (tally.get(s, 0), -s))
    liars = [s for s in info.liars if s in options]
    if liars:
        return max(liars, key=lambda s: (tally.get(s, 0), -s))
    safe = set(info.confirmed_village)
    if st.role == "doctor" and st.last_protect:
        lp = st.last_protect
        if lp.get("seat") and lp.get("night") == st.day \
                and st.day in st.no_death_nights:
            safe.add(lp["seat"])        # I likely saved them: not a wolf
    cands = [s for s in options if s not in safe] or list(options)
    my_tally = tally.get(st.me, 0)
    if my_tally >= 2:
        rivals = [s for s in cands if tally.get(s, 0) > 0]
        if rivals:
            return max(rivals, key=lambda s: (tally.get(s, 0), -s))

    def eff(s):
        base = info.suspicion.get(s, 0.0)
        t = tally.get(s, 0)
        return base + (0.9 * t if base > 0.5 else 0.3 * t)

    best = max(cands, key=lambda s: (eff(s), -h_int(TRIAL_SEED, "vote", game, st.day, s) % 89))
    if eff(best) <= 0:
        joined = [s for s in cands if tally.get(s, 0) > 0]
        if joined:
            return max(joined, key=lambda s: (tally.get(s, 0), -s))
        return cands[h_int(TRIAL_SEED, "d1vote", game, st.day) % len(cands)]
    return best


def wolf_vote(st, info, options):
    tally = st.tally_today()
    partner = st.partner
    if partner is not None and partner in options \
            and partner in info.confirmed_wolves:
        if tally.get(partner, 0) >= 2 or len(st.alive_seats) <= 5:
            return partner              # doomed: bus and blend in
    cands = [s for s in options if s != partner] or list(options)
    my_tally = tally.get(st.me, 0)
    if my_tally >= 1:
        rivals = [s for s in cands if tally.get(s, 0) >= my_tally]
        if rivals:
            return max(rivals, key=lambda s: (tally.get(s, 0), -s))
    led = [s for s in cands if tally.get(s, 0) > 0]
    if led:
        return max(led, key=lambda s: (tally.get(s, 0),
                                       info.pressure.get(s, 0.0), -s))
    return max(cands, key=lambda s: (info.pressure.get(s, 0.0),
                                     info.suspicion.get(s, 0.0), -s))


# --------------------------------------------------------------------------
# The run: observe, decide, act, until the trial is done
# --------------------------------------------------------------------------

def fallback_act(pending):
    game = pending.get("game")
    kind = pending.get("decision")
    if kind == "speak":
        return {"game": game, "decision": "speak", "pass": True}
    return {"game": game, "decision": kind, "target": None}


def decide(pending):
    """One env decision -> one /act body, computed from its view alone."""
    game = pending.get("game")
    kind = pending.get("decision")
    view = pending.get("view") or {}
    st = State(view)
    info = Info(st)
    options = [o for o in (pending.get("options") or []) if isinstance(o, int)]
    if kind == "speak":
        txt = speak_text(st, info, pending.get("max_chars") or 500)
        if txt:
            return {"game": game, "decision": "speak", "text": txt}
        return {"game": game, "decision": "speak", "pass": True}
    if kind == "inspect":
        target = decide_inspect(st, info, options)
    elif kind == "protect":
        target = decide_protect(st, info, options)
    elif kind == "kill":
        target = decide_kill(st, info, options)
    elif kind == "vote":
        target = decide_vote(st, info, options, game)
    else:
        return fallback_act(pending)
    if target is not None and target not in options:
        target = None
    return {"game": game, "decision": kind, "target": target}


def main():
    if not ENV_URL:
        return 1
    rounds = 0
    while rounds < MAX_OBSERVE_ROUNDS:
        rounds += 1
        status, obs = post("/observe", {}, timeout=45)
        if status != 200 or not isinstance(obs, dict):
            time.sleep(1)
            continue
        if obs.get("done"):
            break
        for pending in obs.get("pending") or []:
            try:
                act = decide(pending)
            except Exception:
                act = fallback_act(pending)
            status, _body = post("/act", act, timeout=30)
            if status != 200:
                fb = fallback_act(pending)
                if act != fb:
                    try:
                        post("/act", fb, timeout=30)
                    except Exception:
                        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



