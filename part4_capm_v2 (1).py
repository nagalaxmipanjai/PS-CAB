"""
================================================================================
PART 4 (v2) -- CASE-AWARE PHRASE MODULATION (CAPM)
English -> Telugu NMT, same backbone, same frozen split, same environment as Parts 1-3
================================================================================
WHY THIS PART EXISTS
  Part 1-3 showed that PS-CAB (a gated residual on PREPOSITION tokens of the encoder output)
  does not improve BLEU / chrF++ and that the network learns to discount the added signal.
  Diagnosis: English prepositions are realised in Telugu as case suffixes / postpositions on the
  OBJECT NOUN (e.g. "in Hyderabad" -> హైదరాబాద్‌లో). Emphasising the preposition token alone is
  redundant, and nothing in the training objective rewards getting the suffix right.

  CAPM changes three things, each with its own ablation:
    (1) PHRASE MASK      the module acts on the preposition + its object noun phrase
                         (spaCy dependency parse: ADP -> pobj/pcomp, left edge .. head noun),
                         i.e. where Telugu puts the case marker.
    (2) CASE-WEIGHTED CE target sub-word tokens that realise a Telugu case marker /
                         postposition get weight (1 + case_w) in the cross-entropy.
    (3) GUIDED CROSS-ATTENTION  when the decoder emits a case-marker token, a small auxiliary
                         loss rewards cross-attention mass on the source preposition phrases
                         (supervised attention; mean over heads of a fixed set of decoder layers).

  New automatic metric: CASE-MARKER F1 (reference-based, multiset of 24 Telugu case /
  postposition classes per sentence, corpus micro-F1, exact bootstrap via sufficient statistics).

PROTOCOL (identical to Parts 1-3 so the existing baseline / PS-CAB runs are reused unchanged)
  same 40k split, same 3,000-step budget, same optimiser, same early stopping on UNWEIGHTED
  validation NLL, same seeds, beam 4, same pinned transformers 5.16.1.
  The go / no-go criteria are written to PREREGISTRATION_v2.json BEFORE the first CAPM run
  and are never overwritten. The test set is scored only in the final stage.

QUEUE (2 runs per Kaggle session, ~4.2 h each with eager attention)
  Stage A (pilot, 1 session) : capm_s42, capm_no_module_s42      -> validation go / no-go
  Stage B (2 sessions)       : capm + capm_no_module, seeds 123 and 2024
  Stage C (2 sessions)       : capm_random_phrase, capm_no_attn, capm_no_caseloss, capm_phrase_only (s42)
  Final                      : test + external scoring, seed-aware statistics, tables, figures,
                               human-evaluation sheets, evidence summary.
  (Optional, before Final)   : run the separate COMET notebook; Final picks the scores up.

KAGGLE WORKFLOW
  Every session: attach the newest previous OUTPUT (+ the BPCC dataset + the 3 external test
  sets), run the Part 1 cell (latest version, from the external-evaluation notebook), then this
  cell. It resumes automatically and does whatever stage is next.
================================================================================
"""
assert "run_next_main_runs" in globals() and "PSCABMBart" in globals(), "Run the Part 1 cell first in this session."
assert _tf.__version__ == ENV_PIN["transformers"], "transformers version differs from the pinned one."

from collections import Counter

# ==============================================================================
# 0. SETTINGS (edit here only)
# ==============================================================================
V2 = {
    "ext_dir": "/kaggle/input/datasets/nagalaxmipanja/3-external-test-datasets",
    "ext_files": {"FLORES200_devtest": "FLORES200_en_te_devtest_UTF8.csv",
                  "IN22_Gen": "IN22_Gen_en_te.csv",
                  "IN22_Conv": "IN22_Conv_EN_TE.xlsx"},
    "ext_cols": {},                     # manual override, e.g. {"IN22_Conv": ("English", "Telugu")}
    "require_external": True,           # refuse to train if the external sets are not attached
    "case_w": 1.0,                      # case-marker tokens get weight 1 + case_w = 2.0 in the CE
    "attn_w": 0.05,                     # weight of the guided cross-attention loss
    "attn_layers": [8, 9, 10],          # 0-based decoder layers (mBART-50 has 12) -- fixed a priori
    "phrase_max_words": 6,              # max words of the object NP kept to the left of its head
    "runs_per_session": 2,
    "default_run_hours": 4.4,
    "save_models_for": ["capm_s42"],
    "human_eval_n": 300,
    "boot": 2000,
    "force_continue_after_nogo": False,  # set True only if you consciously override a NO-GO
}

# Pre-registered decision rules (written once to PREREGISTRATION_v2.json, never overwritten).
PREREG = {
    "pilot_gate (validation, seed 42)": {
        "rule": "GO iff delta_caseF1(capm - baseline) >= 0.5 AND delta_chrF++(capm - baseline) >= -0.2",
        "min_delta_casef1": 0.5, "min_delta_chrf": -0.2},
    "final_claims (test set, all seeds)": {
        "C1_case_gain": "capm - baseline caseF1 > 0 in every seed AND seed-aware 95% interval > 0",
        "C2_module_needed": "capm - capm_no_module caseF1 seed-aware 95% interval > 0",
        "C3_no_quality_loss": "capm - baseline chrF++ seed-aware 95% interval lower bound > -0.3 (non-inferiority)",
        "C4_targeting_matters": "capm_s42 - capm_random_phrase_s42 caseF1 > 0 with paired p_ar < 0.05",
        "C5_external": "capm_s42 >= baseline_s42 caseF1 on at least 2 of 3 external sets",
    },
    "hyperparameters_fixed_a_priori": {k: V2[k] for k in ("case_w", "attn_w", "attn_layers", "phrase_max_words")},
}

CONDITIONS.update({
    # full method
    "capm":               {"use_pscab": True,  "gate": "learned", "gamma": "learned", "mask": "phrase",
                           "case_w": V2["case_w"], "attn_w": V2["attn_w"]},
    # objective only, no module  -> is the module needed at all?
    "capm_no_module":     {"use_pscab": False, "gate": "learned", "gamma": "learned", "mask": "phrase",
                           "case_w": V2["case_w"], "attn_w": V2["attn_w"]},
    # module on random tokens (same count), same losses -> does phrase TARGETING of the module matter?
    "capm_random_phrase": {"use_pscab": True,  "gate": "learned", "gamma": "learned", "mask": "phrase_random",
                           "case_w": V2["case_w"], "attn_w": V2["attn_w"]},
    "capm_no_attn":       {"use_pscab": True,  "gate": "learned", "gamma": "learned", "mask": "phrase",
                           "case_w": V2["case_w"], "attn_w": 0.0},
    "capm_no_caseloss":   {"use_pscab": True,  "gate": "learned", "gamma": "learned", "mask": "phrase",
                           "case_w": 0.0, "attn_w": V2["attn_w"]},
    # = PS-CAB with the phrase mask, no new losses
    "capm_phrase_only":   {"use_pscab": True,  "gate": "learned", "gamma": "learned", "mask": "phrase",
                           "case_w": 0.0, "attn_w": 0.0},
})
V2_CONDS = ["capm", "capm_no_module", "capm_random_phrase", "capm_no_attn", "capm_no_caseloss", "capm_phrase_only"]
SEEDS = CFG["experiment"]["seeds"]
STAGE_A = [f"capm_s{SEEDS[0]}", f"capm_no_module_s{SEEDS[0]}"]
STAGE_B = [f"{c}_s{sd}" for sd in SEEDS[1:] for c in ("capm", "capm_no_module")]
STAGE_C = [f"{c}_s{SEEDS[0]}" for c in ("capm_random_phrase", "capm_no_attn", "capm_no_caseloss", "capm_phrase_only")]
V2_QUEUE = STAGE_A + STAGE_B + STAGE_C
DECISION_PATH = f"{DIRS['analysis']}/v2_pilot_decision.json"
PREREG_PATH = f"{R}/PREREGISTRATION_v2.json"


def write_preregistration():
    old = jload(PREREG_PATH)
    if old is None:
        jdump({"written": time.strftime("%Y-%m-%d %H:%M:%S"), **PREREG}, PREREG_PATH)
        print(f"[prereg] decision rules written to {PREREG_PATH} (before any CAPM run).")
        return
    cur = {k: v for k, v in old.items() if k != "written"}
    if json.loads(json.dumps(cur, default=str)) != json.loads(json.dumps(PREREG, default=str)):
        print("[prereg] WARNING: current PREREG / hyper-parameters differ from the file written on "
              f"{old.get('written')}. The FILE is authoritative; report any change as a deviation.")


# ==============================================================================
# 1. TELUGU CASE-MARKER / POSTPOSITION LEXICON
# ==============================================================================
CASE_LEXICON = {
    "DAT":     ["కి", "కు"],                    # to, for (dative)
    "LOC":     ["లో", "లోని"],                  # in, at
    "INTO":    ["లోకి"],                        # into
    "COM":     ["తో", "తోపాటు"],                # with (comitative / instrumental)
    "ABL":     ["నుండి", "నుంచి", "నుండీ"],      # from, since
    "SUPER":   ["పై", "పైన", "మీద"],            # on, over, above
    "BEN":     ["కోసం", "కొరకు"],               # for (benefactive)
    "TERM":    ["వరకు", "దాకా"],                # until, up to
    "MEANS":   ["ద్వారా"],                      # through, by means of
    "ABOUT":   ["గురించి"],                     # about
    "GEN":     ["యొక్క"],                       # of
    "CAUSE":   ["వల్ల", "వలన", "కారణంగా"],      # because of, due to
    "AFTER":   ["తర్వాత", "తరువాత", "అనంతరం"],  # after
    "BEFORE":  ["ముందు"],                       # before, in front of
    "BEHIND":  ["వెనుక", "వెనక"],               # behind
    "UNDER":   ["కింద", "క్రింద"],              # under, below
    "NEAR":    ["దగ్గర", "వద్ద"],               # near, at
    "BETWEEN": ["మధ్య"],                        # between, among
    "AROUND":  ["చుట్టూ"],                      # around
    "TOWARD":  ["వైపు"],                        # towards
    "INSIDE":  ["లోపల"],                        # inside
    "OUTSIDE": ["బయట"],                         # outside
    "WITHOUT": ["లేకుండా"],                     # without
    "AGAINST": ["వ్యతిరేకంగా"],                 # against
}
CASE_CLASSES = list(CASE_LEXICON)
CASE_BOUND = {"కి", "కు", "లో", "లోని", "లోకి", "తో", "తోపాటు", "పై"}   # suffixes: need a stem in front
CASE_STOP = {"ఎందుకు", "అందుకు", "ఇందుకు"}                              # "why", "therefore": not case
_CASE_FORMS = sorted(((f, c) for c, fs in CASE_LEXICON.items() for f in fs), key=lambda x: -len(x[0]))
_TE_WORD = re.compile(r"[\u0C00-\u0C7F\u200C\u200D0-9A-Za-z]+")


def case_spans(te):
    """[(char_start, char_end, class)] of case markers / postpositions in a Telugu string (longest match)."""
    s = str(te)
    out = []
    for m in _TE_WORD.finditer(s):
        w = m.group(0).rstrip("\u200c\u200d")
        if w in CASE_STOP:
            continue
        end = m.start() + len(w)
        for f, c in _CASE_FORMS:
            if w.endswith(f) and (len(w) > len(f) or f not in CASE_BOUND):
                out.append((end - len(f), end, c))
                break
    return out


def case_counts(te):
    return Counter(c for _, _, c in case_spans(te))


class CaseF1Metric:
    """Corpus micro-F1 over case-marker classes; per-sentence stats [tp, n_hyp, n_ref] so that bootstrap
    resampling is an exact corpus-level recomputation (same interface as StatMetric / paired_test)."""
    name = "casef1"

    def stats(self, hyps, refs, seg_scores=None):
        rows = []
        for h, r in zip(hyps, refs):
            ch, cr = case_counts(h), case_counts(r)
            rows.append((sum((ch & cr).values()), sum(ch.values()), sum(cr.values())))
        return np.asarray(rows, dtype=np.float64)

    def score(self, stat_sum, n):
        d = stat_sum[1] + stat_sum[2]
        return float(200.0 * stat_sum[0] / d) if d > 0 else 0.0


CASEF1 = CaseF1Metric()


def case_f1(hyps, refs):
    return round(CASEF1.score(CASEF1.stats(hyps, refs).sum(0), len(refs)), 4)


def case_f1_per_class(hyps, refs):
    tp, nh, nr = Counter(), Counter(), Counter()
    for h, r in zip(hyps, refs):
        ch, cr = case_counts(h), case_counts(r)
        tp.update(ch & cr); nh.update(ch); nr.update(cr)
    return {c: {"tp": tp[c], "n_hyp": nh[c], "n_ref": nr[c],
                "f1": (200.0 * tp[c] / (nh[c] + nr[c])) if (nh[c] + nr[c]) else float("nan")} for c in CASE_CLASSES}


def metrics_v2(hyps, refs):
    m = compute_metrics(hyps, refs)
    m["casef1"] = case_f1(hyps, refs)
    return m


# ==============================================================================
# 2. SOURCE PHRASE MASKS (preposition + object NP) and TARGET CASE MASKS
# ==============================================================================
_NLP_DEP = None
_PHRASE_CACHE = {}


def get_nlp_dep():
    global _NLP_DEP
    if _NLP_DEP is None:
        _NLP_DEP = False
        if _SPACY_OK:
            try:
                _NLP_DEP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            except Exception as e:
                print(f"[phrase] spaCy parser unavailable ({e}); fallback = preposition + next 3 words.")
    return _NLP_DEP


_PHRASE_CACHE_FILE = f"{DIRS['analysis']}/v2_phrase_cache.pkl"
_PHRASE_CACHE_LOADED = False


def _load_phrase_cache():
    global _PHRASE_CACHE_LOADED
    if _PHRASE_CACHE_LOADED:
        return
    _PHRASE_CACHE_LOADED = True
    if os.path.exists(_PHRASE_CACHE_FILE):
        import pickle
        try:
            with open(_PHRASE_CACHE_FILE, "rb") as f:
                c = pickle.load(f)
            if c.get("phrase_max_words") == V2["phrase_max_words"]:
                _PHRASE_CACHE.update(c["spans"])
                print(f"[phrase] {len(c['spans'])} cached parses loaded")
        except Exception as e:
            print(f"[phrase] cache unreadable ({e}); re-parsing")


def _save_phrase_cache():
    import pickle
    with open(_PHRASE_CACHE_FILE, "wb") as f:
        pickle.dump({"phrase_max_words": V2["phrase_max_words"], "spans": _PHRASE_CACHE}, f)


def ensure_phrase_spans(sents):
    """Character spans of every token inside a preposition phrase: the ADP itself + for each pobj/pcomp
    child the words from its left edge to the head (at most phrase_max_words), i.e. where Telugu puts
    the case marker. Same ADP definition as the Part 1 'true' mask."""
    _load_phrase_cache()
    todo = [s for s in dict.fromkeys(sents) if s not in _PHRASE_CACHE]
    if not todo:
        return
    K = V2["phrase_max_words"]
    nlp = get_nlp_dep()
    if nlp:
        for s, doc in zip(todo, nlp.pipe(todo, batch_size=256)):
            keep = set()
            for t in doc:
                if t.pos_ != "ADP":
                    continue
                keep.add(t.i)
                for ch in t.children:
                    if ch.dep_ in ("pobj", "pcomp"):
                        lo = max(ch.left_edge.i, ch.i - (K - 1))
                        keep.update(range(lo, ch.i + 1))
            _PHRASE_CACHE[s] = [(doc[i].idx, doc[i].idx + len(doc[i].text)) for i in sorted(keep)]
    else:
        ensure_spans(todo)
        for s in todo:
            words = [(m.start(), m.end()) for m in re.finditer(r"\S+", s)]
            keep = set()
            for ps, pe, _ in _SPAN_CACHE[s]:
                for k, (ws, we) in enumerate(words):
                    if ws <= ps < we:
                        keep.update(range(k, min(k + 4, len(words))))
            _PHRASE_CACHE[s] = [words[k] for k in sorted(keep)]
    if len(todo) >= 500:
        _save_phrase_cache()


if "_compute_masks_v1" not in globals():
    _compute_masks_v1 = compute_masks


def compute_masks(sents, tokenizer, mode):
    """Extends the Part 1 function with 'phrase' and 'phrase_random' (same token count as 'phrase',
    content tokens only, fixed per sentence). All other modes are delegated unchanged."""
    if mode not in ("phrase", "phrase_random"):
        return _compute_masks_v1(sents, tokenizer, mode)
    L = CFG["model"]["max_source_length"]
    key = (hashlib.md5("\n".join(sents).encode("utf-8")).hexdigest(), mode)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    ensure_phrase_spans(sents)
    out = []
    for i in range(0, len(sents), 1000):
        part = sents[i:i + 1000]
        enc = tokenizer(part, return_offsets_mapping=True, truncation=True, max_length=L, padding="max_length")
        for sent, offs in zip(part, enc["offset_mapping"]):
            content = [k for k, (s, e) in enumerate(offs) if s != e]
            ph = np.zeros(L, dtype=np.uint8)
            for k in content:
                s, e = offs[k]
                for ps, pe in _PHRASE_CACHE[sent]:
                    if s < pe and e > ps:
                        ph[k] = 1
                        break
            if mode == "phrase":
                m = ph
            else:
                m = np.zeros(L, dtype=np.uint8)
                kk = min(int(ph.sum()), len(content))
                if kk:
                    h = int(hashlib.md5(f"{CFG['experiment']['mask_seed']}|phrase|{sent}".encode()).hexdigest()[:8], 16)
                    m[np.random.default_rng(h).choice(content, size=kk, replace=False)] = 1
            out.append(m)
    _MASK_CACHE[key] = out
    return out


_CASE_MASK_CACHE = {}


def compute_case_masks(tes, tokenizer):
    """1 for target sub-word tokens that overlap a case-marker span (aligned with the label sequence)."""
    T = CFG["model"]["max_target_length"]
    key = hashlib.md5("\n".join(tes).encode("utf-8")).hexdigest()
    if key in _CASE_MASK_CACHE:
        return _CASE_MASK_CACHE[key]
    out = []
    for i in range(0, len(tes), 1000):
        part = tes[i:i + 1000]
        enc = tokenizer(text_target=part, return_offsets_mapping=True, truncation=True, max_length=T,
                        padding="max_length")
        for te, offs in zip(part, enc["offset_mapping"]):
            spans = case_spans(te)
            m = np.zeros(T, dtype=np.uint8)
            if spans:
                for k, (s, e) in enumerate(offs):
                    if s == e:
                        continue
                    for cs, ce, _ in spans:
                        if s < ce and e > cs:
                            m[k] = 1
                            break
            out.append(m)
    _CASE_MASK_CACHE[key] = out
    return out


class PairDatasetV2(Dataset):
    def __init__(self, df, tok, spec):
        self.en, self.te = df["en"].tolist(), df["te"].tolist()
        self.mask = compute_masks(self.en, tok, spec["mask"])
        self.true = compute_masks(self.en, tok, "true")
        self.phrase = compute_masks(self.en, tok, "phrase")       # attention target is ALWAYS the true phrase
        self.case = compute_case_masks(self.te, tok)

    def __len__(self):
        return len(self.en)

    def __getitem__(self, i):
        return self.en[i], self.te[i], self.mask[i], self.true[i], self.phrase[i], self.case[i]


def collate_v2(batch, tok):
    en, te, m, t, ph, cs = zip(*batch)
    out = collate(list(zip(en, te, m, t)), tok)
    L, T = out["input_ids"].shape[1], out["labels"].shape[1]
    out["phrase_mask"] = torch.from_numpy(np.stack(ph)[:, :L].astype(np.int64))
    out["case_mask"] = torch.from_numpy(np.stack(cs)[:, :T].astype(np.int64))
    return out


def v2_selftest(n_show=3):
    """Prints how sources / targets are marked and writes a precision-check sheet for the lexicon."""
    path = f"{DIRS['analysis']}/v2_selftest.json"
    tok = get_tokenizer()
    train, val, _ = load_splits()
    sub = val.sample(n=400, random_state=7).reset_index(drop=True)
    ph = compute_masks(sub["en"].tolist(), tok, "phrase")
    tr = compute_masks(sub["en"].tolist(), tok, "true")
    cs = compute_case_masks(sub["te"].tolist(), tok)
    for k in range(n_show):
        s_ids = tok(sub["en"][k], truncation=True, max_length=CFG["model"]["max_source_length"])["input_ids"]
        t_ids = tok(text_target=sub["te"][k], truncation=True, max_length=CFG["model"]["max_target_length"])["input_ids"]
        st, tt = tok.convert_ids_to_tokens(s_ids), tok.convert_ids_to_tokens(t_ids)
        print(f"\n[selftest] EN: {sub['en'][k]}")
        print("  phrase tokens :", [st[j] for j in range(len(st)) if ph[k][j]])
        print("  prep tokens   :", [st[j] for j in range(len(st)) if tr[k][j]])
        print(f"  TE: {sub['te'][k]}")
        print("  case tokens   :", [tt[j] for j in range(len(tt)) if cs[k][j]], "| spans:",
              [(sub['te'][k][a:b], c) for a, b, c in case_spans(sub['te'][k])])
    rep = {"val_sample": len(sub),
           "mean_phrase_tokens_per_sentence": round(float(np.mean([m.sum() for m in ph])), 2),
           "mean_prep_tokens_per_sentence": round(float(np.mean([m.sum() for m in tr])), 2),
           "mean_case_tokens_per_target": round(float(np.mean([m.sum() for m in cs])), 2),
           "targets_with_case_marker_%": round(100 * float(np.mean([m.sum() > 0 for m in cs])), 2)}
    counts = Counter()
    for t in train["te"]:
        counts.update(case_counts(t))
    rep["train_case_class_counts"] = dict(counts.most_common())
    jdump(rep, path)
    print(f"\n[selftest] {rep}")
    # precision-check sheet: a native speaker marks 'correct' (1/0) for 200 random matches
    rng = np.random.default_rng(11)
    rows = []
    for i in rng.choice(len(train), size=min(600, len(train)), replace=False):
        te = train["te"].iloc[i]
        for a, b, c in case_spans(te):
            w0 = te.rfind(" ", 0, a) + 1
            w1 = te.find(" ", b); w1 = len(te) if w1 < 0 else w1
            rows.append({"class": c, "word": te[w0:w1], "marker": te[a:b], "english": train["en"].iloc[i],
                         "telugu": te, "correct_1_0": ""})
    if rows:
        df = pd.DataFrame(rows).sample(n=min(200, len(rows)), random_state=0)
        df.to_csv(f"{DIRS['tables']}/V2_case_lexicon_precision_check.csv", index=False)
        print(f"[selftest] precision-check sheet -> tables/V2_case_lexicon_precision_check.csv ({len(df)} rows)")
    return rep


# ==============================================================================
# 3. TRAINING (identical schedule to Part 1, plus the two auxiliary terms)
# ==============================================================================
def v2_losses(model, batch, spec):
    """Returns (total, ce_weighted, attn_loss). CE uses label smoothing exactly as Part 1."""
    o = CFG["optim"]
    need_attn = spec.get("attn_w", 0.0) > 0
    labels = batch["labels"]
    enc = model.encode(batch["input_ids"], batch["attention_mask"], batch["mask"])
    out = model.backbone(attention_mask=batch["attention_mask"], encoder_outputs=enc,
                         decoder_input_ids=shift_right_mbart(labels, model.pad_id),
                         output_attentions=need_attn, use_cache=False, return_dict=True)
    logits = out.logits
    tok_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
                             label_smoothing=o["label_smoothing"], reduction="none").view_as(labels).float()
    valid = labels.ne(-100)
    w = 1.0 + spec.get("case_w", 0.0) * batch["case_mask"].float()
    ce = (tok_ce * w)[valid].sum() / valid.sum().clamp_min(1)
    la = torch.zeros((), device=ce.device)
    if need_attn:
        A = torch.stack([out.cross_attentions[l] for l in V2["attn_layers"]]).float().mean(dim=(0, 2))  # (B,T,S)
        src = batch["phrase_mask"].bool() & batch["attention_mask"].bool()
        mass = (A * src.unsqueeze(1).float()).sum(-1)                                                    # (B,T)
        sel = batch["case_mask"].bool() & valid & src.any(-1, keepdim=True)
        if sel.any():
            la = -torch.log(mass[sel].clamp_min(1e-6)).mean()
    return ce + spec.get("attn_w", 0.0) * la, ce.detach(), la.detach()


def train_model_v2(cond, seed, train_df, val_df, run_id):
    o = CFG["optim"]
    spec = CONDITIONS[cond]
    print(f"\n{'='*72}\nTRAIN (v2) {run_id}: {spec}\n{'='*72}")
    set_all_seeds(seed)
    tok = get_tokenizer()
    g = torch.Generator(); g.manual_seed(seed)
    tl = DataLoader(PairDatasetV2(train_df, tok, spec), batch_size=o["train_batch_size"], shuffle=True,
                    generator=g, collate_fn=lambda b: collate_v2(b, tok), drop_last=False)
    vl = DataLoader(PairDataset(val_df, tok, spec["mask"]), batch_size=o["eval_batch_size"], shuffle=False,
                    collate_fn=lambda b: collate(b, tok))
    model = PSCABMBart(cond, attn_impl="eager" if spec.get("attn_w", 0) > 0 else None).to(DEVICE)
    init_gate = None
    if model.use_pscab and spec["gate"] == "learned":
        init_gate = {k: v.detach().cpu().clone() for k, v in model.pscab.W_g.state_dict().items()}
    base = [p for n, p in model.named_parameters() if not n.startswith("pscab.")]
    extra = [p for n, p in model.named_parameters() if n.startswith("pscab.")]
    groups = [{"params": base, "lr": o["lr"]}]
    if extra:
        groups.append({"params": extra, "lr": o["pscab_lr"], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, weight_decay=o["weight_decay"])
    total, warm = o["max_steps"], max(1, int(o["warmup_frac"] * o["max_steps"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    scaler = make_scaler()
    params = base + extra

    log, best, best_step, best_state, bad = [], float("inf"), 0, None, 0
    step, micro, nonfinite, stop = 0, 0, 0, "max_steps_reached"
    acc = {"total": 0.0, "ce": 0.0, "attn": 0.0, "n": 0}
    t0 = time.time()
    model.train()
    opt.zero_grad(set_to_none=True)
    while step < total and stop == "max_steps_reached":
        for batch in tl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with amp_ctx():
                loss, ce, la = v2_losses(model, batch, spec)
            if torch.isfinite(loss):
                scaler.scale(loss / o["grad_accum"]).backward()
                acc["total"] += loss.item(); acc["ce"] += ce.item(); acc["attn"] += la.item(); acc["n"] += 1
            else:
                nonfinite += 1
            micro += 1
            if micro % o["grad_accum"] != 0:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, o["clip"])
            scaler.step(opt); scaler.update(); sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % o["eval_every"] == 0 or step == total:
                v = val_nll(model, vl)                       # UNWEIGHTED NLL -> same selection rule as Part 1
                n = max(acc["n"], 1)
                row = {"step": step, "epoch": round(step * o["grad_accum"] * o["train_batch_size"] / len(train_df), 3),
                       "train_loss_total": acc["total"] / n, "train_ce_weighted": acc["ce"] / n,
                       "train_attn_loss": acc["attn"] / n, "val_nll": v, "lr": sched.get_last_lr()[0],
                       "elapsed_h": (time.time() - t0) / 3600}
                if model.use_pscab and spec["gamma"] == "learned":
                    row["gamma"] = float(model.pscab.gamma.item())
                log.append(row)
                acc = {"total": 0.0, "ce": 0.0, "attn": 0.0, "n": 0}
                flag = ""
                if v < best - o["min_delta"]:
                    best, best_step, bad = v, step, 0
                    best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
                    flag = "  <- best"
                else:
                    bad += 1
                print(f"[{run_id}] step {step}/{total} ce(w) {row['train_ce_weighted']:.4f} attn "
                      f"{row['train_attn_loss']:.4f} val_nll {v:.4f} lr {row['lr']:.2e} {row['elapsed_h']:.2f}h{flag}")
                if bad >= o["patience_evals"]:
                    stop = "early_stopping"
                    break
            if step >= total:
                break
    train_h = (time.time() - t0) / 3600
    model.load_state_dict(best_state)
    del opt, sched, scaler, best_state
    free_gpu()
    pd.DataFrame(log).to_csv(f"{DIRS['logs']}/{run_id}_curve.csv", index=False)
    meta = {"steps_trained": step, "best_step": best_step, "best_val_nll": best, "stop_reason": stop,
            "train_hours": train_h, "sec_per_step": train_h * 3600 / max(step, 1),
            "nonfinite_loss_batches": nonfinite, **model.param_report()}
    if model.use_pscab and spec["gamma"] == "learned":
        meta["gamma_at_best"] = float(model.pscab.gamma.item())
    print(f"[{run_id}] done: {step} steps, best step {best_step} (val_nll {best:.4f}), stop={stop}, {train_h:.2f}h")
    return model, tok, meta, init_gate


# ==============================================================================
# 4. EXTERNAL SETS + DIAGNOSTICS
# ==============================================================================
_EXT_CACHE = None


def load_external_v2():
    global _EXT_CACHE
    if _EXT_CACHE is not None:
        return _EXT_CACHE
    out = {}
    for name, fn in V2["ext_files"].items():
        path = f"{V2['ext_dir']}/{fn}"
        if not os.path.exists(path):
            print(f"[ext] {name}: NOT FOUND at {path}")
            continue
        if path.lower().endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(path, sheet_name=None, dtype=str)
            df = max(sheets.values(), key=len)
        else:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        df.columns = [str(c) for c in df.columns]
        df = df.fillna("").astype(str)
        if name in V2["ext_cols"]:
            en_col, te_col = V2["ext_cols"][name]
        else:
            te_sc = {c: df[c].head(500).apply(lambda s: len(TELUGU_RE.findall(s))).mean() for c in df.columns}
            te_col = max(te_sc, key=te_sc.get)
            en_sc = {c: df[c].head(500).apply(lambda s: len(LATIN_RE.findall(s))).mean() for c in df.columns if c != te_col}
            en_col = max(en_sc, key=en_sc.get)
        d = df[[en_col, te_col]].rename(columns={en_col: "en", te_col: "te"})
        d["en"], d["te"] = d["en"].apply(normalize_text), d["te"].apply(normalize_text)
        d = d[(d["en"] != "") & (d["te"] != "")].reset_index(drop=True)
        print(f"[ext] {name}: {len(d)} pairs (EN='{en_col}', TE='{te_col}')")
        out[name] = d
    _EXT_CACHE = out
    return out


@torch.no_grad()
def case_attention_diagnostic(model, tok, df, cond_mask, n=300):
    """Teacher-forced cross-attention mass that CASE-MARKER target tokens put on the source preposition
    phrases (supervised layers and last layer), normalised by the uniform share. Needs an eager model."""
    sents = df["en"].tolist()
    ensure_phrase_spans(sents)
    ph_all = compute_masks(sents, tok, "phrase")
    cs_all = compute_case_masks(df["te"].tolist(), tok)
    cand = [i for i in range(len(df)) if ph_all[i].sum() and cs_all[i].sum()]
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(cand, size=min(n, len(cand)), replace=False))
    applied = compute_masks([sents[i] for i in idx], tok, cond_mask)
    sup, last, ratio = [], [], []
    model.eval()
    for j, i in enumerate(idx):
        src = tok(sents[i], return_tensors="pt", truncation=True, max_length=CFG["model"]["max_source_length"]).to(DEVICE)
        tgt = tok(text_target=df["te"].iloc[i], return_tensors="pt", truncation=True,
                  max_length=CFG["model"]["max_target_length"]).to(DEVICE)
        S, T = src["input_ids"].shape[1], tgt["input_ids"].shape[1]
        pm = torch.from_numpy(applied[j][:S].astype(np.int64)).unsqueeze(0).to(DEVICE)
        sp = torch.from_numpy(ph_all[i][:S].astype(bool)).to(DEVICE)
        ct = torch.from_numpy(cs_all[i][:T].astype(bool)).to(DEVICE)
        if sp.sum() == 0 or ct.sum() == 0:
            continue
        enc = model.encode(src["input_ids"], src["attention_mask"], pm)
        out = model.backbone(attention_mask=src["attention_mask"], encoder_outputs=enc,
                             decoder_input_ids=shift_right_mbart(tgt["input_ids"], model.pad_id),
                             output_attentions=True, use_cache=False, return_dict=True)
        A = torch.stack([a[0].float().mean(0) for a in out.cross_attentions])        # (layers, T, S)
        m_sup = A[V2["attn_layers"]].mean(0)[ct][:, sp].sum(-1).mean()
        m_last = A[-1][ct][:, sp].sum(-1).mean()
        sup.append(float(m_sup)); last.append(float(m_last)); ratio.append(float(m_sup) / (float(sp.sum()) / S))
    return {"n_sentences": len(sup), "case_to_phrase_mass_supervised_layers": float(np.mean(sup)) if sup else None,
            "case_to_phrase_mass_last_layer": float(np.mean(last)) if last else None,
            "case_to_phrase_mass_over_uniform": float(np.mean(ratio)) if ratio else None}


def _eager_copy(model, cond):
    if getattr(model.backbone.config, "_attn_implementation", None) == "eager":
        return model, False
    e = PSCABMBart(cond, attn_impl="eager")
    e.load_state_dict(model.state_dict())
    return e.to(DEVICE).eval(), True


def fp16_roundtrip_(model):
    """External decodes of baseline_s42 / pscab_s42 were made from fp16-saved weights; round-trip ours the
    same way so the external comparison is symmetric."""
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.half().float()


# ==============================================================================
# 5. RUN EXECUTOR + QUEUE
# ==============================================================================
def v2_run_hours():
    hrs = [r["wall_hours_total"] for r in all_records("v2") if r.get("wall_hours_total")]
    return max(hrs) * 1.1 if hrs else V2["default_run_hours"]


def execute_v2_run(run_id):
    if run_done(run_id):
        print(f"[skip] {run_id} already complete.")
        return jload(run_record_path(run_id))
    need = v2_run_hours()
    if session_hours_left() < need:
        print(f"[time] {run_id} needs ~{need:.1f}h, {session_hours_left():.1f}h left -> NOT started. Next session.")
        return None
    ext = load_external_v2()
    if V2["require_external"] and len(ext) < len(V2["ext_files"]):
        raise RuntimeError("External test sets missing: attach the '3-external-test-datasets' input "
                           "(or set V2['require_external']=False and accept missing external results).")
    cond, seed, _ = parse_run_id(run_id)
    spec = CONDITIONS[cond]
    t_all = time.time()
    train, val, test = load_splits()
    model, tok, meta, init_gate = train_model_v2(cond, seed, train, val, run_id)
    rec = {"run_id": run_id, "kind": "v2", "condition": cond, "seed": seed, "spec": spec, "status": "complete",
           "env": environment_info(), "v2_settings": {k: V2[k] for k in ("case_w", "attn_w", "attn_layers",
                                                                       "phrase_max_words")}, **meta}
    rec["generation"] = generation_record(model, tok)
    vp, vinfo = generate(model, tok, val, spec["mask"])
    save_preds(run_id, "val", val, vp)
    rec["val_metrics"] = metrics_v2(vp, val["te"].tolist())
    rec["inference_sentences_per_sec"] = vinfo["sentences_per_sec"]
    print(f"[{run_id}] VAL {rec['val_metrics']}")
    tp, _ = generate(model, tok, test, spec["mask"])
    save_preds(run_id, "test", test, tp)                                    # blind until the final stage
    if model.use_pscab:
        rec["diagnostics"] = pscab_diagnostics(model, tok, val, cond, init_gate, run_id)
    if seed == SEEDS[0]:
        em, made = _eager_copy(model, cond)
        rec["case_attention"] = case_attention_diagnostic(em, tok, val, spec["mask"])
        if made:
            del em; free_gpu()
        rec["attention"] = attention_mass(model, tok, val, cond, run_id)
        rec["robustness_val"] = robustness(model, tok, val, cond)
        print(f"[{run_id}] case attention: {rec['case_attention']}")
    if run_id in V2["save_models_for"]:
        save_model_bundle(model, tok, run_id, rec)
    fp16_roundtrip_(model)
    for name, edf in ext.items():
        ep, _ = generate(model, tok, edf, spec["mask"])
        save_preds(run_id, f"extfull_{name}", edf, ep)
    del model
    free_gpu()
    rec["wall_hours_total"] = (time.time() - t_all) / 3600
    jdump(rec, run_record_path(run_id))
    print(f"[{run_id}] COMPLETE in {rec['wall_hours_total']:.2f}h")
    return rec


# ==============================================================================
# 6. STATISTICS HELPERS
# ==============================================================================
def _val_refs():
    return load_splits()[1]["te"].tolist()


def _metric_objs():
    return {"chrf++": StatMetric("chrf++"), "bleu": StatMetric("bleu"), "casef1": CASEF1}


def _comet_seg(run_id, split):
    p = f"{DIRS['analysis']}/comet__{run_id}__{split}.npy"
    return np.load(p) if os.path.exists(p) else None


def _stats(metric_name, run_id, split, refs, cache):
    key = (metric_name, run_id, split)
    if key not in cache:
        if metric_name == "comet":
            seg = _comet_seg(run_id, split)
            cache[key] = None if seg is None else StatMetric("comet").stats(None, None, seg)
        else:
            preds = load_preds(run_id, split)
            cache[key] = None if preds is None else _metric_objs()[metric_name].stats(preds, refs)
    return cache[key]


def _metric(metric_name):
    return StatMetric("comet") if metric_name == "comet" else _metric_objs()[metric_name]


def seed_aware_bootstrap(pairs, metric, n_boot=None, seed=777):
    """pairs: list of (stats_A, stats_B), one per seed, same sentences. Resample seeds, then sentences.
    Returns mean delta and 95% interval (a seed-aware uncertainty characterisation, only 3 seeds)."""
    n_boot = n_boot or V2["boot"]
    rng = np.random.default_rng(seed)
    N = len(pairs[0][0])
    deltas = [metric.score(a.sum(0), N) - metric.score(b.sum(0), N) for a, b in pairs]
    boots = np.empty(n_boot)
    for k in range(n_boot):
        ss = rng.integers(0, len(pairs), len(pairs))
        idx = rng.integers(0, N, N)
        boots[k] = np.mean([metric.score(pairs[s][0][idx].sum(0), N) - metric.score(pairs[s][1][idx].sum(0), N)
                            for s in ss])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"per_seed": [round(d, 4) for d in deltas], "mean": round(float(np.mean(deltas)), 4),
            "seed_aware_ci_low": round(float(lo), 4), "seed_aware_ci_high": round(float(hi), 4),
            "better_in": f"{sum(d > 0 for d in deltas)}/{len(deltas)}"}


# ==============================================================================
# 7. PILOT GATE (VALIDATION ONLY)
# ==============================================================================
def v2_pilot_decision():
    s0 = SEEDS[0]
    need = [f"baseline_s{s0}", f"pscab_s{s0}"] + STAGE_A
    if not all(run_done(r) for r in need):
        print(f"[pilot] waiting for {[r for r in need if not run_done(r)]}")
        return None
    refs = _val_refs()
    cache, rows, tests = {}, [], []
    for rid in need:
        p = load_preds(rid, "val")
        m = metrics_v2(p, refs)
        rows.append({"run": rid, "BLEU": m["bleu"], "chrF++": m["chrf++"], "TER": m["ter"], "caseF1": m["casef1"]})
    for a, b in [(f"capm_s{s0}", f"baseline_s{s0}"), (f"capm_s{s0}", f"capm_no_module_s{s0}"),
                 (f"capm_no_module_s{s0}", f"baseline_s{s0}"), (f"capm_s{s0}", f"pscab_s{s0}")]:
        for mn in ("casef1", "chrf++", "bleu"):
            r = paired_test(_stats(mn, a, "val", refs, cache), _stats(mn, b, "val", refs, cache), _metric(mn))
            tests.append({"metric": mn, "A": a, "B": b, **r})
    t1, t2 = pd.DataFrame(rows), pd.DataFrame(tests)
    save_table(t1, "V2_P1_pilot_validation"); save_table(t2, "V2_P2_pilot_validation_paired")
    print("\n[pilot] validation results\n" + t1.to_string(index=False) + "\n\n" + t2.to_string(index=False))
    g = (jload(PREREG_PATH) or PREREG)["pilot_gate (validation, seed 42)"]
    d_case = float(t2.query(f"metric=='casef1' and A=='capm_s{s0}' and B=='baseline_s{s0}'")["delta"].iloc[0])
    d_chrf = float(t2.query(f"metric=='chrf++' and A=='capm_s{s0}' and B=='baseline_s{s0}'")["delta"].iloc[0])
    d_mod = float(t2.query(f"metric=='casef1' and A=='capm_s{s0}' and B=='capm_no_module_s{s0}'")["delta"].iloc[0])
    go = d_case >= g["min_delta_casef1"] and d_chrf >= g["min_delta_chrf"]
    dec = {"decided": time.strftime("%Y-%m-%d %H:%M:%S"), "GO": bool(go), "delta_caseF1_vs_baseline": d_case,
           "delta_chrF++_vs_baseline": d_chrf, "delta_caseF1_vs_no_module": d_mod, "rule": g["rule"],
           "note": ("GO: continue with stages B and C." if go else
                    "NO-GO: the pre-registered pilot criterion failed. Recommended: write the negative-result "
                    "paper (Route 3) including CAPM as a second tested hypothesis. Override only consciously "
                    "via V2['force_continue_after_nogo']=True and report it as a deviation.")}
    if go and d_mod < 0.2:
        dec["note"] += (" Caution: capm barely beats capm_no_module; the gain may come from the training "
                        "objective rather than the module -- the paper must then present the objective as the method.")
    jdump(dec, DECISION_PATH)
    print(f"\n[pilot] DECISION: {dec}")
    return dec


# ==============================================================================
# 8. FINAL EVALUATION (TEST + EXTERNAL) -- runs once every queue item is complete
# ==============================================================================
def v2_final_report():
    print(f"\n{'#'*72}\nPART 4 FINAL EVALUATION (test set + external sets)\n{'#'*72}")
    _, val, test = load_splits()
    refs = test["te"].tolist()
    cache = {}
    have_comet = _comet_seg(f"capm_s{SEEDS[0]}", "test") is not None and _comet_seg(f"baseline_s{SEEDS[0]}", "test") is not None
    mets = ["chrf++", "bleu", "casef1"] + (["comet"] if have_comet else [])
    print(f"[final] metrics: {mets}" + ("" if have_comet else "  (COMET .npy not found -> run the COMET notebook first if wanted)"))

    conds = ["baseline", "pscab"] + V2_CONDS
    runs = [f"{c}_s{s}" for c in conds for s in SEEDS if run_done(f"{c}_s{s}")]
    # ---- T1 per run
    rows = []
    for rid in runs:
        p = load_preds(rid, "test")
        if p is None:
            continue
        m = metrics_v2(p, refs)
        c, s, _ = parse_run_id(rid)
        row = {"run": rid, "condition": c, "seed": s, "BLEU": m["bleu"], "chrF++": m["chrf++"], "TER": m["ter"],
               "caseF1": m["casef1"]}
        seg = _comet_seg(rid, "test")
        if seg is not None:
            row["COMET"] = round(float(np.mean(seg)) * 100, 2)
        rows.append(row)
    t1 = pd.DataFrame(rows)
    save_table(t1, "V2_T1_test_per_run")
    # ---- T2 by condition
    agg = []
    for c, g in t1.groupby("condition", sort=False):
        r = {"condition": c, "n_seeds": len(g)}
        for col in [x for x in ("BLEU", "chrF++", "TER", "caseF1", "COMET") if x in g]:
            r[f"{col}_mean"] = round(g[col].mean(), 2)
            r[f"{col}_sd"] = round(g[col].std(ddof=1), 2) if len(g) > 1 else np.nan
        agg.append(r)
    t2 = pd.DataFrame(agg)
    save_table(t2, "V2_T2_test_by_condition")
    print("\n[V2-T2] test by condition\n" + t2.to_string(index=False))

    # ---- T3 same-seed paired tests, Holm per metric
    comps = [("capm", "baseline", SEEDS), ("capm", "capm_no_module", SEEDS), ("capm", "pscab", SEEDS),
             ("capm_no_module", "baseline", SEEDS)]
    comps += [("capm", c, SEEDS[:1]) for c in ("capm_random_phrase", "capm_no_attn", "capm_no_caseloss", "capm_phrase_only")]
    tests = []
    for mn in mets:
        for a, b, sds in comps:
            for s in sds:
                A, B = f"{a}_s{s}", f"{b}_s{s}"
                sa, sb = _stats(mn, A, "test", refs, cache), _stats(mn, B, "test", refs, cache)
                if sa is None or sb is None:
                    continue
                tests.append({"metric": mn, "A": A, "B": B, **paired_test(sa, sb, _metric(mn))})
    t3 = pd.DataFrame(tests)
    if len(t3):
        for mn in t3["metric"].unique():
            sub = t3[t3["metric"] == mn]
            h = holm({f"{r.A}|{r.B}": r.p_ar for r in sub.itertuples()})
            for i in sub.index:
                k = f"{t3.at[i, 'A']}|{t3.at[i, 'B']}"
                t3.at[i, "holm_threshold"] = h[k]["holm_threshold"]
                t3.at[i, "significant_holm"] = h[k]["significant_holm"]
    save_table(t3, "V2_T3_test_paired_tests")
    print("\n[V2-T3] paired tests\n" + t3.to_string(index=False))

    # ---- T4 seed-aware bootstrap
    sa_rows = []
    for mn in mets:
        for a, b in [("capm", "baseline"), ("capm", "capm_no_module"), ("capm", "pscab"), ("capm_no_module", "baseline")]:
            pairs = [(_stats(mn, f"{a}_s{s}", "test", refs, cache), _stats(mn, f"{b}_s{s}", "test", refs, cache)) for s in SEEDS]
            pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
            if len(pairs) >= 2:
                sa_rows.append({"metric": mn, "A": a, "B": b, "n_seeds": len(pairs), **seed_aware_bootstrap(pairs, _metric(mn))})
    t4 = pd.DataFrame(sa_rows)
    save_table(t4, "V2_T4_seed_aware_bootstrap")
    print("\n[V2-T4] seed-aware bootstrap\n" + t4.to_string(index=False))

    # ---- T5 per-class case F1 (mean over seeds)
    pc = []
    for c in ("baseline", "pscab", "capm", "capm_no_module"):
        per = [case_f1_per_class(load_preds(f"{c}_s{s}", "test"), refs) for s in SEEDS if run_done(f"{c}_s{s}")]
        if not per:
            continue
        for k in CASE_CLASSES:
            pc.append({"condition": c, "class": k, "n_ref": per[0][k]["n_ref"],
                       "f1_mean": round(float(np.nanmean([p[k]["f1"] for p in per])), 2)})
    t5 = pd.DataFrame(pc)
    if len(t5):
        t5w = t5.pivot_table(index=["class", "n_ref"], columns="condition", values="f1_mean").reset_index()
        if "capm" in t5w and "baseline" in t5w:
            t5w["capm_minus_baseline"] = (t5w["capm"] - t5w["baseline"]).round(2)
        t5w = t5w.sort_values("n_ref", ascending=False)
        save_table(t5w, "V2_T5_case_f1_per_class")
        print("\n[V2-T5] per-class case F1\n" + t5w.to_string(index=False))

    # ---- T6 by number of source prepositions
    ensure_spans(test["en"].tolist())
    npre = np.array([min(prep_count(s), 3) for s in test["en"]])
    bk = []
    for c in ("baseline", "capm", "capm_no_module"):
        for s in SEEDS:
            rid = f"{c}_s{s}"
            p = load_preds(rid, "test")
            if p is None:
                continue
            for b in range(4):
                ii = np.where(npre == b)[0]
                hh, rr = [p[i] for i in ii], [refs[i] for i in ii]
                bk.append({"condition": c, "seed": s, "n_prep": "3+" if b == 3 else str(b), "n": len(ii),
                           "chrF++": chrf_metric().corpus_score(hh, [rr]).score, "caseF1": case_f1(hh, rr)})
    t6 = pd.DataFrame(bk)
    if len(t6):
        t6 = t6.groupby(["condition", "n_prep", "n"], as_index=False)[["chrF++", "caseF1"]].mean().round(2)
        save_table(t6, "V2_T6_by_preposition_count")

    # ---- T7 external sets
    ext = load_external_v2()
    er, et = [], []
    s0 = SEEDS[0]
    for name, edf in ext.items():
        erefs = edf["te"].tolist()
        for rid in ("zeroshot_mbart50", "nllb200_600m", f"baseline_s{s0}", f"pscab_s{s0}", f"capm_s{s0}", f"capm_no_module_s{s0}"):
            p = load_preds(rid, f"extfull_{name}")
            if p is None or len(p) != len(erefs):
                continue
            m = metrics_v2(p, erefs)
            row = {"set": name, "system": rid, "n": len(erefs), "chrF++": m["chrf++"], "spBLEU": m["bleu"], "TER": m["ter"],
                   "caseF1": m["casef1"]}
            seg = _comet_seg(rid, f"extfull_{name}")
            if seg is not None:
                row["COMET"] = round(float(np.mean(seg)) * 100, 2)
            er.append(row)
        for b in (f"baseline_s{s0}", "nllb200_600m"):
            for mn in mets:
                sa = _stats(mn, f"capm_s{s0}", f"extfull_{name}", erefs, cache)
                sb = _stats(mn, b, f"extfull_{name}", erefs, cache)
                if sa is None or sb is None or len(sa) != len(sb):
                    continue
                et.append({"set": name, "metric": mn, "A": f"capm_s{s0}", "B": b, **paired_test(sa, sb, _metric(mn))})
    t7, t7b = pd.DataFrame(er), pd.DataFrame(et)
    save_table(t7, "V2_T7_external_results"); save_table(t7b, "V2_T7b_external_paired")
    if len(t7):
        print("\n[V2-T7] external\n" + t7.to_string(index=False))

    # ---- T8 diagnostics of v2 runs
    dg = []
    for r in all_records("v2"):
        d = {"run": r["run_id"], "best_step": r.get("best_step"), "stop": r.get("stop_reason"),
             "sec_per_step": round(r.get("sec_per_step", np.nan), 3), "added_params": r.get("pscab_added_parameters")}
        d.update({f"gate_{k}": v for k, v in (r.get("diagnostics") or {}).items() if not isinstance(v, (list, dict))})
        d.update({f"att_{k}": v for k, v in (r.get("case_attention") or {}).items()})
        dg.append(d)
    if dg:
        save_table(pd.DataFrame(dg), "V2_T8_diagnostics")

    v2_figures(t1, t4, t5 if len(t5) else None)
    v2_human_eval_sheets(test)
    v2_evidence_summary(t3, t4, t7)


def v2_figures(t1, t4, t5):
    apply_paper_style()
    # Fig A: per-condition test scores (dots = seeds)
    cols = [c for c in ("chrF++", "caseF1", "COMET") if c in t1 and t1[c].notna().any()]
    order = [c for c in ["baseline", "pscab", "capm_no_module", "capm", "capm_random_phrase", "capm_no_attn",
                         "capm_no_caseloss", "capm_phrase_only"] if c in set(t1["condition"])]
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 4.2))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        for i, c in enumerate(order):
            v = t1[t1["condition"] == c][col].values
            ax.bar(i, v.mean(), color="#4C72B0" if c.startswith("capm") else "#999999", alpha=.75)
            ax.scatter(np.full(len(v), i), v, color="k", s=12, zorder=3)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha="right")
        allv = t1[t1["condition"].isin(order)][col]
        ax.set_ylim(allv.min() - 0.8, allv.max() + 0.6); ax.set_title(col)
    save_fig("V2_FIG1_test_by_condition")
    # Fig B: seed-aware intervals
    if len(t4):
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(t4) + 1.2))
        lab = [f"{r.metric}: {r.A} - {r.B}" for r in t4.itertuples()]
        y = np.arange(len(t4))
        ax.errorbar(t4["mean"], y, xerr=[t4["mean"] - t4["seed_aware_ci_low"], t4["seed_aware_ci_high"] - t4["mean"]],
                    fmt="o", color="k", capsize=3)
        ax.axvline(0, color="r", lw=1, ls="--"); ax.set_yticks(y); ax.set_yticklabels(lab)
        ax.set_xlabel("difference (seed-aware 95% interval)")
        save_fig("V2_FIG2_effect_forest")
    # Fig C: per-class case F1 difference
    if t5 is not None:
        p = t5.pivot_table(index="class", columns="condition", values="f1_mean")
        n = t5.groupby("class")["n_ref"].first()
        if {"capm", "baseline"} <= set(p.columns):
            d = (p["capm"] - p["baseline"]).loc[n[n >= 30].sort_values(ascending=False).index]
            fig, ax = plt.subplots(figsize=(8, 3.8))
            ax.bar(range(len(d)), d.values, color=["#2a9d8f" if x > 0 else "#e76f51" for x in d.values])
            ax.set_xticks(range(len(d))); ax.set_xticklabels([f"{k}\n(n={n[k]})" for k in d.index], rotation=60, fontsize=7)
            ax.axhline(0, color="k", lw=.8); ax.set_ylabel("case-marker F1: CAPM - baseline")
            save_fig("V2_FIG3_case_class_delta")


def v2_human_eval_sheets(test):
    """Blinded A/B sheets for two annotators (same items, same order) + a separate key.
    Items: test sentences with >= 1 English preposition where capm_s42 and baseline_s42 differ
    (disclose this selection in the paper)."""
    s0 = SEEDS[0]
    a, b = load_preds(f"capm_s{s0}", "test"), load_preds(f"baseline_s{s0}", "test")
    if a is None or b is None:
        return
    ensure_spans(test["en"].tolist())
    cand = [i for i in range(len(test)) if prep_count(test["en"].iloc[i]) > 0 and a[i].strip() != b[i].strip()]
    rng = np.random.default_rng(2026)
    idx = sorted(rng.choice(cand, size=min(V2["human_eval_n"], len(cand)), replace=False))
    rows, key = [], []
    for k, i in enumerate(idx):
        flip = bool(rng.random() < 0.5)
        A, B = (b[i], a[i]) if flip else (a[i], b[i])
        rows.append({"item": k + 1, "english_source": test["en"].iloc[i], "telugu_reference": test["te"].iloc[i],
                     "output_A": A, "output_B": B,
                     "A_postpositions_correct (all/some/none)": "", "B_postpositions_correct (all/some/none)": "",
                     "A_adequacy (1-5)": "", "B_adequacy (1-5)": "", "preference (A/B/tie)": "", "comment": ""})
        key.append({"item": k + 1, "test_index": int(i), "A_is": "baseline" if flip else "capm"})
    sheet = pd.DataFrame(rows)
    send = f"{R}/human_eval_SEND_TO_ANNOTATORS"
    os.makedirs(send, exist_ok=True)
    for ann in (1, 2):
        sheet.to_csv(f"{send}/V2_HUMAN_EVAL_annotator{ann}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(f"{DIRS['analysis']}/V2_HUMAN_EVAL_KEY_do_not_share.csv", index=False)
    print(f"[human] {len(idx)} items -> {send}/ (send ONLY this folder; the key stays in analysis/).")


def v2_human_eval_score(path1, path2, key_path=None):
    """Call after both annotators returned their sheets: Cohen's kappa + CAPM win rates."""
    key_path = key_path or f"{DIRS['analysis']}/V2_HUMAN_EVAL_KEY_do_not_share.csv"
    s1, s2, key = (pd.read_csv(p, dtype=str, keep_default_na=False) for p in (path1, path2, key_path))
    from sklearn.metrics import cohen_kappa_score
    out = {}
    pref = "preference (A/B/tie)"
    for nm, col in [("preference", pref), ("A_postpositions", "A_postpositions_correct (all/some/none)"),
                    ("B_postpositions", "B_postpositions_correct (all/some/none)")]:
        x, y = s1[col].str.strip().str.lower(), s2[col].str.strip().str.lower()
        ok = (x != "") & (y != "")
        out[f"kappa_{nm}"] = round(float(cohen_kappa_score(x[ok], y[ok])), 3) if ok.sum() > 10 else None
    for ann, s in (("annotator1", s1), ("annotator2", s2)):
        m = s.merge(key, on="item")
        p = m[pref].str.strip().str.upper()
        capm_win = ((p == "A") & (m["A_is"] == "capm")) | ((p == "B") & (m["A_is"] == "baseline"))
        base_win = ((p == "A") & (m["A_is"] == "baseline")) | ((p == "B") & (m["A_is"] == "capm"))
        out[ann] = {"capm_wins": int(capm_win.sum()), "baseline_wins": int(base_win.sum()),
                    "ties": int((p == "TIE").sum()), "n": int((p != "").sum())}
        k = int(capm_win.sum()); n = int(capm_win.sum() + base_win.sum())
        if n:
            from scipy.stats import binomtest
            out[ann]["sign_test_p"] = round(float(binomtest(k, n, 0.5).pvalue), 5)
    jdump(out, f"{DIRS['analysis']}/V2_HUMAN_EVAL_RESULTS.json")
    print(out)
    return out


def v2_evidence_summary(t3, t4, t7):
    s0 = SEEDS[0]

    def row4(mn, a, b):
        r = t4[(t4["metric"] == mn) & (t4["A"] == a) & (t4["B"] == b)] if len(t4) else t4
        return None if r is None or not len(r) else r.iloc[0]

    c1 = row4("casef1", "capm", "baseline"); c2 = row4("casef1", "capm", "capm_no_module"); c3 = row4("chrf++", "capm", "baseline")
    C = {}
    C["C1_case_gain"] = bool(c1 is not None and all(d > 0 for d in c1["per_seed"]) and c1["seed_aware_ci_low"] > 0)
    C["C2_module_needed"] = bool(c2 is not None and c2["seed_aware_ci_low"] > 0)
    C["C3_no_quality_loss"] = bool(c3 is not None and c3["seed_aware_ci_low"] > -0.3)
    r = t3[(t3["metric"] == "casef1") & (t3["A"] == f"capm_s{s0}") & (t3["B"] == f"capm_random_phrase_s{s0}")] if len(t3) else t3
    C["C4_targeting_matters"] = bool(len(r) and r["delta"].iloc[0] > 0 and r["p_ar"].iloc[0] < 0.05)
    if len(t7):
        e = t7.pivot_table(index="set", columns="system", values="caseF1")
        C["C5_external"] = bool({f"capm_s{s0}", f"baseline_s{s0}"} <= set(e.columns) and
                                int((e[f"capm_s{s0}"] >= e[f"baseline_s{s0}"]).sum()) >= 2)
    else:
        C["C5_external"] = False
    if C["C1_case_gain"] and C["C3_no_quality_loss"] and C["C2_module_needed"]:
        verdict = ("METHOD CLAIM SUPPORTED: CAPM improves case-marker realisation without loss in overall quality, "
                   "and the module contributes beyond the objective. Lead the paper with CAPM.")
    elif C["C1_case_gain"] and C["C3_no_quality_loss"]:
        verdict = ("OBJECTIVE CLAIM SUPPORTED, MODULE NOT: the case-aware objective improves case markers, the encoder "
                   "module adds nothing measurable. Present the objective (weighted CE + guided attention) as the method; "
                   "report the module as an ablation.")
    else:
        verdict = ("NO RELIABLE CASE-MARKER GAIN: report PS-CAB and CAPM as two controlled negative results "
                   "(Route 3). Do not claim improvements.")
    lines = ["=" * 72, "PART 4 (CAPM) EVIDENCE SUMMARY -- criteria from PREREGISTRATION_v2.json", "=" * 72]
    for k, v in C.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")
    for lab, rr in (("caseF1 capm-baseline", c1), ("caseF1 capm-no_module", c2), ("chrF++ capm-baseline", c3)):
        if rr is not None:
            lines.append(f"  {lab}: per-seed {rr['per_seed']}, mean {rr['mean']}, seed-aware 95% "
                         f"[{rr['seed_aware_ci_low']}, {rr['seed_aware_ci_high']}]")
    lines += ["", "VERDICT: " + verdict, "",
              "Report also: pilot decision (analysis/v2_pilot_decision.json), lexicon precision (manual check sheet),",
              "that the test set was previously used for PS-CAB (v1) and CAPM was designed from VALIDATION diagnostics,",
              "and the human-evaluation item selection (only sentences where the two systems differ)."]
    txt = "\n".join(lines)
    with open(f"{R}/V2_EVIDENCE_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


# ==============================================================================
# 9. DRIVER
# ==============================================================================
def v2_status():
    done = [r for r in V2_QUEUE if run_done(r)]
    print(f"[v2] queue {len(done)}/{len(V2_QUEUE)} complete | stage A {sum(run_done(r) for r in STAGE_A)}/{len(STAGE_A)}"
          f" | B {sum(run_done(r) for r in STAGE_B)}/{len(STAGE_B)} | C {sum(run_done(r) for r in STAGE_C)}/{len(STAGE_C)}")


def run_part4():
    write_preregistration()
    if not os.path.exists(f"{DIRS['analysis']}/v2_selftest.json"):
        v2_selftest()
    v2_status()
    started = 0
    while started < V2["runs_per_session"]:
        pending_a = [r for r in STAGE_A if not run_done(r)]
        if pending_a:
            nxt = pending_a[0]
        else:
            dec = jload(DECISION_PATH) or v2_pilot_decision()
            if dec is None:
                return
            if not dec["GO"] and not V2["force_continue_after_nogo"]:
                print("[v2] pilot NO-GO -> stopping the queue (see analysis/v2_pilot_decision.json).")
                break
            pending = [r for r in STAGE_B + STAGE_C if not run_done(r)]
            if not pending:
                break
            nxt = pending[0]
        if execute_v2_run(nxt) is None:
            break
        started += 1
    if all(run_done(r) for r in STAGE_A) and not os.path.exists(DECISION_PATH):
        v2_pilot_decision()
    v2_status()
    dec = jload(DECISION_PATH)
    if all(run_done(r) for r in V2_QUEUE) or (dec and not dec["GO"] and not V2["force_continue_after_nogo"]):
        v2_final_report()
    else:
        print("[v2] Save the version; next session: attach its output, run Part 1, then this cell again.")


run_part4()
