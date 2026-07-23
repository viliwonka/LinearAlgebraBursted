import re, sys, difflib
from collections import OrderedDict

NUMRE = re.compile(r'^-?\d+\.?\d*(E[+-]\d+)?$', re.IGNORECASE)
HDR_UNIT_RE = re.compile(r'\(ms\)|\(us\)|GFLOP|Gelem|speedup', re.IGNORECASE)
TIMING_COL_RE = re.compile(r'\(ms\)|\(us\)', re.IGNORECASE)
NONIDENTITY_RE = re.compile(r'\(ms\)|\(us\)|GFLOP|Gelem|speedup|residual|status|iters|asChanges|KKT', re.IGNORECASE)

def is_title(line):
    s = line.rstrip('\n')
    return s.startswith('===') or s.startswith('---')

def title_text(line):
    return line.strip().strip('=- ').strip()

def is_header(line):
    s = line.strip()
    if not s:
        return False
    if re.match(r'^(float|double)\b', s):
        return False
    return bool(HDR_UNIT_RE.search(s))

def is_data(line):
    s = line.strip()
    if not s:
        return False
    toks = s.split()
    if len(toks) < 3:
        return False
    numcount = sum(1 for t in toks if NUMRE.match(t))
    return numcount >= 2

def parse_blocks(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    result = []
    cur_title = "PREAMBLE"
    cur_header = None
    cur_rows = []
    def flush():
        if cur_header is not None and cur_rows:
            result.append({'title': cur_title, 'header': cur_header, 'rows': list(cur_rows)})
    for raw in lines:
        line = raw.rstrip('\n')
        if is_title(line):
            flush()
            cur_title = title_text(line)
            cur_header = None
            cur_rows = []
            continue
        if is_header(line):
            flush()
            cur_header = line.split()
            cur_rows = []
            continue
        if cur_header is not None and is_data(line):
            toks = line.split()
            cur_rows.append(toks)
            continue
    flush()
    return result

def row_key(header, row):
    # identity = tokens whose header column doesn't look like a timing/derived metric
    n = min(len(header), len(row))
    key = []
    for i in range(n):
        if not NONIDENTITY_RE.search(header[i]):
            key.append(row[i])
    return tuple(key)

def metric_cols(header):
    # return list of (colidx, colname) for timing columns, med(ms) first if present
    cols = [(i, h) for i, h in enumerate(header) if TIMING_COL_RE.search(h)]
    return cols

def primary_metric_idx(header):
    for i, h in enumerate(header):
        if h.lower().startswith('med(ms)') or h.lower() == 'med(ms)':
            return i
    for i, h in enumerate(header):
        if TIMING_COL_RE.search(h):
            return i
    return None

def block_signature(b):
    return b['title']

def main():
    old = parse_blocks('old.txt')
    new = parse_blocks('new.txt')
    old_titles = [b['title'] for b in old]
    new_titles = [b['title'] for b in new]
    sm = difflib.SequenceMatcher(None, old_titles, new_titles, autojunk=False)
    opcodes = sm.get_opcodes()

    pairs = []  # (old_block or None, new_block or None, kind)
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for k in range(i2 - i1):
                pairs.append((old[i1+k], new[j1+k], 'equal'))
        elif tag == 'replace':
            # pair up 1-1 as far as possible (renames), rest are add/remove
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                pairs.append((old[i1+k], new[j1+k], 'renamed'))
            for k in range(n, i2 - i1):
                pairs.append((old[i1+k], None, 'removed'))
            for k in range(n, j2 - j1):
                pairs.append((None, new[j1+k], 'added'))
        elif tag == 'delete':
            for k in range(i1, i2):
                pairs.append((old[k], None, 'removed'))
        elif tag == 'insert':
            for k in range(j1, j2):
                pairs.append((None, new[k], 'added'))

    slower = []   # (ratio, pct, block_title, metric, key, old_val, new_val)
    faster = []
    missing_rows = []   # rows in old block not matched in new
    extra_rows = []      # rows in new block not matched in old
    added_blocks = []
    removed_blocks = []
    renamed_blocks = []

    for ob, nb, kind in pairs:
        if kind == 'added':
            added_blocks.append(nb['title'])
            continue
        if kind == 'removed':
            removed_blocks.append(ob['title'])
            continue
        if kind == 'renamed':
            renamed_blocks.append((ob['title'], nb['title']))
        # compare rows
        oh, nh = ob['header'], nb['header']
        o_pidx = primary_metric_idx(oh)
        n_pidx = primary_metric_idx(nh)
        o_rows = {row_key(oh, r): r for r in ob['rows']}
        n_rows = {row_key(nh, r): r for r in nb['rows']}
        common_keys = set(o_rows) & set(n_rows)
        for k in set(o_rows) - set(n_rows):
            missing_rows.append((ob['title'], k))
        for k in set(n_rows) - set(o_rows):
            extra_rows.append((nb['title'], k))
        if o_pidx is None or n_pidx is None:
            continue
        oname = oh[o_pidx]
        nname = nh[n_pidx]
        for k in common_keys:
            orow = o_rows[k]
            nrow = n_rows[k]
            try:
                ov = float(orow[o_pidx])
                nv = float(nrow[n_pidx])
            except (ValueError, IndexError):
                continue
            if ov <= 0:
                continue
            ratio = nv / ov
            pct = (ratio - 1.0) * 100.0
            title_disp = nb['title'] if nb else ob['title']
            entry = (ratio, pct, title_disp, f"{oname}->{nname}", k, ov, nv)
            if ratio >= 1.10:
                slower.append(entry)
            elif ratio <= 0.75:
                faster.append(entry)

    slower.sort(key=lambda e: -e[0])
    faster.sort(key=lambda e: e[0])

    with open('report.txt', 'w', encoding='utf-8') as out:
        out.write(f"blocks: old={len(old)} new={len(new)} pairs={len(pairs)}\n")
        out.write(f"added_blocks={len(added_blocks)} removed_blocks={len(removed_blocks)} renamed_blocks={len(renamed_blocks)}\n\n")
        out.write("=== ADDED BLOCKS (new only, no baseline) ===\n")
        for t in added_blocks:
            out.write(f"  + {t}\n")
        out.write("\n=== REMOVED BLOCKS (in baseline, missing now) ===\n")
        for t in removed_blocks:
            out.write(f"  - {t}\n")
        out.write("\n=== RENAMED/REALIGNED BLOCKS (title changed, rows compared anyway) ===\n")
        for o, n in renamed_blocks:
            out.write(f"  ~ {o}  ==>  {n}\n")

        out.write(f"\n=== SLOWER rows (new/old ratio >= 1.10) : {len(slower)} ===\n")
        for ratio, pct, title, metric, key, ov, nv in slower:
            out.write(f"  {pct:+7.1f}%  ratio={ratio:.3f}  [{title}] {metric} key={key} old={ov} new={nv}\n")

        out.write(f"\n=== FASTER rows (new/old ratio <= 0.75) : {len(faster)} ===\n")
        for ratio, pct, title, metric, key, ov, nv in faster:
            out.write(f"  {pct:+7.1f}%  ratio={ratio:.3f}  [{title}] {metric} key={key} old={ov} new={nv}\n")

        out.write(f"\n=== ROWS present in OLD block but missing in matched NEW block : {len(missing_rows)} ===\n")
        for title, k in missing_rows[:400]:
            out.write(f"  [{title}] key={k}\n")

        out.write(f"\n=== ROWS present in NEW block but missing in matched OLD block : {len(extra_rows)} ===\n")
        for title, k in extra_rows[:400]:
            out.write(f"  [{title}] key={k}\n")

    print("done")

if __name__ == '__main__':
    main()
