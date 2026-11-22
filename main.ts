// main.ts is mend, the CLI SPEC.md specifies. Node 22, the standard library alone; run it as
// `node main.ts [options] [PATCH]`. It implements all of SPEC.md: the command line, the output and exit codes,
// safe paths, format detection, matching lines at three tiers (section 5), unified diffs (section 6),
// SEARCH/REPLACE blocks (section 7), V4A patches (section 8), JSON edits (section 9) and the
// all-or-nothing write with move reporting (section 10).
//
// In a trial the grader runs `node /bot/main.ts` with MAKEFUN_ENV_URL set, and then this file hands over to
// runner.ts (the last lines below): keep that hand-over and runner.ts as they are, and write the CLI.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

const FORMATS = ['auto', 'unified', 'search-replace', 'v4a', 'json'];

class Exit2 {
  readonly line: string;
  constructor(line: string) {
    this.line = line;
  }
}
class Fail {
  readonly code: string;
  readonly path: string;
  constructor(code: string, path: string) {
    this.code = code;
    this.path = path;
  }
}

const blank = (s: string): boolean => s.trim() === '';

function patchLines(text: string): string[] {
  if (text === '') return [];
  const lines = text.split('\n');
  if (text.endsWith('\n')) lines.pop();
  return lines.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));
}

function safe(p: string): boolean {
  return p !== '' && !p.startsWith('/') && !p.includes('\\') && p.split('/').every((s) => s !== '' && s !== '.' && s !== '..');
}

/** The tree as the edits leave it: changed paths in `now` (null: deleted), the rest read from disk. */
class Tree {
  private readonly orig = new Map<string, string | null>();
  readonly now = new Map<string, string | null>();
  original(p: string): string | null {
    if (!this.orig.has(p)) {
      let v: string | null = null;
      try {
        if (statSync(p).isFile()) v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.orig.set(p, v);
    }
    return this.orig.get(p)!;
  }
  get(p: string): string | null {
    return this.now.has(p) ? this.now.get(p)! : this.original(p);
  }
}

/** Run-wide context the parsers and edits share: --strict, and the moves applied, for `R` lines. */
interface Ctx {
  strict: boolean;
  moves: Array<[string, string]>;
}

type Edit = (tree: Tree) => void;

/** A file split into lines: its own line break, and whether it ends with one (SPEC.md section 3). */
interface SplitFile {
  lines: string[];
  br: string;
  final: boolean;
}

function splitFile(text: string): SplitFile {
  if (text === '') return { lines: [], br: '\n', final: false };
  const nl = text.indexOf('\n');
  const br = nl > 0 && text[nl - 1] === '\r' ? '\r\n' : '\n';
  const lines = text.split(br);
  const final = text.endsWith(br);
  if (final) lines.pop();
  return { lines, br, final };
}

function joinFile(f: SplitFile): string {
  if (f.lines.length === 0) return '';
  return f.lines.join(f.br) + (f.final ? f.br : '');
}

const stripTrail = (s: string): string => s.replace(/[ \t]+$/, '');
const stripBoth = (s: string): string => s.replace(/^[ \t]+/, '').replace(/[ \t]+$/, '');
const leadingWs = (s: string): string => /^[ \t]*/.exec(s)?.[0] ?? '';
const onlyWs = (s: string): boolean => /^[ \t]*$/.test(s);

function lineEqAt(a: string, b: string, tier: number): boolean {
  if (tier === 0) return a === b;
  if (tier === 1) return stripTrail(a) === stripTrail(b);
  return stripBoth(a) === stripBoth(b);
}

/** Positions i >= from where O matches lines[i..], at one tier; eofOnly keeps only matches ending at the last line. */
function occurrences(O: string[], lines: string[], from: number, tier: number, eofOnly: boolean): number[] {
  const out: number[] = [];
  if (O.length === 0 || O.length > lines.length) return out;
  const last = lines.length - O.length;
  for (let i = Math.max(from, 0); i <= last; i++) {
    if (eofOnly && i !== last) continue;
    let ok = true;
    for (let k = 0; k < O.length; k++) {
      if (!lineEqAt(O[k]!, lines[i + k]!, tier)) {
        ok = false;
        break;
      }
    }
    if (ok) out.push(i);
  }
  return out;
}

/** The first tier (0, then 1, then 2; only 0 with --strict) at which O occurs at least once. */
function firstTier(O: string[], lines: string[], from: number, eofOnly: boolean, strict: boolean): { tier: number; positions: number[] } | null {
  const tiers = strict ? [0] : [0, 1, 2];
  for (const t of tiers) {
    const positions = occurrences(O, lines, from, t, eofOnly);
    if (positions.length > 0) return { tier: t, positions };
  }
  return null;
}

/** The occurrence nearest to target; of two equally near, the earlier. */
function nearestPos(positions: number[], target: number): number {
  let best = positions[0]!;
  for (const p of positions) {
    if (Math.abs(p - target) < Math.abs(best - target)) best = p;
  }
  return best;
}

/** One line of a hunk or chunk body: context (also in O), old (only in O) or new (only in N). */
type BodyKind = 'ctx' | 'old' | 'new' | 'slash';
interface BodyLine {
  kind: BodyKind;
  text: string;
}

/** O, N and the no-end-of-line flag from a hunk or chunk body (SPEC.md sections 5 and 6). */
function bodyToON(body: BodyLine[]): { O: string[]; N: Array<{ text: string; ctx: boolean; oIdx: number }>; noEol: boolean } {
  const O: string[] = [];
  const N: Array<{ text: string; ctx: boolean; oIdx: number }> = [];
  for (const b of body) {
    if (b.kind === 'ctx') {
      O.push(b.text);
      N.push({ text: b.text, ctx: true, oIdx: O.length - 1 });
    } else if (b.kind === 'old') {
      O.push(b.text);
    } else if (b.kind === 'new') {
      N.push({ text: b.text, ctx: false, oIdx: -1 });
    }
  }
  let noEol = false;
  for (let j = body.length - 1; j >= 0; j--) {
    const b = body[j]!;
    if (b.kind === 'ctx' || b.kind === 'new') {
      noEol = j + 1 < body.length && body[j + 1]!.kind === 'slash';
      break;
    }
  }
  return { O, N, noEol };
}

/** The lines that replace a match: context lines as the file's own lines, other lines re-indented at tier 2. */
function buildReplacement(
  N: Array<{ text: string; ctx: boolean; oIdx: number }>,
  O: string[],
  fileLines: string[],
  pos: number,
  tier: number,
): string[] {
  const F = leadingWs(fileLines[pos] ?? '');
  const P = leadingWs(O[0] ?? '');
  return N.map((nl) => {
    if (nl.ctx) return fileLines[pos + nl.oIdx]!;
    if (tier === 2) {
      const t = nl.text;
      if (t !== '' && !onlyWs(t) && t.startsWith(P)) return F + t.slice(P.length);
    }
    return nl.text;
  });
}

function detect(lines: string[]): string | null {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) return null;
  if (lines[first]!.trim() === '*** Begin Patch') return 'v4a';
  const c = lines[first]!.trimStart()[0];
  if (c === '[' || c === '{') return 'json';
  if (lines.some((l) => l.trimEnd() === '<<<<<<< SEARCH')) return 'search-replace';
  if (lines.some((l, i) => l.startsWith('diff --git ') || (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')))) return 'unified';
  return null;
}

/** One JSON edit object (SPEC.md section 9), or null when it is not one. */
function jsonEdit(v: unknown): Edit | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const p = o.path;
  if (typeof p !== 'string') return null;
  const exists = (tree: Tree): string => {
    if (!safe(p)) throw new Fail('unsafe', p);
    const t = tree.get(p);
    if (t === null) throw new Fail('missing', p);
    return t;
  };
  if (o.command === 'create' && typeof o.file_text === 'string') {
    const text = o.file_text;
    return (tree) => {
      if (!safe(p)) throw new Fail('unsafe', p);
      if (tree.get(p) !== null) throw new Fail('exists', p);
      tree.now.set(p, text);
    };
  }
  if (o.command === 'str_replace' && typeof o.old_str === 'string' && o.old_str !== '' && (o.new_str === undefined || typeof o.new_str === 'string')) {
    const oldStr = o.old_str;
    const newStr = (o.new_str as string | undefined) ?? '';
    return (tree) => {
      const before = exists(tree);
      const nl = before.indexOf('\n');
      const crlf = nl > 0 && before[nl - 1] === '\r';
      const lf = (s: string): string => (crlf ? s.replaceAll('\r\n', '\n') : s);
      const t = lf(before);
      const old = lf(oldStr);
      let count = 0;
      for (let k = t.indexOf(old); k >= 0; k = t.indexOf(old, k + 1)) count += 1;
      if (count === 0) throw new Fail('no-match', p);
      if (count > 1) throw new Fail('ambiguous', p);
      const at = t.indexOf(old);
      const out = t.slice(0, at) + lf(newStr) + t.slice(at + old.length);
      tree.now.set(p, crlf ? out.replaceAll('\n', '\r\n') : out);
    };
  }
  if (o.command === 'insert' && typeof o.insert_line === 'number' && Number.isInteger(o.insert_line) && typeof o.new_str === 'string') {
    const line = o.insert_line;
    const add = patchLines(o.new_str);
    return (tree) => {
      const before = exists(tree);
      const nl = before.indexOf('\n');
      const br = nl > 0 && before[nl - 1] === '\r' ? '\r\n' : '\n';
      const lines = before === '' ? [] : before.split(br);
      const final = before !== '' && before.endsWith(br);
      if (final) lines.pop();
      if (line < 0 || line > lines.length) throw new Fail('no-match', p);
      lines.splice(line, 0, ...add);
      tree.now.set(p, lines.length === 0 ? '' : lines.join(br) + (final ? br : ''));
    };
  }
  return null;
}

function jsonEdits(text: string, lines: string[]): Edit[] {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  const edits: Edit[] = [];
  if (lines[first]!.trimStart().startsWith('[')) {
    let v: unknown;
    try {
      v = JSON.parse(text);
    } catch {
      throw new Exit2(`error: parse ${first + 1}`);
    }
    if (!Array.isArray(v)) throw new Exit2(`error: parse ${first + 1}`);
    for (const x of v) {
      const e = jsonEdit(x);
      if (e === null) throw new Exit2(`error: parse ${first + 1}`);
      edits.push(e);
    }
    return edits;
  }
  lines.forEach((l, i) => {
    if (blank(l)) return;
    let e: Edit | null = null;
    try {
      e = jsonEdit(JSON.parse(l));
    } catch {
      e = null;
    }
    if (e === null) throw new Exit2(`error: parse ${i + 1}`);
    edits.push(e);
  });
  return edits;
}

/** ---- Unified diffs (SPEC.md section 6) ---- */

interface UnifiedHunk {
  a: number;
  body: BodyLine[];
}

interface UnifiedSection {
  oldP: string | null; // null: /dev/null, no file
  newP: string | null;
  newFile: boolean;
  delFile: boolean;
  hunks: UnifiedHunk[];
}

/** The path of a `--- ` / `+++ ` line: text after the four characters, up to the first tab, trimmed, leading a/ or b/ removed. */
function dashPath(line: string): string | null {
  const raw = line.slice(4).split('\t')[0]!.trim();
  const stripped = raw.startsWith('a/') || raw.startsWith('b/') ? raw.slice(2) : raw;
  return stripped === '/dev/null' ? null : stripped;
}

const devNull = (s: string): string | null => (s === '/dev/null' ? null : s);
const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/;
const DIFF_GIT_RE = /^diff --git a\/(.+) b\/(.+)$/;

function unifiedEdits(lines: string[], ctx: Ctx): Edit[] {
  const n = lines.length;
  const sections: UnifiedSection[] = [];
  let i = 0;
  while (i < n) {
    const l = lines[i]!;
    let sec: UnifiedSection;
    let pairSeen: boolean;
    if (l.startsWith('diff --git ')) {
      const m = DIFF_GIT_RE.exec(l);
      if (!m) throw new Exit2(`error: parse ${i + 1}`);
      sec = { oldP: devNull(m[1]!), newP: devNull(m[2]!), newFile: false, delFile: false, hunks: [] };
      pairSeen = false;
      i += 1;
    } else if (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')) {
      sec = { oldP: dashPath(l), newP: dashPath(lines[i + 1]!), newFile: false, delFile: false, hunks: [] };
      pairSeen = true;
      i += 2;
    } else if (l.startsWith('@@')) {
      throw new Exit2(`error: parse ${i + 1}`); // a hunk before any section
    } else {
      i += 1; // prose outside sections
      continue;
    }
    let renameFrom: string | null = null;
    let renameTo: string | null = null;
    let inHeader = true;
    while (i < n) {
      const h = lines[i]!;
      if (h.startsWith('@@')) {
        const m = HUNK_RE.exec(h);
        if (!m) throw new Exit2(`error: parse ${i + 1}`);
        const a = Number(m[1]);
        i += 1;
        const body: BodyLine[] = [];
        while (i < n) {
          const bl = lines[i]!;
          if (bl.startsWith('diff --git ')) break;
          if (bl.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')) break;
          if (bl === '' || bl.startsWith(' ') || bl.startsWith('-') || bl.startsWith('+') || bl.startsWith('\\')) {
            body.push(
              bl === '' || bl.startsWith(' ')
                ? { kind: 'ctx', text: bl === '' ? '' : bl.slice(1) }
                : bl.startsWith('-')
                  ? { kind: 'old', text: bl.slice(1) }
                  : bl.startsWith('+')
                    ? { kind: 'new', text: bl.slice(1) }
                    : { kind: 'slash', text: bl.slice(1) },
            );
            i += 1;
          } else break;
        }
        while (body.length > 0 && body[body.length - 1]!.kind === 'ctx' && body[body.length - 1]!.text === '') body.pop();
        sec.hunks.push({ a, body });
        inHeader = false;
        continue;
      }
      if (h.startsWith('diff --git ')) break; // a new section starts
      if (h.startsWith('--- ')) {
        if ((lines[i + 1] ?? '').startsWith('+++ ')) {
          if (!inHeader || pairSeen) break; // a new section starts
          sec.oldP = dashPath(h);
          sec.newP = dashPath(lines[i + 1]!);
          pairSeen = true;
          i += 2;
          continue;
        }
        if (inHeader) throw new Exit2(`error: parse ${i + 2}`);
        i += 1; // prose after the hunks
        continue;
      }
      if (h.startsWith('rename from ')) {
        renameFrom = h.slice('rename from '.length).trim();
        i += 1;
        continue;
      }
      if (h.startsWith('rename to ')) {
        renameTo = h.slice('rename to '.length).trim();
        i += 1;
        continue;
      }
      if (h.startsWith('new file mode')) {
        sec.newFile = true;
        i += 1;
        continue;
      }
      if (h.startsWith('deleted file mode')) {
        sec.delFile = true;
        i += 1;
        continue;
      }
      i += 1; // index, old/new mode, similarity, Index: and any other line are ignored
    }
    if (!pairSeen) {
      if (renameFrom !== null) sec.oldP = devNull(renameFrom);
      if (renameTo !== null) sec.newP = devNull(renameTo);
    }
    sections.push(sec);
  }
  return sections.map((sec) => unifiedSectionEdit(sec, ctx));
}

function unifiedSectionEdit(sec: UnifiedSection, ctx: Ctx): Edit {
  return (tree) => {
    const oldP = sec.oldP;
    const newP = sec.newP;
    const isCreate = oldP === null || sec.newFile;
    const isDelete = !isCreate && (newP === null || sec.delFile);
    const isMove = !isCreate && !isDelete && oldP !== newP;
    if (isCreate) {
      const p = newP!;
      if (!safe(p)) throw new Fail('unsafe', p);
      if (tree.get(p) !== null) throw new Fail('exists', p);
      const plus: string[] = [];
      for (const h of sec.hunks) for (const b of h.body) if (b.kind === 'new') plus.push(b.text);
      let noEol = false;
      outer: for (let hi = sec.hunks.length - 1; hi >= 0; hi--) {
        const body = sec.hunks[hi]!.body;
        for (let j = body.length - 1; j >= 0; j--) {
          if (body[j]!.kind === 'new') {
            noEol = j + 1 < body.length && body[j + 1]!.kind === 'slash';
            break outer;
          }
        }
      }
      tree.now.set(p, plus.length === 0 ? '' : `${plus.join('\n')}${noEol ? '' : '\n'}`);
      return;
    }
    if (isDelete) {
      const p = oldP!;
      if (!safe(p)) throw new Fail('unsafe', p);
      if (tree.get(p) === null) throw new Fail('missing', p);
      tree.now.set(p, null);
      return;
    }
    const p = oldP!;
    if (!safe(p)) throw new Fail('unsafe', p);
    if (isMove && !safe(newP!)) throw new Fail('unsafe', newP!);
    const before = tree.get(p);
    if (before === null) throw new Fail('missing', p);
    if (isMove && tree.get(newP!) !== null) throw new Fail('exists', newP!);
    const f = splitFile(before);
    let d = 0;
    for (const h of sec.hunks) {
      const { O, N, noEol } = bodyToON(h.body);
      if (O.length === 0) {
        const ins = Math.max(0, Math.min(h.a + d, f.lines.length));
        const atEnd = h.a + d >= f.lines.length;
        f.lines.splice(ins, 0, ...N.map((nl) => nl.text));
        if (atEnd) f.final = !noEol;
      } else {
        const found = firstTier(O, f.lines, 0, false, ctx.strict);
        if (!found) throw new Fail('no-match', p);
        const pos = nearestPos(found.positions, h.a - 1 + d);
        const runsToEnd = pos + O.length === f.lines.length;
        f.lines.splice(pos, O.length, ...buildReplacement(N, O, f.lines, pos, found.tier));
        if (runsToEnd) f.final = !noEol;
      }
      d += N.length - O.length;
    }
    const text = joinFile(f);
    if (isMove) {
      tree.now.set(newP!, text);
      tree.now.set(p, null);
      ctx.moves.push([p, newP!]);
    } else {
      tree.now.set(p, text);
    }
  };
}

/** ---- SEARCH/REPLACE blocks (SPEC.md section 7) ---- */

function searchReplaceEdits(lines: string[], ctx: Ctx): Edit[] {
  const edits: Edit[] = [];
  let prevReplaceEnd = -1; // 0-based index of the previous block's >>>>>>> REPLACE line, or -1
  let prevPath: string | null = null;
  let i = 0;
  while (i < lines.length) {
    if (lines[i]!.trimEnd() !== '<<<<<<< SEARCH') {
      i += 1;
      continue;
    }
    const searchLine = i;
    let path: string | null = null;
    for (let j = i - 1; j > prevReplaceEnd; j--) {
      const t = lines[j]!.trim();
      if (t === '') continue;
      if (t.startsWith('```')) continue;
      path = t;
      break;
    }
    if (path === null) {
      if (prevPath === null) throw new Exit2(`error: parse ${searchLine + 1}`);
      path = prevPath;
    }
    let eq = -1;
    for (let j = i + 1; j < lines.length; j++) {
      if (lines[j]!.trimEnd() === '=======') {
        eq = j;
        break;
      }
      if (lines[j]!.trimEnd() === '>>>>>>> REPLACE') break; // a REPLACE before any ======= is malformed
    }
    if (eq === -1) throw new Exit2(`error: parse ${searchLine + 1}`);
    let rep = -1;
    for (let j = eq + 1; j < lines.length; j++) {
      if (lines[j]!.trimEnd() === '>>>>>>> REPLACE') {
        rep = j;
        break;
      }
    }
    if (rep === -1) throw new Exit2(`error: parse ${searchLine + 1}`);
    const oldLines = lines.slice(i + 1, eq);
    const newLines = lines.slice(eq + 1, rep);
    edits.push(searchReplaceEdit(path, oldLines, newLines, ctx));
    prevPath = path;
    prevReplaceEnd = rep;
    i = rep + 1;
  }
  return edits;
}

function searchReplaceEdit(path: string, oldLines: string[], newLines: string[], ctx: Ctx): Edit {
  return (tree) => {
    if (!safe(path)) throw new Fail('unsafe', path);
    if (oldLines.length === 0) {
      if (tree.get(path) !== null) throw new Fail('exists', path);
      tree.now.set(path, newLines.length === 0 ? '' : `${newLines.join('\n')}\n`);
      return;
    }
    const before = tree.get(path);
    if (before === null) throw new Fail('missing', path);
    const f = splitFile(before);
    const found = firstTier(oldLines, f.lines, 0, false, ctx.strict);
    if (!found) throw new Fail('no-match', path);
    if (found.positions.length > 1) throw new Fail('ambiguous', path);
    const pos = found.positions[0]!;
    const N = newLines.map((t) => ({ text: t, ctx: false, oIdx: -1 }));
    f.lines.splice(pos, oldLines.length, ...buildReplacement(N, oldLines, f.lines, pos, found.tier));
    tree.now.set(path, joinFile(f));
  };
}

/** ---- V4A patches (SPEC.md section 8) ---- */

interface V4aChunk {
  anchor: string | null;
  body: BodyLine[];
  eof: boolean;
}

/** The first line at or after cursor equal to the anchor, at the first tier with one; -1 when there is none. */
function findAnchor(anchor: string, lines: string[], cursor: number, strict: boolean): number {
  const tiers = strict ? [0] : [0, 1, 2];
  for (const t of tiers) {
    for (let i = cursor; i < lines.length; i++) {
      if (lineEqAt(anchor, lines[i]!, t)) return i;
    }
  }
  return -1;
}

function v4aEdits(lines: string[], ctx: Ctx): Edit[] {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  if (lines[first]!.trim() !== '*** Begin Patch') throw new Exit2(`error: parse ${first + 1}`);
  let last = lines.length - 1;
  while (last >= 0 && blank(lines[last]!)) last -= 1;
  if (last < 0 || lines[last]!.trim() !== '*** End Patch') throw new Exit2(`error: parse ${lines.length + 1}`);
  const isSectionStart = (l: string): boolean =>
    l.startsWith('*** Add File: ') || l.startsWith('*** Delete File: ') || l.startsWith('*** Update File: ');
  const edits: Edit[] = [];
  let i = first + 1;
  while (i < last) {
    while (i < last && blank(lines[i]!)) i += 1; // blank lines between sections are ignored
    if (i >= last) break;
    const l = lines[i]!;
    const sectionLine = i + 1;
    if (l.startsWith('*** Add File: ') || l.startsWith('*** Delete File: ') || l.startsWith('*** Update File: ')) {
      const isAdd = l.startsWith('*** Add File: ');
      const isDelete = l.startsWith('*** Delete File: ');
      const p = l.slice((isAdd ? '*** Add File: ' : isDelete ? '*** Delete File: ' : '*** Update File: ').length).trim();
      i += 1;
      const raw: Array<{ n: number; text: string }> = [];
      while (i < last && !isSectionStart(lines[i]!)) {
        raw.push({ n: i + 1, text: lines[i]! });
        i += 1;
      }
      while (raw.length > 0 && blank(raw[raw.length - 1]!.text)) raw.pop(); // blank lines at the end of a section
      if (isAdd) {
        const fileLines: string[] = [];
        for (const r of raw) {
          if (r.text.startsWith('+')) fileLines.push(r.text.slice(1));
          else if (r.text === '') fileLines.push('');
          else throw new Exit2(`error: parse ${r.n}`);
        }
        edits.push((tree) => {
          if (!safe(p)) throw new Fail('unsafe', p);
          if (tree.get(p) !== null) throw new Fail('exists', p);
          tree.now.set(p, fileLines.length === 0 ? '' : `${fileLines.join('\n')}\n`);
        });
      } else if (isDelete) {
        if (raw.length > 0) throw new Exit2(`error: parse ${raw[0]!.n}`);
        edits.push((tree) => {
          if (!safe(p)) throw new Fail('unsafe', p);
          if (tree.get(p) === null) throw new Fail('missing', p);
          tree.now.set(p, null);
        });
      } else {
        let newP: string | null = null;
        let rest = raw;
        if (rest.length > 0 && rest[0]!.text.startsWith('*** Move to: ')) {
          newP = rest[0]!.text.slice('*** Move to: '.length).trim();
          rest = rest.slice(1);
        }
        const chunks: V4aChunk[] = [];
        let cur: V4aChunk | null = null;
        let afterEof = false;
        for (const r of rest) {
          const t = r.text;
          if (t.startsWith('@@')) {
            if (cur) chunks.push(cur);
            const anchor = t.slice(2).trim();
            cur = { anchor: anchor === '' ? null : anchor, body: [], eof: false };
            afterEof = false;
          } else if (t === '*** End of File') {
            if (!cur) throw new Exit2(`error: parse ${r.n}`);
            cur.eof = true;
            chunks.push(cur);
            cur = null;
            afterEof = true;
          } else if (t === '' || t.startsWith(' ') || t.startsWith('-') || t.startsWith('+')) {
            if (!cur) {
              if (afterEof) throw new Exit2(`error: parse ${r.n}`);
              cur = { anchor: null, body: [], eof: false };
            }
            cur.body.push(
              t === '' || t.startsWith(' ')
                ? { kind: 'ctx', text: t === '' ? '' : t.slice(1) }
                : t.startsWith('-')
                  ? { kind: 'old', text: t.slice(1) }
                  : { kind: 'new', text: t.slice(1) },
            );
          } else {
            throw new Exit2(`error: parse ${r.n}`);
          }
        }
        if (cur) chunks.push(cur);
        if (chunks.length === 0 && newP === null) throw new Exit2(`error: parse ${sectionLine}`);
        edits.push((tree) => {
          if (!safe(p)) throw new Fail('unsafe', p);
          if (newP !== null && !safe(newP)) throw new Fail('unsafe', newP);
          const before = tree.get(p);
          if (before === null) throw new Fail('missing', p);
          if (newP !== null && tree.get(newP) !== null) throw new Fail('exists', newP);
          const f = splitFile(before);
          let cursor = 0;
          for (const ch of chunks) {
            if (ch.anchor !== null) {
              const ai = findAnchor(ch.anchor, f.lines, cursor, ctx.strict);
              if (ai < 0) throw new Fail('no-match', p);
              cursor = ai + 1;
            }
            const { O, N } = bodyToON(ch.body);
            if (O.length === 0) {
              const ins = ch.eof ? f.lines.length : cursor;
              f.lines.splice(ins, 0, ...N.map((nl) => nl.text));
              cursor = ins + N.length;
            } else {
              const found = firstTier(O, f.lines, cursor, ch.eof, ctx.strict);
              if (!found) throw new Fail('no-match', p);
              const pos = found.positions[0]!;
              f.lines.splice(pos, O.length, ...buildReplacement(N, O, f.lines, pos, found.tier));
              cursor = pos + N.length;
            }
          }
          const text = joinFile(f);
          if (newP !== null) {
            tree.now.set(newP, text);
            tree.now.set(p, null);
            ctx.moves.push([p, newP]);
          } else {
            tree.now.set(p, text);
          }
        });
      }
    } else {
      throw new Exit2(`error: parse ${i + 1}`);
    }
  }
  return edits;
}

function run(args: string[]): { code: number; out: string[] } {
  let format: string | null = null;
  let check = false;
  let strict = false;
  const pos: string[] = [];
  let ended = false;
  for (const a of args) {
    if (!ended && a === '--') ended = true;
    else if (!ended && a.startsWith('-') && a !== '-') {
      if (a.startsWith('--format=') && format === null && FORMATS.includes(a.slice(9))) format = a.slice(9);
      else if (a === '--check' && !check) check = true;
      else if (a === '--strict' && !strict) strict = true;
      else return { code: 2, out: ['error: usage'] };
    } else pos.push(a);
  }
  if (pos.length > 1) return { code: 2, out: ['error: usage'] };
  const file = pos.length === 0 || pos[0] === '-' ? null : pos[0]!;
  let text = '';
  try {
    text = readFileSync(file ?? 0, 'utf8');
  } catch {
    if (file !== null) return { code: 2, out: ['error: no-patch'] };
  }
  const lines = patchLines(text);
  const kind = format === null || format === 'auto' ? detect(lines) : format;
  if (kind === null) return { code: 2, out: ['error: format'] };
  const ctx: Ctx = { strict, moves: [] };
  let edits: Edit[];
  try {
    edits =
      kind === 'json'
        ? jsonEdits(text, lines)
        : kind === 'unified'
          ? unifiedEdits(lines, ctx)
          : kind === 'search-replace'
            ? searchReplaceEdits(lines, ctx)
            : v4aEdits(lines, ctx);
    if (edits.length === 0) throw new Exit2('error: parse 1');
  } catch (e) {
    if (e instanceof Exit2) return { code: 2, out: [e.line] };
    throw e;
  }
  const tree = new Tree();
  const fails: Fail[] = [];
  for (const edit of edits) {
    const before = new Map(tree.now);
    try {
      edit(tree);
    } catch (e) {
      if (!(e instanceof Fail)) throw e;
      tree.now.clear();
      for (const [k, v] of before) tree.now.set(k, v);
      fails.push(e);
    }
  }
  if (fails.length > 0) return { code: 1, out: [...fails.map((f) => `error: ${f.code} ${f.path}`), `failed: ${fails.length} of ${edits.length} edits`] };
  // A move that succeeded is one `R old -> new` line, sorting by the new path, instead of a `D` and an `A` line.
  const rRows: Array<{ key: string; line: string }> = [];
  const rOld = new Set<string>();
  const rNew = new Set<string>();
  for (const [o, nw] of ctx.moves) {
    if (tree.original(o) !== null && tree.get(o) === null && tree.original(nw) === null && tree.get(nw) !== null) {
      rOld.add(o);
      rNew.add(nw);
      rRows.push({ key: nw, line: `R ${o} -> ${nw}` });
    }
  }
  const rows: Array<{ key: string; line: string }> = [...rRows];
  for (const p of [...tree.now.keys()]) {
    const was = tree.original(p);
    const now = tree.now.get(p)!;
    if (was === now) continue;
    if (rOld.has(p) || rNew.has(p)) continue;
    rows.push({ key: p, line: `${was === null ? 'A' : now === null ? 'D' : 'M'} ${p}` });
  }
  if (!check) {
    for (const p of [...tree.now.keys()].sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)))) {
      const was = tree.original(p);
      const now = tree.now.get(p)!;
      if (was === now) continue;
      if (now === null) rmSync(p, { force: true });
      else {
        mkdirSync(dirname(p), { recursive: true });
        writeFileSync(p, now);
      }
    }
  }
  rows.sort((a, b) => Buffer.compare(Buffer.from(a.key), Buffer.from(b.key)));
  return { code: 0, out: [...rows.map((r) => r.line), `ok: ${rows.length} ${rows.length === 1 ? 'file' : 'files'}`] };
}

const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv !== undefined && trialEnv !== '') {
  // A trial: the kit's runner reads the checks from the env and runs this file on each one, without this variable.
  const { runTrial } = await import('./runner.ts');
  await runTrial(trialEnv);
} else {
  const r = run(process.argv.slice(2));
  writeSync(1, r.out.map((l) => `${l}\n`).join(''));
  process.exitCode = r.code;
}
