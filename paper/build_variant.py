import sys, subprocess
from pathlib import Path
REPO = Path(r"E:/ns_mc_gan_gi_code_fcc_phase1")
sys.path.insert(0, str(REPO))
import md_to_pdf as m
m.PAPER = REPO / "paper"
STEM = sys.argv[1] if len(sys.argv) > 1 else "OPTICS_DRAFT"
src = REPO / f"paper/{STEM}.md"
raw = src.read_text(encoding="utf-8")
_out, _buf, _in = [], [], False
for _ln in raw.split("\n"):
    if _ln.strip() == "$$":
        if _in:
            _out.append("$$" + " ".join(_buf) + "$$"); _buf, _in = [], False
        else:
            _in = True
    elif _in:
        _buf.append(_ln.strip())
    else:
        _out.append(_ln)
if _in:
    _out.append("$$" + " ".join(_buf) + "$$")
raw = "\n".join(_out)
tex = m.convert(raw)
(REPO / f"paper/{STEM}.tex").write_text(tex, encoding="utf-8")
r = None
for _ in range(2):
    r = subprocess.run([m.XELATEX, "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
                       cwd=str(REPO / "paper"), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
pdf = REPO / f"paper/{STEM}.pdf"
print("PDF:", pdf if pdf.exists() else "FAILED")
if not pdf.exists():
    out = (r.stdout or "") + (r.stderr or "")
    errs = [l for l in out.splitlines() if l.startswith("!") or "Error" in l or "Undefined" in l]
    print("\n".join(errs[:20]) or out[-3000:])
