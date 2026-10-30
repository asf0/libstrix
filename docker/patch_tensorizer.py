"""tensorizer 2.10.1 on Python 3.14: its Constants metaclass builds a NamedTuple
from the class-body __annotations__, which PEP 649 replaced with a lazy
__annotate_func__. Same fix as the native env's patch-py314-deps.sh."""

import pathlib
import sysconfig

p = pathlib.Path(sysconfig.get_paths()["purelib"], "tensorizer/_crypt/_encryption.py")
s = p.read_text()
old = '        annotations = dct.pop("__annotations__", {})\n'
new = (
    '        annotations = dct.pop("__annotations__", None)\n'
    "        if annotations is None:\n"
    '            annotate = dct.pop("__annotate_func__", None) or dct.pop("__annotate__", None)\n'
    "            annotations = annotate(1) if annotate is not None else {}\n"
)
if "__annotate_func__" not in s:
    assert old in s, "tensorizer patch anchor not found"
    s = s.replace(old, new, 1).replace(
        "annotations[k] for k in entries", "annotations.get(k, int) for k in entries", 1
    )
    p.write_text(s)
    print("patched", p)
