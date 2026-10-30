"""Replace duplicate ROCm files in the devel tree with relative symlinks.

rocm-sdk init links the runtime libraries of _rocm_sdk_libraries and
_rocm_sdk_core into _rocm_sdk_devel. Docker layers do not keep those as hard
links, so the image would store them twice (about 2 GB). Every devel file whose
relative path exists with identical bytes in one of those trees becomes a
relative symlink to it; symlinks survive layers and resolve to the same file.

    python dedup_rocm.py <dir holding the _rocm_sdk_* trees>
"""

import filecmp
import os
import sys

root = sys.argv[1]
devel = os.path.join(root, "_rocm_sdk_devel")
linked = saved = 0
for name in ("_rocm_sdk_libraries", "_rocm_sdk_core"):
    tree = os.path.join(root, name)
    for d, _, files in os.walk(tree):
        for f in files:
            src = os.path.join(d, f)
            dst = os.path.join(devel, os.path.relpath(src, tree))
            if os.path.islink(src) or os.path.islink(dst) or not os.path.isfile(dst):
                continue
            if os.path.getsize(src) != os.path.getsize(dst):
                continue
            if not filecmp.cmp(src, dst, shallow=False):
                continue
            saved += os.path.getsize(dst)
            os.unlink(dst)
            os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
            linked += 1
print(f"dedup: {linked} devel files -> symlinks, {saved / 2**30:.2f} GiB")
