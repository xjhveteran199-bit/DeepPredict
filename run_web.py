"""Launch wrapper with error capture"""
import sys
import traceback

try:
    import deeppredict_web
except Exception:
    print("IMPORT ERROR:", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
