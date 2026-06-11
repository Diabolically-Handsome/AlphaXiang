import torch, sys
for p in sys.argv[1:]:
    try:
        s = torch.load(p, map_location="cpu", weights_only=False)
        print(f"{p}")
        print(f"  step={s.get('global_step','?')}")
    except Exception as e:
        print(f"{p}: ERROR {type(e).__name__}: {e}")
