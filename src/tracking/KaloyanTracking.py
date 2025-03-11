# Helper file. All of Kaloyan tracking code is in src/kaloyanBarry

def import_kaloyan_tracking():
    import sys
    import os
    kaloyan_path = os.path.join('src/kaloyanBarry')
    sys.path.append(kaloyan_path)

    try:
        import apps.lateral_tracking.tracking as kaloyan_tracking
    except ImportError as e:
        raise ImportError(f"Failed to import kaloyan tracking module: {e}")

    return kaloyan_tracking