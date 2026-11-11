from pathlib import Path

def get_latest_model(model_name, artifacts_dir='../artifacts'):

    model_files = list(Path(artifacts_dir).glob(f'*{model_name}*.bin'))

    if not model_files:
        raise FileNotFoundError(f'No model found for {model_name}')

    return max(model_files, key=lambda file: file.stat().st_mtime)