from torch import save, load

from pathlib import Path
import sys

def main(from_path=str, to_path=str):
    if not Path(from_path).exists():
        print(f"No saved training found in {from_path}")
        return
    
    training_save = load(from_path)
    save(training_save["model_state_dict"], to_path)
    print(f"Saved model to {to_path}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Not enough arguments")