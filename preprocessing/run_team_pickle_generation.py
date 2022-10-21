from typing import Dict
from pathlib import Path
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.data_loader import Team
from preprocessing.read_in_data import generate_teams


def save_as_pickle(path_to_save: Path, teams_obj: Dict[str, Team]):
    pickle.dump(teams_obj, open(path_to_save / "teams.pkl", "wb"))


if __name__ == "__main__":
    path_to_folder = Path(__file__).parent.parent / "input" / "features"
    path_to_save_folder = Path(__file__).parent.parent / "input"
    teams = generate_teams(path_to_folder)
    save_as_pickle(path_to_save_folder, teams)
