import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from preprocessing.data_loader import Illness, Injury, Performance, SoccerPlayer, Team


def flatten_list(any_list: List[List[Any]]) -> List[Any]:
    return [item for sublist in any_list for item in sublist]


def clean_suffix(file_name: str) -> str:
    return file_name.split(".")[0]


def read_in_csv_file(path_to_csv_file: Path) -> pd.DataFrame:
    return pd.read_csv(path_to_csv_file)


def read_in_json(path_to_json: Path, variable_name) -> Dict[str, List[int]]:
    with open(path_to_json) as json_file:
        return json.load(json_file)[variable_name]


def get_player_ids(csv_file: pd.DataFrame) -> List[str]:
    return csv_file.columns[1:]


def initialise_injuries(
    injury_df: pd.DataFrame, names: List[str]
) -> Dict[str, List[Injury]]:
    injuries: Dict = {name: [] for name in names}
    for name, injuries_by_player in injury_df.groupby("player_name"):
        injuries[name] = []
        for _, row in injuries_by_player.iterrows():
            injuries[name].append(
                Injury(row["player_name"], row["type"], row["timestamp"])
            )
    return injuries


def initialise_illness(
    illness_df: pd.DataFrame, names: List[str]
) -> Dict[str, List[Injury]]:
    illness: Dict = {name: [] for name in names}
    for name, illness_by_player in illness_df.groupby("player_name"):
        illness[name] = []
        for _, row in illness_df.iterrows():
            illness[name].append(
                Illness(row["player_name"], row["problems"], row["timestamp"])
            )
    return illness


def initialise_performance(
    performance_df: pd.DataFrame, names: List[str]
) -> Dict[str, List[Injury]]:
    performances: Dict = {name: [] for name in names}
    for name, performance_by_player in performance_df.groupby("player_name"):
        performances[name] = []
        for _, row in performance_by_player.iterrows():
            performances[name].append(
                Performance(
                    row["player_name"],
                    row["team_performance"],
                    row["offensive_performance"],
                    row["defensive_performance"],
                    row["timestamp"],
                )
            )
    return performances


def read_in_variable_files(path_to_variable_folder: Path) -> Dict[str, Dict[str, Any]]:
    file_names = os.listdir(path_to_variable_folder)
    json_files = {
        clean_suffix(file): read_in_json(
            path_to_variable_folder / file, clean_suffix(file)
        )
        for file in file_names
        if file.endswith(".json")
    }
    csv_files = {
        clean_suffix(file): read_in_csv_file(path_to_variable_folder / file)
        for file in file_names
        if file.endswith(".csv")
    }
    names = get_player_ids(csv_files["stress"])
    csv_files["performance"] = initialise_performance(csv_files["performance"], names)
    csv_files["illness"] = initialise_illness(csv_files["illness"], names)
    csv_files["injuries"] = initialise_injuries(csv_files["injuries"], names)
    return {**json_files, **csv_files}


def initialise_player(name: str, variables: Dict[str, Dict[str, Any]]) -> SoccerPlayer:

    time_index = pd.to_datetime(
        variables["daily_load"]["Date"].values, format="%d.%m.%Y"
    )
    return SoccerPlayer(
        name,
        pd.Series(variables["daily_load"][name]).set_axis(time_index),
        variables["srpe"][name],
        variables["rpe"][name],
        variables["duration"][name],
        pd.Series(variables["atl"][name]).set_axis(time_index),
        pd.Series(variables["weekly_load"][name]).set_axis(time_index),
        pd.Series(variables["monotony"][name]).set_axis(time_index),
        pd.Series(variables["strain"][name]).set_axis(time_index),
        pd.Series(variables["acwr"][name]).set_axis(time_index),
        pd.Series(variables["ctl28"][name]).set_axis(time_index),
        pd.Series(variables["ctl42"][name]).set_axis(time_index),
        pd.Series(variables["fatigue"][name]).set_axis(time_index),
        pd.Series(variables["mood"][name]).set_axis(time_index),
        pd.Series(variables["readiness"][name]).set_axis(time_index),
        pd.Series(variables["sleep_duration"][name]).set_axis(time_index),
        pd.Series(variables["sleep_quality"][name]).set_axis(time_index),
        pd.Series(variables["soreness"][name]).set_axis(time_index),
        pd.Series(variables["stress"][name]).set_axis(time_index),
        variables["injuries"][name],
        variables["illness"][name],
        variables["performance"][name],
    )


def initialise_players(path_to_data: Path) -> List[SoccerPlayer]:
    files = read_in_variable_files(path_to_data)
    names = get_player_ids(files["stress"])
    players = []
    for name in names:
        players.append(initialise_player(name, files))
    return players


def get_team_name(player_id: str) -> str:
    return player_id[:5]


def get_team_game_performance(players: List[SoccerPlayer]) -> pd.DataFrame:
    all_performances = flatten_list([player.performance for player in players])
    return pd.DataFrame(
        {
            "name": [performance.name for performance in all_performances],
            "team_performance": [
                performance.team_performance for performance in all_performances
            ],
            "offensive_performance": [
                performance.offensive_performance for performance in all_performances
            ],
            "defensive_performance": [
                performance.defensive_performance for performance in all_performances
            ],
            "timestamp": [performance.timestamp for performance in all_performances],
        }
    )


def get_game_ts(time_index: pd.Index, time_stamps: pd.Series) -> pd.Series:
    """Extract the times when injuries occurred. For now only the time stamps are extracted and
    an injury is binary event. However, there are more information stored and can be extracted, like
    how many, injuries, what is injured and severity."""
    binary_timeseries = {
        time: (0 if time not in time_stamps else 1) for time in time_index.tolist()
    }
    return pd.Series(binary_timeseries.values(), index=binary_timeseries.keys())


def generate_team(players: List[SoccerPlayer], team_name: str) -> Team:
    team_players = {
        player.name: player for player in players if team_name in player.name
    }
    time_index = list(team_players.values())[0].stress.index
    game_performance = get_team_game_performance(list(team_players.values()))
    game_ts = get_game_ts(time_index, game_performance["timestamp"])
    return Team(team_name, game_performance, game_ts, team_players)


def generate_teams(path_to_data: Path) -> Dict[str, Team]:
    team_names = ["TeamA", "TeamB"]
    players = initialise_players(path_to_data)
    return {team_name: generate_team(players, team_name) for team_name in team_names}
