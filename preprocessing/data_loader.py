from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
from uuid import uuid4

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import warnings


class NoDateIndex(Exception):
    def __init__(self, value):
        message = f"Variable {value} cannot be indexed by date"
        super().__init__(message)


class VarNotFound(Exception):
    def __init__(self, value):
        message = f"Class does not contain variable {value}"
        super().__init__(message)


class DateNotInRange(Exception):
    def __init__(self):
        message = f"Selected date does not exist"
        super().__init__(message)


wellness_sheets_names = [
    "Game Performance",
    "Injury",
    "Illness",
    "Fatigue",
    "Mood",
    "Readiness",
    "SleepDurH",
    "SleepQuality",
    "Soreness",
    "Stress",
]

sheets = [
    "Game Performance",
    "Injury",
    "Fatigue",
    "Mood",
    "Readiness",
    "SleepDurH",
    "SleepQuality",
    "Soreness",
    "Stress",
]


def check_if_variable_callable(variable_name, player):
    if variable_name not in player.__annotations__.keys():
        raise VarNotFound(variable_name)
    if variable_name in ["srpe", "rpe", "injuries", "illness", "performance", "name"]:
        raise NoDateIndex(variable_name)


@dataclass(frozen=True)
class Illness:
    player: str
    problems: List[str]
    timestamp: pd.Index


@dataclass(frozen=True)
class Injury:
    player: str
    type: Dict[str, str]
    timestamp: pd.Index


@dataclass(frozen=True)
class Performance:
    name: str
    team_performance: int
    offensive_performance: int
    defensive_performance: int
    timestamp: pd.Index


@dataclass(frozen=True)
class SoccerPlayer:
    name: str
    daily_load: pd.Series
    srpe: pd.Series
    rpe: pd.Series
    duration: pd.Series
    atl: pd.Series
    weekly_load: pd.Series
    monotony: pd.Series
    strain: pd.Series
    acwr: pd.Series
    ctl28: pd.Series
    ctl42: pd.Series
    fatigue: pd.Series
    mood: pd.Series
    readiness: pd.Series
    sleep_duration: pd.Series
    sleep_quality: pd.Series
    soreness: pd.Series
    stress: pd.Series
    injuries: List[Injury]
    illness: List[Illness]
    performance: List[Performance]

    def get_variable_names(self) -> List[str]:
        return list(self.__annotations__.keys())

    def get_variables_by_date(
        self,
        variable_names: List[str],
        from_date: str = "01.01.2020",
        until_date: str = "31.12.2021",
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                get_variable_by_date(self, variable_name, from_date, until_date)
                for variable_name in variable_names
            ]
        ).T

    def to_dataframe(self, pseudonym):
        feature_df = pd.DataFrame(
            {
                "player_name": [pseudonym] * len(self.daily_load),
                "daily_load": self.daily_load,
                "atl": self.atl,
                "weekly_load": self.weekly_load,
                "monotony": self.monotony,
                "strain": self.strain,
                "acwr": self.acwr,
                "ctl28": self.ctl28,
                "ctl42": self.ctl42,
                "fatigue": self.fatigue,
                "mood": self.mood,
                "readiness": self.readiness,
                "sleep-duration": self.sleep_duration,
                "sleep-quality": self.sleep_quality,
                "soreness": self.soreness,
                "stress": self.stress,
                "injury_ts": self.injury_ts,
            },
        )
        return feature_df

    def to_session_dataframe(self, pseudonym):
        session_feature_df = pd.DataFrame(
            {
                "player_name": [pseudonym] * len(self.srpe),
                "srpe": self.srpe,
                "rpe": self.rpe,
                "duration": self.duration,
            },
        )
        return session_feature_df

    def to_injuries_to_dataframe(self, pseudonym):
        if self.injuries:
            injury_dicts = [asdict(injury) for injury in self.injuries]
            pseudonymed = [
                dict(item, **{"player_name": pseudonym}) for item in injury_dicts
            ]
            return [pd.DataFrame(pseudonymed)[["player_name", "type", "timestamp"]]]
        return []

    def to_illness_to_dataframe(self, pseudonym):
        if self.illness:
            illness_dicts = [asdict(illness) for illness in self.illness]
            pseudonymed = [
                dict(item, **{"player_name": pseudonym}) for item in illness_dicts
            ]
            return [pd.DataFrame(pseudonymed)[["player_name", "problems", "timestamp"]]]
        return []

    def to_performance_to_dataframe(self, pseudonym):
        if self.performance:
            performance_dicts = [asdict(perf) for perf in self.performance]
            pseudonymed = [
                dict(item, **{"player_name": pseudonym}) for item in performance_dicts
            ]
            return [
                pd.DataFrame(pseudonymed)[
                    [
                        "player_name",
                        "team_performance",
                        "offensive_performance",
                        "defensive_performance",
                        "timestamp",
                    ]
                ]
            ]
        return []


@dataclass(frozen=True)
class Team:
    name: str
    game_performance: pd.DataFrame
    game_ts: pd.Series
    players: Dict[str, SoccerPlayer]

    def get_player(self, player_name: str) -> SoccerPlayer:
        return self.players[player_name]

    def get_players(self, player_names: List[str]) -> List[SoccerPlayer]:
        return [self.get_player(player_name) for player_name in player_names]


def get_variable_by_date(
    player,
    variable_name: str,
    from_date: str = "01.01.2020",
    until_date: str = "31.12.2021",
):

    check_if_variable_callable(variable_name, player)
    variable = player.__getattribute__(variable_name)
    variable.name = variable_name
    if (from_date not in variable.index) or (until_date not in variable.index):
        raise DateNotInRange
    return variable[
        variable.index.get_loc(from_date) : variable.index.get_loc(until_date)
    ]


def has_numbers(string):
    return any(char.isdigit() for char in string)


def get_player_names(wellness_data) -> List[str]:
    return wellness_data["Fatigue"].columns[1:]


def get_player_data(
    wellness_data: Dict[str, pd.DataFrame], player_name: str
) -> Dict[str, pd.Series]:
    init = {}
    for attribute, data in wellness_data.items():
        init[attribute] = pd.Series(data[player_name]).set_axis(
            data[f"{attribute} Data"], axis=0
        )
    return init


def clean_duration_of_sleep(sleep_duration_ts: pd.Series) -> pd.Series:
    """High numbers are potentially in minutes and not hours --> divide by 60 if higher than x"""
    return sleep_duration_ts.apply(lambda x: x / 60 if x > 24 else x)


def get_player_game_performance(
    player_performance: pd.DataFrame, player_name: str
) -> Union[List, List[Performance]]:
    players = set(player_performance["Player"])
    if player_name in players:
        players_perf = []
        for _, performance in player_performance.loc[
            player_performance["Player"] == player_name
        ].iterrows():
            players_perf.append(
                Performance(
                    player_name,
                    performance["Team Overall Performance"],
                    performance["Individual Offensive Performance"],
                    performance["Individual Defensive Performance"],
                    performance["Date"],
                )
            )
        return players_perf
    return []


def get_player_injuries(
    player_injuries: pd.DataFrame, player_name: str
) -> Union[List, List[Injury]]:
    injured_players = set(player_injuries["Player"])
    if player_name in injured_players:
        players_injuries = []
        for _, injury in player_injuries.loc[
            player_injuries["Player"] == player_name
        ].iterrows():
            players_injuries.append(
                Injury(player_name, injury["Injuries"], injury["Date"])
            )
        return players_injuries
    return []


def get_player_illness(
    player_illness: pd.DataFrame, player_name: str
) -> Union[List, List[Injury]]:
    injured_players = set(player_illness["Player"])
    if player_name in injured_players:
        players_illness = []
        for _, injury in player_illness.loc[
            player_illness["Player"] == player_name
        ].iterrows():
            players_illness.append(
                Illness(player_name, injury["Problems"], injury["Date"])
            )
        return players_illness
    return []


def create_game_ts(time_index: pd.Index, game_performance: pd.DataFrame):
    binary_game_timeseries = {
        time: (
            0
            if time not in game_performance["Date"].tolist()
            else game_performance.loc[game_performance["Date"] == time][
                "Team Overall Performance"
            ].iat[0]
        )
        for time in time_index.tolist()
    }
    return pd.Series(
        binary_game_timeseries.values(), index=binary_game_timeseries.keys()
    )


def create_ts_of_injures(time_index: pd.Index, injuries: List[Injury]):
    """Extract the times when injuries occurred. For now only the time stamps are extracted and
    an injury is binary event. However, there are more information stored and can be extracted, like
    how many, injuries, what is injured and severity."""
    injury_timestamps = [injury.timestamp for injury in injuries]
    binary_injury_timeseries = {
        time: (0 if time not in injury_timestamps else 1)
        for time in time_index.tolist()
    }
    return pd.Series(
        binary_injury_timeseries.values(), index=binary_injury_timeseries.keys()
    )


def initialise_players(
    wellness_sheets: Dict[str, pd.DataFrame],
    player_records: Dict[str, Dict[str, pd.Series]],
    name_mapping: Dict[str, str],
) -> Dict[str, SoccerPlayer]:
    names = [
        name
        for name in list(get_player_names(wellness_sheets))
        if not has_numbers(name)
    ]
    players_injuries = wellness_sheets["Injury"]
    players_illness = wellness_sheets["Illness"]
    players_performance = wellness_sheets["Game Performance"]
    del wellness_sheets["Injury"]
    del wellness_sheets["Illness"]
    del wellness_sheets["Game Performance"]
    inv_map = {v: k for k, v in name_mapping.items()}
    players = {}
    for name in names:
        values = get_player_data(wellness_sheets, name)
        records = player_records[name]
        injuries = get_player_injuries(players_injuries, name)
        performance = get_player_game_performance(players_performance, name)
        illness = get_player_illness(players_illness, name)
        injury_ts = create_ts_of_injures(values["Fatigue"].index, injuries)
        sleep_duration = clean_duration_of_sleep(values["SleepDurH"],)
        date_index = values["Stress"].index
        date_index.name = "Date"
        players[inv_map[name]] = SoccerPlayer(
            inv_map[name],
            records["Daily Load"],
            records["SRPE"],
            records["RPE"],
            records["Duration [min]"],
            records["ATL"],
            records["Weekly Load"],
            records["Monotony"],
            records["Strain"],
            records["Acwr"],
            records["Ctl28"],
            records["Ctl42"],
            values["Fatigue"],
            values["Mood"],
            values["Readiness"],
            sleep_duration,
            values["SleepQuality"],
            values["Soreness"],
            values["Stress"],
            injuries,
            injury_ts,
            illness,
            performance,
        )
    return players


def load_in_workbooks(path_to_file: List[Path]) -> Dict[str, pd.DataFrame]:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        workbooks = []
        for path in path_to_file:
            workbooks.append(pd.read_excel(path, sheet_name=None, engine="openpyxl"))
    merged_dictionaries = defaultdict(list)

    for workbook in workbooks:
        for sheet_name, sheet in workbook.items():
            if sheet_name in wellness_sheets_names:
                merged_dictionaries[sheet_name].append(sheet)
            else:
                merged_dictionaries[sheet_name].append(sheet.iloc[:-1, :])

    return {
        sheet_name: pd.concat(sheet, axis=0, ignore_index=True)
        for sheet_name, sheet in merged_dictionaries.items()
    }


def clean_workbooks(
    workbook: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, Dict[Any, Any]], Dict[str, Any]]:
    player_sheets = {
        name: sheet
        for name, sheet in workbook.items()
        if name not in sheets + ["Illness"]
    }
    wellness_sheets = {
        name: sheet for name, sheet in workbook.items() if name in wellness_sheets_names
    }
    recorded_signals = {}
    for name, sheet in player_sheets.items():
        non_continuous_signals = ["SRPE", "RPE", "Duration [min]"]
        filled_dates = sheet["Date"].ffill()
        dates = sheet["Date"].dropna()
        player_records = {}
        for col_name, column in sheet.iteritems():
            if col_name in non_continuous_signals:
                signal = pd.Series(column)
                signal.index = filled_dates
                player_records[col_name] = signal

            else:
                signal = pd.Series(column.dropna())
                signal.index = dates
                player_records[col_name] = signal
        recorded_signals[name] = player_records
    return recorded_signals, wellness_sheets


def generate_team_data(
    team_name, name_mapping: Dict[str, str], path_to_data: List[Path]
) -> Team:
    raw_workbook = load_in_workbooks(path_to_data)
    game_performance = raw_workbook["Game Performance"]
    recorded_signals, workbook = clean_workbooks(raw_workbook)
    players = initialise_players(workbook, recorded_signals, name_mapping)
    # players = initialise_players(
    #    {k: v for k, v in workbook.items() if k != "Game Performance"}, recorded_signals
    # )

    games_ts = create_game_ts(list(players.values())[0].stress.index, game_performance)
    return Team(team_name, game_performance, games_ts, players)


def generate_teams(
    path_to_teams_files: List[List[Path]], team_names: List[Dict[str, str]]
) -> Dict[str, Team]:
    return {
        team["pseudonym"]: generate_team_data(
            team["team_name"], team["players"], path_to_team
        )
        for team, path_to_team in zip(team_names, path_to_teams_files)
    }
