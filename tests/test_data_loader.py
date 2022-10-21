from preprocessing.data_loader import initialise_players

import pandas as pd  # type: ignore
import numpy as np  # type: ignore

test_records_df = pd.DataFrame({
        "Date": [
                "01.01.2000",
                "02.01.2000",
                "03.01.2000",
                "04.01.2000",
                "05.01.2000",
                "06.01.2000",
                "07.01.2000",
                "07.01.2000",
                "09.01.2000",
                "10.01.2000",
                "11.01.2000",
                "12.01.2000",
            ],
        "Daily Load": np.zeros(12),
        "SRPE":np.zeros(12),
        "RPE":np.zeros(12),
        "Duration [min]":np.zeros(12),
        "ATL":np.zeros(12),
        "Weekly Load":np.zeros(12),
        "Monotony":np.zeros(12),
        "Strain":np.zeros(12),
        "Acwr":np.zeros(12),
        "Ctl28":np.zeros(12),
        "Ctl42":np.zeros(12), })

wellness_sheets = {
    "Injury": pd.DataFrame(
        {
            "Date": ["01.01.2000", "02.02.2000", "10.02.2000"],
            "Player": ["A", "B", "C"],
            "Injuries": ["left_knee", "hip", "back"],
        }
    ),
    "Fatigue": pd.DataFrame(
        {
            "Fatigue Data": [
                "01.01.2000",
                "02.01.2000",
                "03.01.2000",
                "04.01.2000",
                "05.01.2000",
                "06.01.2000",
                "07.01.2000",
                "07.01.2000",
                "09.01.2000",
                "10.01.2000",
                "11.01.2000",
                "12.01.2000",
            ],
            "A": [np.nan, np.nan, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "B": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, np.nan, np.nan],
            "C": [np.nan, np.nan, 1, 2, 3, np.nan, np.nan, 1, 2, 3, 4, 5],
            "D": [2, 3, 1, 2, np.nan, 4, np.nan, 1, 2, 3, np.nan, 5],
        }
    ),
    "Mood": pd.DataFrame(
        {
            "Mood Data": [
                "01.01.2000",
                "02.01.2000",
                "03.01.2000",
                "04.01.2000",
                "05.01.2000",
                "06.01.2000",
                "07.01.2000",
                "07.01.2000",
                "09.01.2000",
                "10.01.2000",
                "11.01.2000",
                "12.01.2000",
            ],
            "A": [np.nan, np.nan, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "B": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, np.nan, np.nan],
            "C": [np.nan, np.nan, 1, 2, 3, np.nan, np.nan, 1, 2, 3, 4, 5],
            "D": [2, 3, 1, 2, np.nan, 4, np.nan, 1, 2, 3, np.nan, 5],
        }
    ),
    "Readiness": pd.DataFrame(
        {
            "Readiness Data": [
                "01.01.2000",
                "02.01.2000",
                "03.01.2000",
                "04.01.2000",
                "05.01.2000",
                "06.01.2000",
                "07.01.2000",
                "07.01.2000",
                "09.01.2000",
                "10.01.2000",
                "11.01.2000",
                "12.01.2000",
            ],
            "A": [np.nan, np.nan, 10, 2, 8, 4, 5, 1, 2, 3, 4, 5],
            "B": [10, 2, 3, 4, 8, 10, 2, 8, 4, 5, np.nan, np.nan],
            "C": [np.nan, np.nan, 10, 2, 3, np.nan, np.nan, 1, 2, 8, 4, 9],
            "D": [2, 3, 10, 8, np.nan, 4, np.nan, 10, 2, 8, np.nan, 5],
        }
    ),
    "SleepDurH": pd.DataFrame(
        {
            "SleepDurH Data": [
                "01.01.2000",
                "02.01.2000",
                "03.01.2000",
                "04.01.2000",
                "05.01.2000",
                "06.01.2000",
                "07.01.2000",
                "07.01.2000",
                "09.01.2000",
                "10.01.2000",
                "11.01.2000",
                "12.01.2000",
            ],
            "A": [np.nan, np.nan, 8, 7, 5, 5, 7, 8, 9, 10, 5, 12],
            "B": [1, 2, 3, 4, 5, 420, 2, 3, 4, 5, np.nan, np.nan],
            "C": [np.nan, np.nan, 1, 20, 3, np.nan, np.nan, 1, 2, 3, 4, 5],
            "D": [2, 3, 420, 2, np.nan, 4, np.nan, 1, 2, 3, np.nan, 5],
        }
    ),
    "SleepQuality": pd.DataFrame(
        {
            "SleepQuality Data": [
                "01.01.2000",
                "02.01.2000",
                "03.01.2000",
                "04.01.2000",
                "05.01.2000",
                "06.01.2000",
                "07.01.2000",
                "07.01.2000",
                "09.01.2000",
                "10.01.2000",
                "11.01.2000",
                "12.01.2000",
            ],
            "A": [np.nan, np.nan, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "B": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, np.nan, np.nan],
            "C": [np.nan, np.nan, 1, 2, 3, np.nan, np.nan, 1, 2, 3, 4, 5],
            "D": [2, 3, 1, 2, np.nan, 4, np.nan, 1, 2, 3, np.nan, 5],
        }
    ),
    "Soreness": pd.DataFrame(
        {
            "Soreness Data": [
                "01.01.2000",
                "02.01.2000",
                "03.01.2000",
                "04.01.2000",
                "05.01.2000",
                "06.01.2000",
                "07.01.2000",
                "07.01.2000",
                "09.01.2000",
                "10.01.2000",
                "11.01.2000",
                "12.01.2000",
            ],
            "A": [np.nan, np.nan, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "B": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, np.nan, np.nan],
            "C": [np.nan, np.nan, 1, 2, 3, np.nan, np.nan, 1, 2, 3, 4, 5],
            "D": [2, 3, 1, 2, np.nan, 4, np.nan, 1, 2, 3, np.nan, 5],
        }
    ),
    "Stress": pd.DataFrame(
        {
            "Stress Data": [
                "01.01.2000",
                "02.01.2000",
                "03.01.2000",
                "04.01.2000",
                "05.01.2000",
                "06.01.2000",
                "07.01.2000",
                "07.01.2000",
                "09.01.2000",
                "10.01.2000",
                "11.01.2000",
                "12.01.2000",
            ],
            "A": [np.nan, np.nan, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "B": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, np.nan, np.nan],
            "C": [np.nan, np.nan, 1, 2, 3, np.nan, np.nan, 1, 2, 3, 4, 5],
            "D": [2, 3, 1, 2, np.nan, 4, np.nan, 1, 2, 3, np.nan, 5],
        }
    ),
}
record_sheets = {
    "A": test_records_df,
    "B": test_records_df,
    "C": test_records_df,
    "D": test_records_df,
}


def test_initialise_players():
    """More tests should be implemented."""
    output = initialise_players(wellness_sheets, record_sheets)
    assert output["0"].name == "A"
    assert output["3"].name == "D"
    assert output["1"].stress.equals(
        pd.Series(
            [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, np.nan, np.nan],
            name="B",
            index=[
                "01.01.2000",
                "02.01.2000",
                "03.01.2000",
                "04.01.2000",
                "05.01.2000",
                "06.01.2000",
                "07.01.2000",
                "07.01.2000",
                "09.01.2000",
                "10.01.2000",
                "11.01.2000",
                "12.01.2000",
            ],
        )
    )
    assert output["2"].strain.equals(pd.Series(
        np.zeros(12),
        name="Strain",
    ))
    assert output["3"].srpe.equals(pd.Series(
        np.zeros(12),
        name="SRPE",
    ))
    assert output["2"].strain.equals(pd.Series(
        np.zeros(12),
        name="Strain",
    ))
    assert output["3"].sleep_duration.equals(
        pd.Series(
            [2, 3, 7, 2, np.nan, 4, np.nan, 1, 2, 3, np.nan, 5],
            name="D",
            index=[
                "01.01.2000",
                "02.01.2000",
                "03.01.2000",
                "04.01.2000",
                "05.01.2000",
                "06.01.2000",
                "07.01.2000",
                "07.01.2000",
                "09.01.2000",
                "10.01.2000",
                "11.01.2000",
                "12.01.2000",
            ],
        )
    )
