import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "sentiment",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, (
        "train_size + val_size + test_size must equal 1.0"
    )

    df_train, df_temp = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df[label_col],
        random_state=random_state,
    )

    relative_test_size = test_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test_size,
        stratify=df_temp[label_col],
        random_state=random_state,
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
