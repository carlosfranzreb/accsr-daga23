"""Transform a Tensorboard log to a Pandas dataframe."""


from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from os import listdir, mkdir


def dump_tb_as_df(log_folder):
    folder = f"{log_folder}/dataframes"
    try:
        mkdir(folder)
    except FileExistsError:
        print("Folder already exists. Dataframes will be overwritten.")
    val, train = list(), list()
    subfolders = ["."] + listdir(log_folder)
    if "checkpoints" in subfolders:
        subfolders.remove("checkpoints")
    for subfolder in subfolders:
        ea = event_accumulator.EventAccumulator(f"{log_folder}/{subfolder}")
        ea.Reload()
        mnames = ea.Tags()["scalars"]
        for n in mnames:
            if n in ["hp_metric", "lr", "epoch"]:
                continue
            else:
                added = False
                for metric in ["AC Loss", "ASR Loss", "Total Loss"]:
                    if metric in n:
                        train.append(get_scalars(subfolder, n, ea))
                        added = True
                        break
                if not added:
                    val.append(get_scalars(subfolder, n, ea))
    merge_dataframes(train, f"{folder}/train.pkl")
    merge_dataframes(val, f"{folder}/val.pkl")


def get_scalars(subfolder, n, ea):
    """Extract scalars from file and transform them to a dataframe. Remove the
    timestamp column before returning it. If the file comes from a subfolder that
    is not the log's root, name the column as the subfolder. Subfolders only store
    one metric each, whereas the root file stores LR, training losses and others."""
    col_name = n if subfolder == "." else subfolder
    df = pd.DataFrame(ea.Scalars(n), columns=["timestamp", "step", col_name])
    df.drop(columns="timestamp", inplace=True)
    df.drop(columns="step", inplace=True)
    return df


def merge_dataframes(df_list, dump_file):
    """Merge the list of dataframes into a single dataframe and dump them in the
    given file."""
    if "val" in dump_file:
        for df in df_list:
            if len(df) != 50:
                print(df.columns, len(df.columns))
    df = pd.concat(df_list, axis=1)
    pd.to_pickle(df, dump_file)
