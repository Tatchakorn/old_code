import pandas as pd

df = pd.read_pickle("./output/table/human_table.pkl")

df = df.astype({"win%": "float", "win%_human": "float", "win%_bot": "float", "bot%": "float", "wb%": "float",
                "ww%": "float", "wb_human%": "float", "wb_bot%": "float", "ww_human%": "float", "ww_bot%": "float"})

# The dictionary to pass to pandas DataFrame
df_dict = {}
# A counter used to add entries to "df_dict"
df_dict_index = 0

# Bot players and Non-bot players table
df_bp = df[df["bot%"] >= 60]
df_nbp = df[df["bot%"] < 60]

# Stat Tables
df.describe().to_pickle("./output/table/stat.pkl")
df_bp.describe().to_pickle("./output/table/bp_stat.pkl")
df_nbp.describe().to_pickle("./output/table/nbp_stat.pkl")

df.describe().to_csv("./output/table/stat.csv")
df_bp.describe().to_csv("./output/table/bp_stat.csv")
df_nbp.describe().to_csv("./output/table/nbp_stat.csv")

n_player = len(df.index)    # Total number of players
n_bp = len(df_bp.index)     # number of bot players
n_nbp = len(df_nbp.index)   # number of non-bot players
p_bp = n_bp/n_player*100    # bot player percentage
p_nbp = n_nbp/n_player*100  # non-bot players percentage

# Info Table
df_dict[df_dict_index] = {"n_player": n_player, "n_bp": n_bp, "n_nbp": n_nbp, "bp%": p_bp, "nbp%": p_nbp}
info_df = pd.DataFrame.from_dict(df_dict, "index")
info_df.to_pickle("./output/table/info_table.pkl")
info_df.to_csv("./output/table/info_table.csv")
