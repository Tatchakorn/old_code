import pandas as pd
import numpy as np

# The dictionary to pass to pandas DataFrame
df_dict = {}
# A counter used to add entries to "df_dict"
df_dict_index = 0

df = pd.read_pickle('./output/Go_Table.pkl')

# Union sets of black and white player to get all players' names
b_player_list = {player for player in df["Black player"]}
w_player_list = {player for player in df["White player"]}
all_player = b_player_list.union(w_player_list)

# Bot names begin with "GoTrend"
bot_player = {i for i in all_player if "Cho" in i}
human_player = all_player - bot_player

for player_name in human_player:
    # The rows that the player's name appear
    sub_df = df.loc[(df["Black player"] == player_name) | (df["White player"] == player_name)]

    # Count the total  number of games by the number of rows of sub_df
    n_game = len(sub_df.index)

    # Count everything
    count_bot = 0
    count_win = 0
    count_win_bot = 0

    # Count win as black or white
    count_black = 0
    count_wb = 0
    count_wb_bot = 0

    for index, row in sub_df.iterrows():
        if "Cho" in row["Black player"] or "Cho" in row["White player"]:
            count_bot += 1
            play_with_bot = True
        else:
            play_with_bot = False

        if player_name == row["Black player"]:
            count_black += 1
            if "B" in row["Result"]:
                count_win += 1
                count_wb += 1
                if play_with_bot:
                    count_win_bot += 1
                    count_wb_bot += 1

        if player_name == row["White player"] and "W" in row["Result"]:
            count_win += 1
            if play_with_bot:
                count_win_bot += 1

    n_human = n_game - count_bot
    n_win_human = count_win - count_win_bot
    n_lose = n_game - count_win
    n_white = n_game - count_black
    n_ww = count_win - count_wb
    n_ww_bot = count_win_bot - count_wb_bot
    n_wb_human = count_wb - count_wb_bot
    n_ww_human = n_ww - n_ww_bot

    try:
        win_rate = "{:.2f}".format(count_win/n_game*100)
    except ZeroDivisionError:
        win_rate = np.nan

    try:
        win_rate_bot = "{:.2f}".format(count_win_bot / count_bot*100)
    except ZeroDivisionError:
        win_rate_bot = np.nan

    try:
        win_rate_human = "{:.2f}".format(n_win_human/n_human*100)
    except ZeroDivisionError:
        win_rate_human = np.nan

    try:
        p_bot = "{:.2f}".format(count_bot/n_game*100)
    except ZeroDivisionError:
        p_bot = np.nan

    try:
        p_wb = "{:.2f}".format(count_wb/count_win*100)
    except ZeroDivisionError:
        p_wb = np.nan

    try:
        p_ww = "{:.2f}".format(n_ww/count_win*100)
    except ZeroDivisionError:
        p_ww = np.nan

    try:
        p_wb_human = "{:.2f}".format(n_wb_human/count_wb*100)
    except ZeroDivisionError:
        p_wb_human = np.nan

    try:
        p_ww_human = "{:.2f}".format(n_ww_human/n_ww*100)
    except ZeroDivisionError:
        p_ww_human = np.nan

    try:
        p_wb_bot = "{:.2f}".format(count_wb_bot/count_wb*100)
    except ZeroDivisionError:
        p_wb_bot = np.nan

    try:
        p_ww_bot = "{:.2f}".format(n_ww_bot/n_ww*100)
    except ZeroDivisionError:
        p_ww_bot = np.nan

    # Add data to a dictionary
    df_dict[df_dict_index] = {"name": player_name, "n_game": n_game, "n_win": count_win, "n_lose": n_lose,
                              "n_black": count_black, "n_white": n_white, "n_human": n_human, "n_win_human": n_win_human,
                              "n_bot": count_bot, "n_win_bot": count_win_bot,
                              "n_wb": count_wb, "n_ww": n_ww, "n_wb_human": n_wb_human, "n_ww_human": n_ww_human,
                              "n_wb_bot": count_wb_bot, "n_ww_bot": n_ww_bot, "win%": win_rate, "win%_human": win_rate_human,
                              "win%_bot": win_rate_bot, "bot%": p_bot, "wb%": p_wb, "ww%": p_ww,
                              "wb_human%": p_wb_human, "wb_bot%": p_wb_bot, "ww_human%": p_ww_human, "ww_bot%": p_ww_bot}
    df_dict_index += 1

human_df = pd.DataFrame.from_dict(df_dict, "index")
print(human_df.to_string())
human_df.to_pickle("./output/table/human_table.pkl")

# /////---------- Parameters Description ----------/////
# "n_game": Total number of games
# "n_win": Number of games the player's won
# "n_lose": Number of games the player's lost
# "n_black": Number of games played as black
# "n_white": Number of games played as white
# "n_human": Number of games played with other players
# "n_win_human": Number of games won against other players
# "n_bot": Number of games played with bots
# "n_win_bot": Number of games won against bots
# "n_wb": Number of games won as black
# "n_ww": Number of games won as white
# "n_wb_human": Number of games won as black against other players
# "n_ww_human": Number of games won as white against other players
# "n_wb_bot": Number of games won as black against bots
# "n_ww_bot": Number of games won as white against bots
# "win%": Win rate
# "win%_human": Win rate against human
# "win%_bot": Win rate against bot
# "bot%": Percentage of bot games
# "wb%": Win rate as black
# "ww%": Win rate as white
# "wb_human%": Win rate against other players as black
# "wb_bot%": Win rate against bots as black
# "ww_human%": Win rate against other players as white
# "ww_bot%": Win rate against bots as white

