from sgfmill import sgf
import pandas as pd
import os

path = r"./go_data/seperate_02"

# Next file
# path = r"C:\Users\AILAB\Desktop\go_data\newGoData"

df_dict = {}  # The dictionary to pass to pandas DataFrame
df_dict_index = 0  # A counter used to add entries to "df_dict"

# get all directories from a directory
directory_list = [os.path.join(path, i) for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]
print("-------------- Run ---------------")

for directory in directory_list:  # <<< read files in current folder <<<
    print("now running:", directory)
    for sgf_file in os.listdir(directory):
        if sgf_file.endswith(".sgf"):
            try:
                with open(directory + '\\' + sgf_file, "rb") as f:
                    game = sgf.Sgf_game.from_bytes(f.read())
            except:
                print("Read Error: at", sgf_file)

            # Get game attributes
            b_player = game.get_player_name("b")
            w_player = game.get_player_name("w")

            # Ignore games without players' names
            if b_player is None or w_player is None:
                continue

            try:
                root_node = game.get_root()
                b_rank = root_node.get("BR")
                w_rank = root_node.get("WR")
                result = root_node.get("RE")
            except:
                print("get identifier error!", sgf_file)
                continue

            # List of the game sequence
            try:
                game_sequence = [node.get_move() for node in game.get_main_sequence()]
                game_sequence.pop(0)  # get rid of the first none sequence
            except ValueError:
                print("game_sequence Error at:", sgf_file)
                continue

            # Append each attribute in a dictionary
            df_dict[df_dict_index] = {"Black player": b_player, "Black Rank": b_rank, "White player": w_player,
                                      "White Rank": w_rank, "Result": result, "Record": game_sequence}
            df_dict_index += 1



# Save the Tabular Format to pickle
# df = pd.DataFrame(Go_table)
# Don't forget to update the number
print(df_dict)
go_table_df = pd.DataFrame.from_dict(df_dict, "index")
go_table_df.to_pickle("./output/raw/raw_02.pkl")

print("--------------Done----------------")
