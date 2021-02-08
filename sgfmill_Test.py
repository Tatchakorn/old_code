from sgfmill import sgf, boards, ascii_boards
import os

# # print 19x19 board
# b = boards.Board(19)
# b.play(16, 15, 'b')
# b.play(3, 3, 'w')
# print(ascii_boards.render_board(b))
path = r".\go_data"
print(path)
for sgf_file in [i for i in os.listdir('.\go_data') if i.endswith('.sgf')]:
    with open('.\go_data\\' + sgf_file, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    # Get game attributes
    b_player = game.get_player_name("b")
    w_player = game.get_player_name("w")

    # Ignore games without players' names
    if b_player is None or w_player is None:
        continue

    board_size = game.get_size()
    root_node = game.get_root()
    b_rank = root_node.get("BR")
    w_rank = root_node.get("WR")
    result = root_node.get("RE")

    print('file name:', sgf_file)
    print("Black player:\t", b_player)
    print("White player:\t", w_player)
    print("Black Rank:\t", b_rank)
    print("White Rank:\t", w_rank)
    print("result:\t", result)

    # List of game sequence
    game_sequence = [node.get_move() for node in game.get_main_sequence() if node.get_move()[1] is not None] # get rid of the first none sequence
    print(game_sequence)


