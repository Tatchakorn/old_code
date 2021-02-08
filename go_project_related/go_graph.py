import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read All player table
df = pd.read_pickle("./output/human_table.pkl")

# Convert all columns with object type to float
df = df.astype({"win%": "float", "win%_human": "float", "win%_bot": "float", "bot%": "float", "wb%": "float",
                "ww%": "float", "wb_human%": "float", "wb_bot%": "float", "ww_human%": "float", "ww_bot%": "float"})

df_bp = df[df["bot%"] >= 60]  # bot players
df_nbp = df[df["bot%"] < 60]  # non-bot players

plt.style.use("seaborn")  # Graph style "fivethirtyeight" or "seaborn"

# \\\--- All players Graph ----- \\\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\

# """

# Number of games and win%

df = df.sort_values(by="n_game", ascending=True, na_position='first')

ax = df.plot(kind="scatter", x="n_game", y="win%_bot", color="r", label="win%_bot", title="All players")
df.plot(kind="scatter", x="n_game", y="win%_human", color="g", label="win%_human", ax=ax)
df.plot(kind="scatter", x="n_game", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df["n_game"], y=df["win%_bot"], color="r")
sns.regplot(x=df["n_game"], y=df["win%_human"], color="g")
sns.regplot(x=df["n_game"], y=df["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("number of games")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_a01.png")
plt.cla()

# Number of games w/ human and win%
df = df.sort_values(by="n_bot", ascending=True, na_position='first')

ax = df.plot(kind="scatter", x="n_bot", y="win%_bot", color="r", label="win%_bot", title="All players")
df.plot(kind="scatter", x="n_bot", y="win%_human", color="g", label="win%_human", ax=ax)
df.plot(kind="scatter", x="n_bot", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df["n_bot"], y=df["win%_bot"], color="r")
sns.regplot(x=df["n_bot"], y=df["win%_human"], color="g")
sns.regplot(x=df["n_bot"], y=df["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("number of games w/ bot")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_a02.png")
plt.cla()

# Number of games w/ bot and win%
df = df.sort_values(by='n_human', ascending=True, na_position='first')
ax = df.plot(kind="scatter", x="n_human", y="win%_bot", color="r", label="win%_bot", title="All players")
df.plot(kind="scatter", x="n_human", y="win%_human", color="g", label="win%_human", ax=ax)
df.plot(kind="scatter", x="n_human", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df["n_human"], y=df["win%_bot"], color="r")
sns.regplot(x=df["n_human"], y=df["win%_human"], color="g")
sns.regplot(x=df["n_human"], y=df["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("number of games w/ human")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_a03.png")
plt.cla()
# """

# bot% and win%
df = df.sort_values(by="bot%", ascending=True, na_position='first')
ax = df.plot(kind="scatter", x="bot%", y="win%_bot", color="r", label="win%_bot", title="All players")
df.plot(kind="scatter", x="bot%", y="win%_human", color="g", label="win%_human", ax=ax)
df.plot(kind="scatter", x="bot%", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df["bot%"], y=df["win%_bot"], color="r")
sns.regplot(x=df["bot%"], y=df["win%_human"], color="g")
sns.regplot(x=df["bot%"], y=df["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("bot%")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_a04.png")
plt.cla()

# \\\--- BP players Graph ---\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\
# """
# Number of games and win%
df_bp = df_bp.sort_values(by="n_game", ascending=True, na_position="first")
ax = df_bp.plot(kind="scatter", x="n_game", y="win%_bot", color="r", label="win%_bot", title="BP players")
df_bp.plot(kind="scatter", x="n_game", y="win%_human", color="g", label="win%_human", ax=ax)
df_bp.plot(kind="scatter", x="n_game", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df_bp["n_game"], y=df_bp["win%_bot"], color="r")
sns.regplot(x=df_bp["n_game"], y=df_bp["win%_human"], color="g")
sns.regplot(x=df_bp["n_game"], y=df_bp["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("number of games")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_b01.png")
plt.cla()

# Number of games w/ bot and win%
df_bp = df_bp.sort_values(by='n_bot', ascending=True, na_position="first")
ax = df_bp.plot(kind="scatter", x="n_bot", y="win%_bot", color="r", label="win%_bot", title="BP players")
df_bp.plot(kind="scatter", x="n_bot", y="win%_human", color="g", label="win%_human", ax=ax)
df_bp.plot(kind="scatter", x="n_bot", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df_bp["n_bot"], y=df_bp["win%_bot"], color="r")
sns.regplot(x=df_bp["n_bot"], y=df_bp["win%_human"], color="g")
sns.regplot(x=df_bp["n_bot"], y=df_bp["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("number of games w/ bot")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_b02.png")
plt.cla()

# Number of games w/ human and win%
df_bp = df_bp.sort_values(by='n_human', ascending=True, na_position="first")
ax = df_bp.plot(kind="scatter", x="n_human", y="win%_bot", color="r", label="win%_bot", title="BP players")
df_bp.plot(kind="scatter", x="n_human", y="win%_human", color="g", label="win%_human", ax=ax)
df_bp.plot(kind="scatter", x="n_human", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df_bp["n_human"], y=df_bp["win%_bot"], color="r")
sns.regplot(x=df_bp["n_human"], y=df_bp["win%_human"], color="g")
sns.regplot(x=df_bp["n_human"], y=df_bp["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("number of games w/ human")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_b03.png")
plt.cla()

# """

# bot% and win%
df_bp = df_bp.sort_values(by="bot%", ascending=True, na_position='first')
ax = df_bp.plot(kind="scatter", x="bot%", y="win%_bot", color="r", label="win%_bot", title="BP players")
df_bp.plot(kind="scatter", x="bot%", y="win%_human", color="g", label="win%_human", ax=ax)
df_bp.plot(kind="scatter", x="bot%", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df_bp["bot%"], y=df_bp["win%_bot"], color="r")
sns.regplot(x=df_bp["bot%"], y=df_bp["win%_human"], color="g")
sns.regplot(x=df_bp["bot%"], y=df_bp["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("bot%")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_b04.png")
plt.cla()


# \\\--- NBP players Graph ---\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\ ----- \\\\\
# """
# Number of games and win%
df_nbp = df_nbp.sort_values(by="n_game", ascending=True, na_position="first")
ax = df_nbp.plot(kind="scatter", x="n_game", y="win%_bot", color="r", label="win%_bot", title="NBP players")
df_nbp.plot(kind="scatter", x="n_game", y="win%_human", color="g", label="win%_human", ax=ax)
df_nbp.plot(kind="scatter", x="n_game", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df_nbp["n_game"], y=df_nbp["win%_bot"], color="r")
sns.regplot(x=df_nbp["n_game"], y=df_nbp["win%_human"], color="g")
sns.regplot(x=df_nbp["n_game"], y=df_nbp["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("number of games")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_nb01.png")
plt.cla()

# Number of games w/ bot and win%
df_nbp = df_nbp.sort_values(by='n_bot', ascending=True, na_position="first")
ax = df_nbp.plot(kind="scatter", x="n_bot", y="win%_bot", color="r", label="win%_bot", title="NBP players")
df_nbp.plot(kind="scatter", x="n_bot", y="win%_human", color="g", label="win%_human", ax=ax)
df_nbp.plot(kind="scatter", x="n_bot", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df_nbp["n_bot"], y=df_nbp["win%_bot"], color="r")
sns.regplot(x=df_nbp["n_bot"], y=df_nbp["win%_human"], color="g")
sns.regplot(x=df_nbp["n_bot"], y=df_nbp["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("number of games w/ bot")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_nb02.png")
plt.cla()

# Number of games w/ human and win%
df_nbp = df_nbp.sort_values(by='n_human', ascending=True, na_position="first")
ax = df_nbp.plot(kind="scatter", x="n_human", y="win%_bot", color="r", label="win%_bot", title="NBP players")
df_nbp.plot(kind="scatter", x="n_human", y="win%_human", color="g", label="win%_human", ax=ax)
df_nbp.plot(kind="scatter", x="n_human", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df_nbp["n_human"], y=df_nbp["win%_bot"], color="r")
sns.regplot(x=df_nbp["n_human"], y=df_nbp["win%_human"], color="g")
sns.regplot(x=df_nbp["n_human"], y=df_nbp["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("number of games w/ human")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_nb03.png")
plt.cla()
# """
# bot% and win%
df_nbp = df_nbp.sort_values(by="bot%", ascending=True, na_position='first')
ax = df_nbp.plot(kind="scatter", x="bot%", y="win%_bot", color="r", label="win%_bot", title="NBP players")
df_nbp.plot(kind="scatter", x="bot%", y="win%_human", color="g", label="win%_human", ax=ax)
df_nbp.plot(kind="scatter", x="bot%", y="win%", color="b", label="win%", ax=ax)
# ----- Linear Regression ----- #
sns.regplot(x=df_nbp["bot%"], y=df_nbp["win%_bot"], color="r")
sns.regplot(x=df_nbp["bot%"], y=df_nbp["win%_human"], color="g")
sns.regplot(x=df_nbp["bot%"], y=df_nbp["win%"], color="b")
# ----- Linear Regression ----- #
ax.set_xlabel("bot%")
ax.set_ylabel("win rate")
plt.ylim(0, 100)
plt.savefig("./output/graph/graph_nb04.png")
plt.cla()

