import pandas as pd
import glob
import re


def split_to_groups(list_, n_group):
    assert (len(list_) / n_group) < 998, "Can't split to {} groups".format(len(l) / n)

    if list_ == []:
        return []

    else:
        f = [list_[:n_group]]
        f.extend(split_to_groups(list_[n_group:], n_group))
        return f


file_path = glob.glob(r"./kluay_csv/*.csv")

for file in file_path:

    dict = {}
    dict_index = 0
    print(file[12:-4])

    df = pd.read_csv(file)
    num_table = re.findall("\d+\.\d+", str(df["Length"].to_string()))
    num_table.pop(-1)

    n = split_to_groups(num_table, 3)

    for n_1, n_2, n_3 in n:
        dict[dict_index] = {"latitude": n_1, "longtitude": n_2, "height": n_3}
        dict_index = dict_index + 1

    df = pd.DataFrame.from_dict(dict, "index")

    path = f'.//kluay_csv//{file[12:-4]}_LLL.csv'
    df.to_csv(path)
