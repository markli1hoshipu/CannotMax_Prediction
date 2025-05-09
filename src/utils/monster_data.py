import pandas as pd
import csv

def sort_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    id_col = df.columns[0]
    df_sorted = df.sort_values(by = id_col)
    df_sorted.to_csv(output_file, index = False)

def get_monster_info(monster_id):
    df = pd.read_csv("data/monsters.csv")
    line = df.loc[df['ID'] == monster_id].squeeze()
    return f"{line['名称']}: {'物理' if line['法伤'] != 1 else '法术'}攻击{line['攻击力']}, 攻击间隔{line['攻击间隔']}, '范围:' {line['攻击范围半径'] if not pd.isna(line['攻击范围半径']) else '近战'}, \
血量{line['生命值']}, 防御{line['防御力']}, 法抗{line['法术抗性']}, 移速{line['移动速度']}\n\
{line['特殊能力']+'\n' if not pd.isna(line['特殊能力']) else ''}\n"

def similar(a,b):
    l, r = [], []
    for i in range(len(a)):
        if a[i] != '0' and b[i] != 0:
            if i<len(a)/2:
                l.append(a[i])
            else:
                r.append(a[i])
        elif a[i] != '0' or b[i] != 0:
            return False, []
    if not l or not r:
        return False, []
    return True, '/'.join(l)+':'+'/'.join(r)
        

def find_identical_rows(key):
    ret = []
    with open("data/arknights.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            tf, val = similar(row[:-1], key)
            if tf:  ret.append(val+"   "+ ("左赢" if row[-1] == "L" else "右赢"))
    return ret

