import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit


### replace tag
replace_dict = {
    "향": "향/냄새",
    "용량": "용량/사이즈",
    "용량/개수": "용량/사이즈",
    "사이즈/두께": "용량/사이즈",
    "보습력/수분감": "보습력/수분감/쿨링감",
    "청량감/쿨링감": "보습력/수분감/쿨링감",
    "염색력": "발색력",
    "세팅력/고정력": "스타일링효과",
    "클렌징/제거력": "세정력",
    "그립감": "용기",
    "지속력": "지속력/유지력"
}

def modify_aspects(df):
    for i, row in tqdm(df.iterrows(), total=len(df)):
        for old, new in replace_dict.items():
            if row['Aspect'] in [f'B-{old}', f'I-{old}']:
                df.loc[i, 'Aspect'] = df.loc[i, 'Aspect'].replace(f'-{old}', f'-{new}')
                break
    return df

dir = "../Dataset/Aspect"
df = pd.read_csv(f"{dir}/cosmetic_aspect.csv")
df = modify_aspects(df)


### divide set
test_split = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=42).split(df, groups=df['Index'])
train_val_idxs, test_idxs = next(test_split)

train_val = df.iloc[train_val_idxs]
test = df.iloc[test_idxs]

val_split = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=42).split(train_val, groups=train_val['Index'])
train_idxs, val_idxs = next(val_split)
train = train_val.iloc[train_idxs]
val = train_val.iloc[val_idxs]

train.to_csv(f"{dir}/cosmetic_aspect_train.csv", encoding='utf-8', index=False)
val.to_csv(f"{dir}/cosmetic_aspect_valid.csv", encoding='utf-8', index=False)
test.to_csv(f"{dir}/cosmetic_aspect_test.csv", encoding='utf-8', index=False)