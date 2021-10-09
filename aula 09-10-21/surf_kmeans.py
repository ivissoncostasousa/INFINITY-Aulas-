import pandas as pd

def classify_board_adequate(board_adequate):
    if board_adequate == 'Very inadequate':
        return 0
    if board_adequate == 'Inadequate':
        return 1
    if board_adequate == 'More or less':
        return 2
    if board_adequate == 'Suitable':
        return 3
    if board_adequate == 'Very suitable':
        return 4


ds = pd.read_csv('df_surf2.csv', sep=',')
print(ds.head(20))
print(ds.info())
print(ds['board_adequate'].unique())
ds['board_adequate'] = ds['board_adequate'].apply(classify_board_adequate)
ds.drop(['surfer_weight_distribution','board_tail_rocker','board_nose_rocker'], axis=1, inplace=True)
print(ds.info())