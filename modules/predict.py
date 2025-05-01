import dill
import os
import pandas as pd
import json
import glob


def predict():
    path = os.environ.get('PROJECT_PATH', '.')

    model_files = glob.glob(f'{path}/data/models/cars_pipe_*.pkl')
    latest_model_path = max(model_files, key=os.path.getctime)

    with open(latest_model_path, 'rb') as file:
        model = dill.load(file)

    folder_path = f'{path}/data/test'
    dfs = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame([data])
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    preds_df = pd.DataFrame({
        'id': final_df['id'],
        'prediction': model.predict(final_df)
    })

    preds_df.to_csv(f'{path}/data/predictions/prediction.csv', index=False)


if __name__ == '__main__':
    predict()
