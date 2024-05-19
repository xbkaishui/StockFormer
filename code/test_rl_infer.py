import os
import time

from MySAC.models.DRLAgent import DRLAgent
from loguru import logger 


def test_rl_infer():
    cur_dir = os.path.dirname(__file__)
    model_name = 'StockFormer'
    version = 'CSI'
    model_path = os.path.join(cur_dir, 'trained_models', version, model_name, 'model30000.zip')
    start = time.time()
    results = DRLAgent.DRL_prediction_load_from_file(model_name='maesac',environment=test_trade_gym, cwd=model_path)
    end = time.time()
    print("Test time: %.3f"%(end-start))

    df_root = 'results/df_print/'+version+model_name
    os.makedirs(df_root, exist_ok=True)
    assets_his, df_actions = results[1], results[2]
    df_actions.to_csv(df_root+'df_actions_test.csv')
    assets_his.to_csv(df_root+'df_assets_his_test.csv')


if __name__ == '__main__':
    test_rl_infer()