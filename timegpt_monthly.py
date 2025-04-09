import contextlib
import os
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys
from nixtla import NixtlaClient
from dotenv import load_dotenv

from utils.tools import log_into_csv

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

load_dotenv()
nixtla_client = NixtlaClient(api_key = os.getenv('nixtla_key'), timeout=None)
nixtla_client.validate_api_key()

data_dir = Path('dataset/carbon/res')
results_dir = Path('results/carbon_monthly')
results_dir.mkdir(parents=True, exist_ok=True)

TARGET = 'Price'
DT_NAME = 'Month-Year'
# EXOG_LIST = ['CopperIndex']
# EXOG_LIST = []

df = pd.read_csv(data_dir / 'merged_data.csv')
df[DT_NAME] = pd.to_datetime(df[DT_NAME])
df['unique_id'] = 'carbon' # for TimeGPT-1 processing, required by Nixtla's API - doesn't really do much

SKIP_FINETUNE = True
CROSS_VAL = False
FINETUNED_ID = 'c_default'

TRAIN_LEN = int(0.6 * (len(df)))
VAL_LEN = int(0.2 * (len(df)))
TEST_LEN = int(0.2 * (len(df)))

ctxs = [2, 20]
pred_lens = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
feat_pct = [0, 0.2]
feats_map = {}

# ******** finetune first (prediction length independent) ********
finetuned_model_list = nixtla_client.finetuned_models()
finetuned_model_list = [model.id for model in finetuned_model_list]

if not SKIP_FINETUNE:
    print('Finetuning...')
    for model_id in finetuned_model_list:
        nixtla_client.delete_finetuned_model(model_id)

    finetuned_model_list = []
    for loss_type in ['default', 'mse']: # maybe add more in the future
        for model_type in ['timegpt-1', 'timegpt-1-long-horizon']:
            model_id = nixtla_client.finetune(
                df=df.iloc[:TRAIN_LEN],
                target_col=TARGET,
                time_col=DT_NAME,
                finetune_steps=10,
                finetune_loss=loss_type,  # default is a proprietary function (?)
                model=model_type,
                output_model_id=f'c_{loss_type}{model_type.removeprefix("timegpt-1")}',
            )
            finetuned_model_list.append(model_id)
print('\n[INFO]: Available finetuned model IDs:', finetuned_model_list)


for pct in feat_pct:
    # Load in selected features based on the spearman correlation analysis
    sel_features_df0 = pd.read_excel("dataset/carbon/res/ranked_abs_features_monthly.xlsx")
    sel_feature_len = int(pct * len(sel_features_df0))
    sel_features_df0.sort_values(by="Correlation", ascending=False, inplace=True)
    sel_feature_names = sel_features_df0["Factor"][0:sel_feature_len].tolist()
    sel_feature_names = [val for val in sel_feature_names if 'Historical Price' not in val]

    # to make sure there's no dupes for each pct
    feats_map[(tuple(sel_feature_names))] = pct

for feats, pct in feats_map.items():
    feats = list(feats)
    if len(feats) == 0:
        feats = None # for processing by NixtlaClient
    num_features = 0 if feats is None else len(feats)

    for CTX in ctxs:
        for pl in pred_lens:
            print(f'[INFO]: Forecasting for pl-{pl} with {pct} features')
            print('[INFO]: Selected features:', feats)
            if pl < 14:
                model = 'timegpt-1'
            else:
                model = 'timegpt-1-long-horizon'


            # ******** single forecast ********
            # forecasts = nixtla_client.forecast(
            #     df=df.iloc[-512:],
            #     h=90,
            #     target_col=TARGET,
            #     time_col=DT_NAME,
            #     hist_exog_list=feats,
            #     model=model,
            # )
            # forecasts.to_csv(results_dir / 'test1.csv')
            # exit()

            if CROSS_VAL:
                # ********  zero-shot cross-validation/CV (window-based eval) ********
                # NOTE this doesn't do fixed-context-length predictions
                #      only the very first window is (should be) 512, and increases by 'step_size' for every window
                #      but the good part is that it only makes a single API call
                # NOTE2 forced modification in nixtla_client.py line 2002 and 2010, removing futr_exog usage
                # cv_forecasts = nixtla_client.cross_validation(
                #     df=df.iloc[-TEST_LEN - CTX:],
                #     h=pl,
                #     n_windows=TEST_LEN - pl + 1,
                #     step_size=1,
                #     target_col=TARGET,
                #     time_col=DT_NAME,
                #     hist_exog_list=feats,
                #     model=model
                # )
                # cv_forecasts.to_csv(results_dir / f'TimeGPT_feat{pct}-{num_features}_pl{pl}_zeroshot.csv')
                #
                # log_into_csv(
                #     cv_forecasts.rename(columns={'Price': 'true'}),
                #     name=f'carbon_monthly_{pct}-{num_features}',
                #     stage='zeroshot',
                #     model = 'TimeGPT',
                #     seq_len=-1,
                #     pred_len=pl,
                #     bsz=-1,
                #     log_file_name='Carbon_timegpt',
                #     pred_col_name='TimeGPT'
                # )


                # ******** finetuned CV ********
                finetuned_model_id = FINETUNED_ID + model.removeprefix('timegpt-1')
                assert finetuned_model_id in finetuned_model_list

                cv_forecasts = nixtla_client.cross_validation(
                    df=df.iloc[-TEST_LEN - CTX:],
                    h=pl,
                    n_windows=TEST_LEN - pl + 1,
                    step_size=1,
                    target_col=TARGET,
                    time_col=DT_NAME,
                    finetuned_model_id=finetuned_model_id, # this parameter uses the model type specified in a previous training run
                    hist_exog_list=feats,
                    model=model
                )
                cv_forecasts.to_csv(results_dir / f'TimeGPT_feat{pct}-{num_features}_pl{pl}_finetuned.csv')

                log_into_csv(
                    cv_forecasts.rename(columns={'Price': 'true'}),
                    name=f'carbon_monthly_{pct}-{num_features}',
                    stage='finetuned',
                    model='TimeGPT',
                    seq_len=-1,
                    pred_len=pl,
                    bsz=-1,
                    log_file_name='Carbon_timegpt',
                    pred_col_name='TimeGPT'
                )
            else:
                # ******** repeated single forecasts with fixed context length ********
                num_windows = TEST_LEN - pl + 1
                outputs = []

                for i in tqdm(range(num_windows)):
                    with HiddenPrints():
                        forecasts = nixtla_client.forecast(
                            df=df.iloc[-TEST_LEN - CTX + i:-TEST_LEN + i],
                            h=pl,
                            target_col=TARGET,
                            time_col=DT_NAME,
                            hist_exog_list=feats,
                            model=model,
                            freq='MS',
                        )
                    outputs.append(forecasts)

                outputs = pd.concat(outputs, ignore_index=True)
                outputs = pd.merge(outputs, df[[DT_NAME, 'Price']], on=DT_NAME, how='left')
                outputs.to_csv(results_dir / f'TimeGPT_{pct}-{num_features}_pl{pl}_finetuned.csv')

                log_into_csv(
                    outputs.rename(columns={'Price': 'true'}),
                    name=f'carbon_monthly_{pct}-{num_features}',
                    stage='finetuned',
                    model='TimeGPT-fixed_ctx',
                    seq_len=CTX,
                    pred_len=pl,
                    bsz=-1,
                    log_file_name='Carbon_timegpt',
                    pred_col_name='TimeGPT'
                )

            print('[INFO]: Sleeping...')
            time.sleep(5)