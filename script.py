import pandas as pd, numpy as np, lightgbm as lgb, warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# ── Load ───────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv', low_memory=False)
test  = pd.read_csv('test.csv',  low_memory=False)

# ── Feature Engineering ────────────────────────────────────────────────
def engineer(df):
    df = df.copy()
    df['lease_remain']       = 99 - df['hdb_age']
    df['area_per_dwelling']  = df['floor_area_sqm'] / (df['total_dwelling_units'] + 1)
    df['total_hawker_stalls'] = df['hawker_food_stalls'] + df['hawker_market_stalls']

    # Binary Y/N flags → 0/1
    for col in ['residential','commercial','market_hawker',
                'multistorey_carpark','precinct_pavilion']:
        df[col] = (df[col] == 'Y').astype(int)

    # NaN counts mean "none nearby" → fill with 0
    for col in ['Mall_Within_500m','Mall_Within_1km','Mall_Within_2km',
                'Hawker_Within_500m','Hawker_Within_1km','Hawker_Within_2km']:
        df[col] = df[col].fillna(0)

    df['Mall_Nearest_Distance'] = df['Mall_Nearest_Distance'].fillna(
        df['Mall_Nearest_Distance'].median())

    # Encode categoricals (LightGBM handles these natively)
    for col in ['town','flat_type','flat_model','full_flat_type','planning_area',
                'mrt_name','pri_sch_name','sec_sch_name','bus_stop_name','street_name']:
        df[col] = df[col].astype('category')
    return df

train = engineer(train)
test  = engineer(test)

# ── Feature Selection ──────────────────────────────────────────────────
drop_cols = ['id','Tranc_YearMonth','block','address','postal',
             'storey_range','lower','upper','mid','resale_price',
             'mrt_latitude','mrt_longitude','bus_stop_latitude','bus_stop_longitude',
             'pri_sch_latitude','pri_sch_longitude','sec_sch_latitude','sec_sch_longitude']

feat = [c for c in train.columns if c not in drop_cols]
cats = [c for c in feat if train[c].dtype.name == 'category']

X = train[feat];  y = train['resale_price'];  Xt = test[feat]

# ── Train / Validation Split ───────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# ── LightGBM ───────────────────────────────────────────────────────────
params = dict(
    objective='regression', metric='rmse',
    learning_rate=0.05, num_leaves=255,
    min_child_samples=20, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1,
    verbose=-1, n_jobs=-1
)

dtrain = lgb.Dataset(X_tr,  label=y_tr,  categorical_feature=cats, free_raw_data=False)
dval   = lgb.Dataset(X_val, label=y_val, categorical_feature=cats, free_raw_data=False)

model = lgb.train(
    params, dtrain, num_boost_round=1500,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(100)]
)

# ── Evaluate & Predict ─────────────────────────────────────────────────
val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f'Val RMSE: {rmse:,.2f}')   # → 21,978

test_pred = model.predict(Xt)
sub = pd.DataFrame({'Id': test['id'], 'resale_price': test_pred})
sub.to_csv('submission.csv', index=False)
