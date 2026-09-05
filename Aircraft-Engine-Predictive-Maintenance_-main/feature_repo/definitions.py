"""
Feast feature definitions for the aircraft engine predictive maintenance
project.

Defines the `engine` entity and two families of feature views, both
partitioned identically (train/val/test) using the engine-wise split locked
in by data_validation.py's engine_split_manifest.json — no stage reading from
this feature store can leak an engine across partitions:

  - engine_raw_sensors_{train,val,test}      raw validated sensor readings
                                              (data_validation.py)
  - engine_engineered_features_{train,val,test}  post-EDA engineered features:
                                              constant/redundant columns
                                              dropped, skew-transformed,
                                              rolling-window stats, RUL target
                                              (feature_engineering.py)

The engineered views infer their schema directly from the parquet files
written by build_feature_store_sources.py instead of a hardcoded field list,
because which columns exist (rolling windows, which sensors got dropped,
whether RUL is present) is itself a product of the upstream EDA/feature
engineering reports and can change as those evolve. Run
`python build_feature_store_sources.py` from the repo root before `feast
apply` so these files exist.
"""

from datetime import timedelta
from pathlib import Path

import pyarrow.parquet as pq
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String, ValueType

REPO_DIR = Path(__file__).resolve().parent

engine = Entity(
    name="engine",
    join_keys=["id"],
    value_type=ValueType.INT64,
    description="Aircraft engine unit from the NASA C-MAPSS FD001 dataset",
)

_NON_FEATURE_COLUMNS = {"id", "event_timestamp", "created_timestamp"}
_PANDAS_TO_FEAST_TYPE = {
    "int64": Int64,
    "int32": Int64,
    "float64": Float32,
    "float32": Float32,
}


def _infer_schema(parquet_path: Path) -> list[Field]:
    """Builds a Feast Field list from a parquet file's actual columns/dtypes."""
    schema = pq.read_schema(parquet_path)
    fields = []
    for name in schema.names:
        if name in _NON_FEATURE_COLUMNS:
            continue
        pandas_dtype = schema.field(name).type.to_pandas_dtype().__name__
        feast_type = _PANDAS_TO_FEAST_TYPE.get(pandas_dtype, String)
        fields.append(Field(name=name, dtype=feast_type))
    return fields

train_source = FileSource(
    name="engine_train_source",
    path="data/engine_train.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

val_source = FileSource(
    name="engine_val_source",
    path="data/engine_val.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

test_source = FileSource(
    name="engine_test_source",
    path="data/engine_test.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

_SENSOR_AND_SETTING_FIELDS = [
    Field(name=f"setting{i}", dtype=Float32) for i in range(1, 4)
] + [
    Field(name=f"s{i}", dtype=Float32) for i in range(1, 22)
]

engine_raw_sensors_train = FeatureView(
    name="engine_raw_sensors_train",
    entities=[engine],
    ttl=timedelta(days=3650),
    schema=[Field(name="cycle", dtype=Int64), *_SENSOR_AND_SETTING_FIELDS],
    online=True,
    source=train_source,
    tags={"split": "train", "stage": "data_validation"},
)

engine_raw_sensors_val = FeatureView(
    name="engine_raw_sensors_val",
    entities=[engine],
    ttl=timedelta(days=3650),
    schema=[Field(name="cycle", dtype=Int64), *_SENSOR_AND_SETTING_FIELDS],
    online=True,
    source=val_source,
    tags={"split": "val", "stage": "data_validation"},
)

engine_raw_sensors_test = FeatureView(
    name="engine_raw_sensors_test",
    entities=[engine],
    ttl=timedelta(days=3650),
    schema=[Field(name="cycle", dtype=Int64), *_SENSOR_AND_SETTING_FIELDS],
    online=True,
    source=test_source,
    tags={"split": "test", "stage": "data_validation"},
)

# ----------------------------------------------------------------------
# Engineered features (feature_engineering.py output)
# ----------------------------------------------------------------------

engineered_train_source = FileSource(
    name="engineered_train_source",
    path="data/engineered_train.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

engineered_val_source = FileSource(
    name="engineered_val_source",
    path="data/engineered_val.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

engineered_test_source = FileSource(
    name="engineered_test_source",
    path="data/engineered_test.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

engine_engineered_features_train = FeatureView(
    name="engine_engineered_features_train",
    entities=[engine],
    ttl=timedelta(days=3650),
    schema=_infer_schema(REPO_DIR / "data" / "engineered_train.parquet"),
    online=True,
    source=engineered_train_source,
    tags={"split": "train", "stage": "feature_engineering"},
)

engine_engineered_features_val = FeatureView(
    name="engine_engineered_features_val",
    entities=[engine],
    ttl=timedelta(days=3650),
    schema=_infer_schema(REPO_DIR / "data" / "engineered_val.parquet"),
    online=True,
    source=engineered_val_source,
    tags={"split": "val", "stage": "feature_engineering"},
)

engine_engineered_features_test = FeatureView(
    name="engine_engineered_features_test",
    entities=[engine],
    ttl=timedelta(days=3650),
    schema=_infer_schema(REPO_DIR / "data" / "engineered_test.parquet"),
    online=True,
    source=engineered_test_source,
    tags={"split": "test", "stage": "feature_engineering"},
)
