CREATE TABLE IF NOT EXISTS indexlake_namespace (
    namespace_id BLOB PRIMARY KEY,
    namespace_name VARCHAR NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS indexlake_table (
    table_id BLOB PRIMARY KEY,
    table_name VARCHAR NOT NULL,
    namespace_id BLOB NOT NULL,
    config VARCHAR NOT NULL,
    schema_metadata VARCHAR NOT NULL,
    UNIQUE (namespace_id, table_name)
);

CREATE TABLE IF NOT EXISTS indexlake_field (
    field_id BLOB PRIMARY KEY,
    table_id BLOB NOT NULL,
    field_name VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,
    nullable BOOLEAN NOT NULL,
    default_value BLOB,
    metadata VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS indexlake_task (
    task_id VARCHAR PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS indexlake_data_file (
    data_file_id BLOB PRIMARY KEY,
    table_id BLOB NOT NULL,
    format VARCHAR NOT NULL,
    relative_path VARCHAR NOT NULL,
    size BIGINT NOT NULL,
    record_count BIGINT NOT NULL,
    validity BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS indexlake_index (
    index_id BLOB PRIMARY KEY,
    table_id BLOB NOT NULL,
    index_name VARCHAR NOT NULL,
    index_kind VARCHAR NOT NULL,
    key_field_ids VARCHAR NOT NULL,
    params VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS indexlake_index_file (
    index_file_id BLOB PRIMARY KEY,
    table_id BLOB NOT NULL,
    index_id BLOB NOT NULL,
    data_file_id BLOB NOT NULL,
    relative_path VARCHAR NOT NULL,
    size BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS indexlake_inline_index (
    index_id BLOB NOT NULL,
    index_data BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS indexlake_data_file_stats (
    data_file_id BLOB NOT NULL,
    field_id BLOB NOT NULL,
    min_value VARCHAR,
    max_value VARCHAR
);