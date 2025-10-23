CREATE TABLE indexlake_namespace (
    namespace_id UUID PRIMARY KEY,
    namespace_name VARCHAR NOT NULL UNIQUE
);

CREATE TABLE indexlake_table (
    table_id UUID PRIMARY KEY,
    table_name VARCHAR NOT NULL,
    namespace_id UUID NOT NULL,
    config VARCHAR NOT NULL,
    schema_metadata VARCHAR NOT NULL,
    UNIQUE (namespace_id, table_name)
);

CREATE TABLE indexlake_field (
    field_id UUID PRIMARY KEY,
    table_id UUID NOT NULL,
    field_name VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,
    nullable BOOLEAN NOT NULL,
    default_value BYTEA,
    metadata VARCHAR NOT NULL
);

CREATE TABLE indexlake_dump_task (
    table_id UUID PRIMARY KEY
);

CREATE TABLE indexlake_data_file (
    data_file_id UUID PRIMARY KEY,
    table_id UUID NOT NULL,
    format VARCHAR NOT NULL,
    relative_path VARCHAR NOT NULL,
    record_count BIGINT NOT NULL,
    validity BYTEA NOT NULL
);

CREATE TABLE indexlake_index (
    index_id UUID PRIMARY KEY,
    table_id UUID NOT NULL,
    index_name VARCHAR NOT NULL,
    index_kind VARCHAR NOT NULL,
    key_field_ids VARCHAR NOT NULL,
    params VARCHAR NOT NULL
);

CREATE TABLE indexlake_index_file (
    index_file_id UUID PRIMARY KEY,
    table_id UUID NOT NULL,
    index_id UUID NOT NULL,
    data_file_id UUID NOT NULL,
    relative_path VARCHAR NOT NULL
);

CREATE TABLE indexlake_inline_index (
    index_id UUID NOT NULL,
    index_data BYTEA NOT NULL
);

CREATE TABLE indexlake_data_file_stats (
    data_file_id UUID NOT NULL,
    field_id UUID NOT NULL,
    min_value VARCHAR,
    max_value VARCHAR
);