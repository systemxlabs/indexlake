# IndexLake

![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![Crates.io](https://img.shields.io/crates/v/indexlake.svg)](https://crates.io/crates/indexlake)
[![Docs](https://docs.rs/indexlake/badge.svg)](https://docs.rs/indexlake/latest/indexlake/)

**IndexLake** is an experimental table format with extensible index and inline table support.

- **Extensible Index System**: Pluggable index types (BM25, B-Tree, HNSW, R*-tree)
- **Inline Tables**: Store small datasets directly within the catalog
- **ACID Transaction**: Support transaction through sql catalog
- **Flexible Catalog**: PostgreSQL and SQLite catalog support
- **Flexible Storage**: Local filesystem and S3-compatible storage backends
- **DataFusion Integration**: Native Apache DataFusion support for SQL query workloads

## Basic Example (CRUD)

See `indexlake/examples/basic_crud.rs` for a minimal end-to-end example that:

- creates a table
- inserts data
- reads data
- updates data
- deletes data

Run it with:

```bash
cargo run -p indexlake --example basic_crud
```

## Project Structure

```
IndexLake
├── Core Library (indexlake)        # Core table format and APIs
├── Catalogs
│   ├── PostgreSQL                  # Postgres catalog
│   └── SQLite                      # SQLite catalog
├── Indexes
│   ├── BM25                        # Full-text search index
│   ├── B-Tree                      # Traditional B-Tree index
│   ├── HNSW                        # Hierarchical navigable small world graph
│   └── R*-tree                     # Spatial index for multidimensional data
├── Storages
│   ├── FS                          # Local filesystem storage
│   └── S3                          # S3-compatible object storage
└── Integrations
    └── DataFusion                  # Apache DataFusion SQL engine integration
```
