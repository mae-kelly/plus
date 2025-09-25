# CMDB+ 

Builds a Configuration Management Database by scanning BigQuery tables to discover hosts and their attributes.

## What it does

Scans your BigQuery projects looking for anything that looks like a hostname or IP address, then collects all the data associated with those hosts from across all your tables. Outputs a DuckDB database with everything organized.

## Requirements

- Mac with M1/M2/M3 chip
- Python 3.8 or newer
- Google Cloud project with BigQuery
- Service account key with BigQuery read permissions

## Setup

1. Clone this repo

2. Install Python packages:
```bash
pip3 install -r requirements.txt
```

3. Get PyTorch working on your Mac:
```bash
pip3 install torch torchvision torchaudio
```

If that doesn't work, run the install script:
```bash
chmod +x install.sh
./install.sh
```

4. Put your GCP service account key in the root directory as `gcp_prod_key.json`

5. Edit `config.json` and add your BigQuery project IDs:
```json
{
  "projects": ["my-project-1", "my-project-2"],
  "max_workers": 5,
  "rows_per_batch": 10000
}
```

## Running it

```bash
python3 main.py
```

It runs in 4 phases:
1. Learns what hostnames look like by scanning host/hostname columns
2. Scans all tables looking for hosts and collecting associated data
3. Analyzes and aggregates the data
4. Builds the CMDB database

## Output

You get:
- `new_cmdb.db` - DuckDB database with all discovered hosts
- `cmdb_backup.csv` - CSV backup of the data
- `cmdb_statistics.json` - Stats about what was found
- `logs/` - Detailed logs of the run

## Querying the CMDB

Connect to the DuckDB database:
```bash
duckdb new_cmdb.db
```

Example queries:
```sql
-- See all hosts
SELECT * FROM hosts LIMIT 10;

-- Find hosts with high confidence
SELECT hostname, confidence, occurrence_count 
FROM hosts 
WHERE confidence > 0.8
ORDER BY occurrence_count DESC;

-- Search for specific host
SELECT * FROM hosts WHERE hostname LIKE '%server%';
```

## Configuration options

Edit `config.json`:

- `projects` - List of GCP project IDs to scan
- `max_workers` - Parallel threads (default 5)
- `rows_per_batch` - Rows to process at once (default 10000)
- `sampling_percent_for_large_tables` - For tables over 1M rows, what percent to sample (default 10)
- `large_table_threshold_rows` - What counts as a large table (default 1000000)
- `checkpoint_enabled` - Save progress for resume on failure (default true)

## Troubleshooting

**PyTorch not found**
Run `./setup.py` to check your environment. May need to reinstall PyTorch.

**Dataset not found in location US**
Your datasets are in a different region. The code handles this automatically but you might see warnings.

**Out of memory**
Lower `rows_per_batch` and `max_workers` in config.json

**Authentication failed**
Check your service account key has BigQuery Data Viewer role at minimum.

## How it works

The system uses machine learning to identify hostnames without needing predefined patterns. It:

1. Trains on known hostname columns to learn patterns
2. Uses neural networks on Mac GPU to classify potential hostnames
3. Normalizes different hostname formats (handles unicode, special characters, etc)
4. Aggregates data from multiple tables for each host
5. Builds relationships between hosts based on shared attributes

The checkpoint system means if it crashes, you can restart and it picks up where it left off.

## Files

- `main.py` - Main orchestrator
- `discovery_engine.py` - Hostname detection
- `column_classifier.py` - Figures out what type of data each column contains
- `normalization_engine.py` - Cleans up messy data
- `pattern_recognition.py` - ML pattern learning
- `cmdb_builder.py` - Creates the output database
- `bigquery_client.py` - BigQuery connection handling
- `gpu_accelerator.py` - Mac GPU setup

## Limits

- Requires Mac with Apple Silicon (M1/M2/M3)
- Processes up to 100 most important columns
- Stores up to 50 values per column per host
- For large tables (>1M rows) only samples 10% by default