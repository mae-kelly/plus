async def _scan_table_with_ml(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
    """Scan table with ML-enhanced column detection using multiple strategies"""
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    # Define multiple scanning strategies
    strategies = [
        ('iterator_simple', self._scan_strategy_iterator_simple),
        ('iterator_with_conversion', self._scan_strategy_iterator_with_conversion),
        ('to_arrow', self._scan_strategy_arrow),
        ('to_dataframe_safe', self._scan_strategy_dataframe_safe),
        ('direct_sql', self._scan_strategy_direct_sql),
        ('fallback_minimal', self._scan_strategy_fallback_minimal)
    ]
    
    # Try each strategy until one works
    for strategy_name, strategy_func in strategies:
        try:
            logger.debug(f"Attempting strategy: {strategy_name} for table {table_id}")
            result = await strategy_func(client, project_id, dataset_id, table_id)
            
            if result is not None:
                logger.info(f"✅ Successfully scanned {table_id} using strategy: {strategy_name}")
                return result
                
        except Exception as e:
            logger.debug(f"Strategy {strategy_name} failed for {table_id}: {str(e)[:100]}")
            continue
    
    # If all strategies fail, return minimal data
    logger.warning(f"All strategies failed for {full_table_id}, returning minimal data")
    return await self._get_minimal_table_info(client, project_id, dataset_id, table_id)

async def _scan_strategy_iterator_simple(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
    """Strategy 1: Simple iterator without pandas"""
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    table = client.get_table(full_table_id)
    if table.num_rows == 0:
        return None
    
    sample_size = min(
        int(table.num_rows * self.config['bigquery']['sample_percent'] / 100),
        self.config['bigquery'].get('max_rows_per_table', 100000)
    )
    
    query = f"""
    SELECT *
    FROM `{full_table_id}`
    TABLESAMPLE SYSTEM ({min(self.config['bigquery']['sample_percent'], 100)} PERCENT)
    LIMIT {sample_size}
    """
    
    query_job = client.query(query)
    rows = []
    columns_with_features = {}
    
    # Initialize columns from schema
    for field in table.schema:
        columns_with_features[field.name] = {
            'name': field.name,
            'samples': [],
            'features': None,
            'statistics': {},
            'type': field.field_type,
            'mode': field.mode
        }
    
    # Process rows without pandas
    for row in query_job.result():
        row_dict = {}
        for field in table.schema:
            value = row.get(field.name)
            
            # Handle special types
            if value is not None:
                if field.field_type in ['RECORD', 'STRUCT', 'JSON']:
                    value = str(value)
                elif field.mode == 'REPEATED':
                    value = str(value) if value else '[]'
                elif hasattr(value, 'isoformat'):
                    value = value.isoformat()
            
            row_dict[field.name] = value
            
            # Collect samples
            if len(columns_with_features[field.name]['samples']) < 100:
                columns_with_features[field.name]['samples'].append(value)
        
        rows.append(row_dict)
    
    # Extract features for each column
    for col_name, col_data in columns_with_features.items():
        values = col_data['samples']
        
        # Extract features using feature extractor
        features = await self.feature_extractor.extract(col_name, values)
        col_data['features'] = features
        col_data['statistics'] = self._analyze_column(col_name, values)
    
    self.stats['rows_processed'] += len(rows)
    
    return {
        'name': table_id,
        'full_name': full_table_id,
        'rows': rows,
        'columns': columns_with_features,
        'row_count': len(rows),
        'total_rows': table.num_rows
    }

async def _scan_strategy_iterator_with_conversion(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
    """Strategy 2: Iterator with TO_JSON_STRING for complex types"""
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    table = client.get_table(full_table_id)
    if table.num_rows == 0:
        return None
    
    # Build query with JSON conversion for complex types
    select_parts = []
    complex_fields = []
    
    for field in table.schema:
        if field.mode == 'REPEATED' or field.field_type in ['RECORD', 'STRUCT', 'JSON', 'GEOGRAPHY', 'ARRAY']:
            select_parts.append(f"TO_JSON_STRING({field.name}) AS {field.name}")
            complex_fields.append(field.name)
        else:
            select_parts.append(field.name)
    
    sample_size = min(
        int(table.num_rows * self.config['bigquery']['sample_percent'] / 100),
        self.config['bigquery'].get('max_rows_per_table', 100000)
    )
    
    query = f"""
    SELECT {', '.join(select_parts)}
    FROM `{full_table_id}`
    TABLESAMPLE SYSTEM ({min(self.config['bigquery']['sample_percent'], 100)} PERCENT)
    LIMIT {sample_size}
    """
    
    query_job = client.query(query)
    rows = []
    columns_with_features = {}
    
    # Process results
    for row in query_job.result():
        row_dict = dict(row)
        rows.append(row_dict)
        
        for key, value in row_dict.items():
            if key not in columns_with_features:
                columns_with_features[key] = {
                    'name': key,
                    'samples': [],
                    'features': None,
                    'statistics': {}
                }
            
            if len(columns_with_features[key]['samples']) < 100:
                columns_with_features[key]['samples'].append(value)
    
    # Extract features
    for col_name, col_data in columns_with_features.items():
        values = col_data['samples']
        features = await self.feature_extractor.extract(col_name, values)
        col_data['features'] = features
        col_data['statistics'] = self._analyze_column(col_name, values)
    
    self.stats['rows_processed'] += len(rows)
    
    return {
        'name': table_id,
        'full_name': full_table_id,
        'rows': rows,
        'columns': columns_with_features,
        'row_count': len(rows),
        'total_rows': table.num_rows
    }

async def _scan_strategy_arrow(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
    """Strategy 3: Use PyArrow for better type handling"""
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    table = client.get_table(full_table_id)
    if table.num_rows == 0:
        return None
    
    sample_size = min(
        int(table.num_rows * self.config['bigquery']['sample_percent'] / 100),
        self.config['bigquery'].get('max_rows_per_table', 100000)
    )
    
    query = f"""
    SELECT *
    FROM `{full_table_id}`
    TABLESAMPLE SYSTEM ({min(self.config['bigquery']['sample_percent'], 100)} PERCENT)
    LIMIT {sample_size}
    """
    
    query_job = client.query(query)
    
    # Try to use Arrow
    arrow_table = query_job.to_arrow()
    
    rows = []
    columns_with_features = {}
    
    # Convert Arrow table to Python objects
    for col_name in arrow_table.column_names:
        column = arrow_table.column(col_name)
        
        columns_with_features[col_name] = {
            'name': col_name,
            'samples': [],
            'features': None,
            'statistics': {}
        }
        
        # Convert to Python list safely
        for i in range(min(len(column), sample_size)):
            value = column[i].as_py() if hasattr(column[i], 'as_py') else str(column[i])
            
            if i < 100:
                columns_with_features[col_name]['samples'].append(value)
            
            if i < len(rows):
                rows[i][col_name] = value
            else:
                rows.append({col_name: value})
    
    # Extract features
    for col_name, col_data in columns_with_features.items():
        values = col_data['samples']
        features = await self.feature_extractor.extract(col_name, values)
        col_data['features'] = features
        col_data['statistics'] = self._analyze_column(col_name, values)
    
    self.stats['rows_processed'] += len(rows)
    
    return {
        'name': table_id,
        'full_name': full_table_id,
        'rows': rows,
        'columns': columns_with_features,
        'row_count': len(rows),
        'total_rows': table.num_rows
    }

async def _scan_strategy_dataframe_safe(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
    """Strategy 4: Use DataFrame with safe conversion"""
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    table = client.get_table(full_table_id)
    if table.num_rows == 0:
        return None
    
    sample_size = min(
        int(table.num_rows * self.config['bigquery']['sample_percent'] / 100),
        self.config['bigquery'].get('max_rows_per_table', 100000)
    )
    
    query = f"""
    SELECT *
    FROM `{full_table_id}`
    TABLESAMPLE SYSTEM ({min(self.config['bigquery']['sample_percent'], 100)} PERCENT)
    LIMIT {sample_size}
    """
    
    query_job = client.query(query)
    
    # Use to_dataframe with specific dtypes to avoid issues
    import pandas as pd
    
    # Set string dtype for problematic columns
    dtype_kwargs = {}
    for field in table.schema:
        if field.field_type in ['RECORD', 'STRUCT', 'JSON', 'ARRAY'] or field.mode == 'REPEATED':
            dtype_kwargs[field.name] = 'object'
    
    if dtype_kwargs:
        df = query_job.to_dataframe(create_bqstorage_client=False, dtypes=dtype_kwargs)
    else:
        df = query_job.to_dataframe(create_bqstorage_client=False)
    
    # Convert DataFrame to safe format
    rows = []
    columns_with_features = {}
    
    for col_name in df.columns:
        columns_with_features[col_name] = {
            'name': col_name,
            'samples': [],
            'features': None,
            'statistics': {}
        }
        
        # Get samples safely
        col_values = df[col_name]
        for i in range(min(100, len(col_values))):
            value = col_values.iloc[i]
            
            # Handle pandas NA and numpy arrays
            if pd.isna(value):
                value = None
            elif hasattr(value, 'tolist'):
                value = value.tolist()
            elif hasattr(value, 'item'):
                value = value.item()
            else:
                value = str(value) if value is not None else None
            
            columns_with_features[col_name]['samples'].append(value)
    
    # Convert rows
    for _, row in df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            
            # Safe conversion
            if pd.isna(value):
                value = None
            elif hasattr(value, 'tolist'):
                value = value.tolist()
            elif hasattr(value, 'item'):
                value = value.item()
            else:
                value = str(value) if value is not None else None
            
            row_dict[col] = value
        
        rows.append(row_dict)
    
    # Extract features
    for col_name, col_data in columns_with_features.items():
        values = col_data['samples']
        features = await self.feature_extractor.extract(col_name, values)
        col_data['features'] = features
        col_data['statistics'] = self._analyze_column(col_name, values)
    
    self.stats['rows_processed'] += len(rows)
    
    return {
        'name': table_id,
        'full_name': full_table_id,
        'rows': rows,
        'columns': columns_with_features,
        'row_count': len(rows),
        'total_rows': table.num_rows
    }

async def _scan_strategy_direct_sql(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
    """Strategy 5: Direct SQL with explicit casting"""
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    table = client.get_table(full_table_id)
    if table.num_rows == 0:
        return None
    
    # Get column info first
    info_query = f"""
    SELECT 
        column_name,
        data_type
    FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table_id}'
    """
    
    try:
        info_job = client.query(info_query)
        column_info = {row['column_name']: row['data_type'] for row in info_job.result()}
    except:
        # Fallback to schema
        column_info = {field.name: field.field_type for field in table.schema}
    
    # Build safe query with casting
    select_parts = []
    for col_name, col_type in column_info.items():
        if col_type in ['ARRAY', 'STRUCT', 'JSON', 'GEOGRAPHY']:
            select_parts.append(f"CAST({col_name} AS STRING) AS {col_name}")
        else:
            select_parts.append(col_name)
    
    if not select_parts:
        select_parts = ['*']
    
    sample_size = min(1000, table.num_rows)  # Smaller sample for safety
    
    query = f"""
    SELECT {', '.join(select_parts)}
    FROM `{full_table_id}`
    LIMIT {sample_size}
    """
    
    query_job = client.query(query)
    
    rows = []
    columns_with_features = {}
    
    for row in query_job.result():
        row_dict = dict(row)
        rows.append(row_dict)
        
        for key, value in row_dict.items():
            if key not in columns_with_features:
                columns_with_features[key] = {
                    'name': key,
                    'samples': [],
                    'features': None,
                    'statistics': {}
                }
            
            if len(columns_with_features[key]['samples']) < 100:
                columns_with_features[key]['samples'].append(value)
    
    # Extract features
    for col_name, col_data in columns_with_features.items():
        values = col_data['samples']
        features = await self.feature_extractor.extract(col_name, values)
        col_data['features'] = features
        col_data['statistics'] = self._analyze_column(col_name, values)
    
    self.stats['rows_processed'] += len(rows)
    
    return {
        'name': table_id,
        'full_name': full_table_id,
        'rows': rows,
        'columns': columns_with_features,
        'row_count': len(rows),
        'total_rows': table.num_rows
    }

async def _scan_strategy_fallback_minimal(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
    """Strategy 6: Minimal scan - just get schema and a few rows"""
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    table = client.get_table(full_table_id)
    
    columns_with_features = {}
    rows = []
    
    # Just get schema info
    for field in table.schema:
        columns_with_features[field.name] = {
            'name': field.name,
            'type': field.field_type,
            'mode': field.mode,
            'samples': [],
            'features': None,
            'statistics': {}
        }
    
    # Try to get just a few rows
    try:
        query = f"SELECT * FROM `{full_table_id}` LIMIT 10"
        query_job = client.query(query)
        
        for row in query_job.result():
            row_dict = {}
            for field in table.schema:
                try:
                    value = row.get(field.name)
                    row_dict[field.name] = str(value) if value is not None else None
                    
                    if len(columns_with_features[field.name]['samples']) < 10:
                        columns_with_features[field.name]['samples'].append(str(value) if value is not None else None)
                except:
                    row_dict[field.name] = None
            
            rows.append(row_dict)
    except:
        logger.warning(f"Could not sample rows from {table_id}")
    
    self.stats['rows_processed'] += len(rows)
    
    return {
        'name': table_id,
        'full_name': full_table_id,
        'rows': rows,
        'columns': columns_with_features,
        'row_count': len(rows),
        'total_rows': table.num_rows,
        'scan_type': 'minimal'
    }

async def _get_minimal_table_info(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
    """Get absolute minimal info when all strategies fail"""
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    try:
        table = client.get_table(full_table_id)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': [],
            'columns': {field.name: {'name': field.name, 'type': field.field_type} for field in table.schema},
            'row_count': 0,
            'total_rows': table.num_rows,
            'scan_type': 'metadata_only',
            'error': 'All scan strategies failed'
        }
    except:
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': [],
            'columns': {},
            'row_count': 0,
            'total_rows': 0,
            'scan_type': 'failed',
            'error': 'Could not access table'
        }