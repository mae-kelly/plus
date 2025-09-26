# debug_bigquery.py
"""
Debug script to find where the NA/array error is ACTUALLY coming from
"""

import sys
import os
import logging
import traceback

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_direct_bigquery():
    """Test BigQuery directly without any scanner"""
    from google.cloud import bigquery
    
    client = bigquery.Client(project='prj-tiay-p-gcss-sas-dl9dd01ddf')
    
    # Test a problematic table directly
    query = """
    SELECT * 
    FROM `prj-tiay-p-gcss-sas-dl9dd01ddf.CSIRT.ctl_tip_backup`
    LIMIT 5
    """
    
    print("\n" + "="*80)
    print("TEST 1: Direct BigQuery Query (No DataFrame)")
    print("="*80)
    
    try:
        query_job = client.query(query)
        
        # Method 1: Iterator only
        print("\nMethod 1: Using iterator only...")
        for row in query_job.result():
            print(f"Row type: {type(row)}")
            for key in row.keys():
                value = row[key]
                print(f"  {key}: type={type(value)}, value={str(value)[:50]}")
            break  # Just first row
        print("✅ Iterator method worked!")
        
    except Exception as e:
        print(f"❌ Iterator failed: {e}")
        traceback.print_exc()
    
    print("\n" + "-"*80)
    
    # Method 2: Try to_dataframe
    print("\nMethod 2: Using to_dataframe()...")
    try:
        query_job = client.query(query)
        df = query_job.to_dataframe()
        print(f"✅ to_dataframe worked! Shape: {df.shape}")
    except Exception as e:
        print(f"❌ to_dataframe failed: {e}")
        traceback.print_exc()
    
    print("\n" + "-"*80)
    
    # Method 3: Try to_arrow
    print("\nMethod 3: Using to_arrow()...")
    try:
        query_job = client.query(query)
        arrow_table = query_job.to_arrow()
        print(f"✅ to_arrow worked! Rows: {len(arrow_table)}")
    except Exception as e:
        print(f"❌ to_arrow failed: {e}")
        traceback.print_exc()

def test_imports():
    """Test if the issue is in imports"""
    print("\n" + "="*80)
    print("TEST 2: Testing Imports")
    print("="*80)
    
    # Test pandas import
    try:
        import pandas as pd
        print(f"✅ Pandas version: {pd.__version__}")
        
        # Test NA handling
        try:
            val = pd.NA
            if val:  # This should trigger the error if it's a pandas issue
                pass
        except Exception as e:
            print(f"❌ Pandas NA issue: {e}")
            
    except ImportError as e:
        print(f"❌ Can't import pandas: {e}")
    
    # Test numpy import
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"❌ Can't import numpy: {e}")
    
    # Test google-cloud-bigquery version
    try:
        import google.cloud.bigquery
        print(f"✅ BigQuery client version: {google.cloud.bigquery.__version__}")
    except ImportError as e:
        print(f"❌ Can't import BigQuery: {e}")

def test_scanner_minimal():
    """Test the scanner with minimal code"""
    print("\n" + "="*80)
    print("TEST 3: Minimal Scanner Test")
    print("="*80)
    
    try:
        from core.bigquery_client_manager import BigQueryClientManager
        
        manager = BigQueryClientManager(project_id='prj-tiay-p-gcss-sas-dl9dd01ddf')
        client = manager.get_client()
        
        # Get table directly
        table = client.get_table('prj-tiay-p-gcss-sas-dl9dd01ddf.CSIRT.ctl_tip_backup')
        print(f"✅ Got table: {table.table_id}, rows: {table.num_rows}")
        
        # Try simple query
        query = "SELECT * FROM `prj-tiay-p-gcss-sas-dl9dd01ddf.CSIRT.ctl_tip_backup` LIMIT 1"
        query_job = client.query(query)
        
        for row in query_job.result():
            print(f"✅ Got row with {len(row)} fields")
            break
            
    except Exception as e:
        print(f"❌ Scanner test failed: {e}")
        traceback.print_exc()

def find_the_real_issue():
    """Try to find where the error is REALLY coming from"""
    print("\n" + "="*80)
    print("TEST 4: Finding the REAL issue")
    print("="*80)
    
    # Check if it's in streaming_handler.py
    try:
        from streaming_handler import StreamingHandler
        print("⚠️ StreamingHandler imported - checking if it uses pandas...")
        
        # Check the source
        import inspect
        source = inspect.getsource(StreamingHandler)
        if 'pandas' in source or 'pd.' in source:
            print("❌ StreamingHandler uses pandas!")
        if 'to_dataframe' in source:
            print("❌ StreamingHandler calls to_dataframe()!")
            
    except Exception as e:
        print(f"Couldn't check StreamingHandler: {e}")
    
    # Check if it's in main.py
    try:
        with open('main.py', 'r') as f:
            main_content = f.read()
            if 'to_dataframe' in main_content:
                print("❌ main.py uses to_dataframe()!")
            if '.where(' in main_content or 'pd.notnull' in main_content:
                print("❌ main.py has pandas operations that might cause NA issues!")
    except:
        pass
    
    # Check config.json processing
    try:
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
            print(f"✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Config issue: {e}")

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           BigQuery NA/Array Error Debugger              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run all tests
    test_imports()
    test_direct_bigquery()
    test_scanner_minimal()
    find_the_real_issue()
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)
    print("\nBased on the results above, we can identify where the issue is coming from.")

if __name__ == "__main__":
    main()