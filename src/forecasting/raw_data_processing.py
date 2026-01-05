
import pandas as pd
import numpy as np

import os
import warnings

warnings.filterwarnings('ignore')

#### Load Raw Data
def load_raw_data_fn(data_info_dict, project_path):        
    """Load multiple raw CSV files into a dictionary of DataFrames"""
   
    res = {}    
    for df_name, data_file_path in data_info_dict.items():
        res[df_name] = pd.read_csv(str(project_path) + data_file_path)        
    return res  

#### Process Sales Data
def process_sales_data_fn(sales_df_raw, calendar_df_raw):    
    """Process sales data"""
    
    # Get the sales history of interested department and store
    sales = sales_df_raw.copy()
    sales_t = pd.melt(
        sales, 
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
        var_name='d', 
        value_name='sales'
        )
    
    # Add calendar dates
    sales_cal = sales_t.merge(calendar_df_raw[['date', 'd', 'wm_yr_wk']], on='d')
    sales_cal.drop(columns=['d'], inplace=True)
    
    # Drop the weeks that don't have 7 days
    id_sales_week_days = sales_cal.groupby(['dept_id', 'store_id', 'wm_yr_wk'])['date'].nunique().reset_index()
    valid_weeks = id_sales_week_days[id_sales_week_days['date']==7][['dept_id', 'store_id', 'wm_yr_wk']].drop_duplicates()
    
    # Aggregate sales at dept × store × date
    sales_dept_store=(
        sales_cal
        .groupby(['dept_id', 'store_id', 'wm_yr_wk'])
        .agg(sales=('sales', 'sum'),
             date =('date', 'first'))
        .reset_index()
    )
    
    sales_dept_store = sales_dept_store.merge(valid_weeks, on=['dept_id', 'store_id', 'wm_yr_wk'])

    # Format data    
    sales_dept_store['id'] = sales_dept_store['dept_id'] + '_' + sales_dept_store['store_id']   
    sales_dept_store['date'] =  pd.to_datetime(sales_dept_store['date'])
    sales_dept_store['sales'] = sales_dept_store['sales'].astype('float')
    
    for col in ['id', 'dept_id', 'store_id']:
        sales_dept_store[col] = sales_dept_store[col].astype('category')     
        
    # Sort data
    sales_dept_store = sales_dept_store[['id', 'dept_id', 'store_id','date', 'sales']].sort_values(['id', 'date'])  
    
    return sales_dept_store


#### Process Price Data
def process_price_data_fn(price_df_raw, sales_df_raw, calendar_df_raw):
    """Process price Data"""
    
    # Get the average prices of products at dept x store x week  
    item_dept_store = sales_df_raw[['item_id', 'dept_id', 'store_id']].drop_duplicates()     
    price_item = price_df_raw.merge(item_dept_store, on=['item_id', 'store_id'])    
    price_dept_store = price_item.groupby(['dept_id', 'store_id', 'wm_yr_wk'])['sell_price'].mean().reset_index()
    
    # Get the calendar weeks
    calendar_weeks = calendar_df_raw[['date', 'wm_yr_wk']].groupby(['wm_yr_wk']).agg(date=('date', 'first')).reset_index()
    
    # Merge price data with calendar data
    price_dept_store_cal = price_dept_store.merge(calendar_weeks[['date', 'wm_yr_wk']], on = ['wm_yr_wk'])
    price_dept_store_cal.drop(columns=['wm_yr_wk'], inplace=True)
    
    # Format data
    price_dept_store_cal['id'] = price_dept_store_cal['dept_id'] + '_' + price_dept_store_cal['store_id']        
    price_dept_store_cal['date'] = pd.to_datetime(price_dept_store_cal['date'])
    price_dept_store_cal['sell_price'] = price_dept_store_cal['sell_price'].astype('float').round(4)
    price_dept_store_cal['id'] = price_dept_store_cal['id'].astype('category')
    
    # Sort data
    price_dept_store_cal = price_dept_store_cal[['id', 'date', 'sell_price']].sort_values(['id', 'date'])
    
    return price_dept_store_cal


#### Process Calendar Data
def process_calendar_data_fn(calendar_df_raw):    
    """Process calendar data"""
    
    calendar_weeks = calendar_df_raw.drop(columns=['weekday', 'wday', 'd'])
    
    # Fill missing values of events
    for col in ['event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']:
        calendar_weeks[col] = calendar_weeks[col].fillna('0_NoEvent')     
    
    # Aggregate multiple events on the same week
    calendar_df = (
        calendar_weeks
        .groupby(['wm_yr_wk'])
        .agg(date=('date', 'first'),
             month=('month', 'first'),
             year=('year', 'first'),
             event_name_1=('event_name_1', 'max'),
             event_type_1=('event_type_1', 'max'),
             event_name_2=('event_name_2', 'max'),
             event_type_2=('event_type_2', 'max'),
             snap_CA=('snap_CA', 'max'),
             snap_TX=('snap_TX', 'max'),
             snap_WI=('snap_WI', 'max')
             ).reset_index()
        )

        
     # Format data
    for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'month']:
        calendar_df[col] = calendar_df[col].astype('category')

    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    
    # Sort data
    calendar_df = calendar_df.drop(columns=['wm_yr_wk']).sort_values(['date'])    
        
    return calendar_df

#### Combine Sales, Price, and Calendar Data
def combine_sales_price_calender_fn(sales_df, price_df, calendar_df):
    """Process sales, price, and calendar data"""
    
    # Combine datasets
    sales_price_calendar=sales_df.merge(price_df, on=['id', 'date'], how='left').merge(calendar_df, on=['date'], how='left')

    # Process SNAP fields
    sales_price_calendar['snap']=(sales_price_calendar[['store_id', 'snap_CA', 'snap_TX', 'snap_WI']]
                                .apply(lambda x: x['snap_CA'] if x['store_id'].find('CA')>-1
                                     else (x['snap_TX'] if x['store_id'].find('TX')>-1 else x['snap_WI']),
                                     axis=1)
                             )
    sales_price_calendar.drop(columns=['snap_CA', 'snap_TX', 'snap_WI'], inplace=True)
    
    # Sort dataset
    sales_price_calendar.sort_values(['id', 'date'], inplace=True)
    
    return sales_price_calendar


 # Orchestration Function
def process_raw_data_fn(raw_data_cfg, project_path):    
    """Raw data preprocessing pipeline"""
   
    # Load raw inputs
    data = load_raw_data_fn(raw_data_cfg, project_path)
    
    # Apply processing
    sales_df = process_sales_data_fn(data['sales'], data['calendar'])
    price_df = process_price_data_fn(data['price'], data['sales'], data['calendar'])
    calendar_df = process_calendar_data_fn(data['calendar'])
    sales_price_calendar_df = combine_sales_price_calender_fn(sales_df, price_df, calendar_df)    
    
    # Ensure output_dir exists
    processed_data_dir = project_path+"/data/processed/"
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Save outputs
    file_path = os.path.join(processed_data_dir, "sales_price_calendar.csv")       
    sales_price_calendar_df.to_csv(file_path, index=False)
    
    print("Preprocessing complete.")
   
    return file_path


