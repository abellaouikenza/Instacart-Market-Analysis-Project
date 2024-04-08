import pandas as pd
import numpy as np
import gc
from scipy import stats

import pandas as pd
import numpy as np
import gc
from scipy import stats



def get_user_features(df_order_product_prior, df_orders):
    """
    Computes user features based on prior order history and order data.

    Args:
        df_order_product_prior (DataFrame): Users' prior order product DataFrame.
        df_orders (DataFrame): Users' order data DataFrame.

    Returns:
        DataFrame: DataFrame containing computed user features.
    """
    # Merge df_order_product_prior with df_orders to get user_id information
    df_order_product_prior = pd.merge(df_order_product_prior, df_orders[['order_id', 'user_id']], on='order_id', how='left')

    # Compute user-level features
    user = df_orders.groupby('user_id').agg({
    'order_number': 'max',
    'days_since_prior_order': ['mean', 'std'],
    'order_dow': lambda x: stats.mode(x)[0][0],  # Extract the mode value from mode array
    'order_hour_of_day': lambda x: stats.mode(x)[0][0],  # Extract the mode value from mode array
}).reset_index()
    user.columns = ['user_id', 'u_total_orders', 'u_average_days_between_orders',
                    'u_days_between_orders_std', 'u_dow_most_orders', 'u_hod_most_orders']

    # Compute total items bought and unique products
    user_counts = df_order_product_prior.groupby('user_id').agg({
        'order_id': 'count',
        'product_id': lambda x: x.nunique()
    }).reset_index()
    user_counts.columns = ['user_id', 'u_total_items_bought', 'u_total_unique_prod']

    # Merge user-level features with counts
    user = pd.merge(user, user_counts, on='user_id', how='left')

    # Compute user's registration date
    users_fe = df_orders.query("order_number_reverse != 0").groupby("user_id").agg({'date': 'max'}).rename(columns={'date': 'u_registration_date'}).reset_index()
    user = pd.merge(user, users_fe, on='user_id', how='left')

    # Compute user's average basket size
    basket_size_per_order = df_order_product_prior.groupby(['user_id', 'order_id'])['product_id'].count().reset_index(name='basket_size_per_order')
    avg_basket_size = basket_size_per_order.groupby('user_id')['basket_size_per_order'].agg(['sum', 'mean', 'std']).reset_index()
    avg_basket_size.columns = ['user_id', 'u_basket_sum', 'u_avg_basket_size', 'u_basket_std']
    del basket_size_per_order
    gc.collect()

    # Merge user features with average basket size
    user = pd.merge(user, avg_basket_size, on='user_id', how='left')

    return user




def get_product_features(df_order_product_prior, df_products, df_orders):
    """
    Computes product features based on prior order history and product data.

    Args:
        df_order_product_prior (DataFrame): Users' prior order history DataFrame.
        df_products (DataFrame): Product data DataFrame.

    Returns:
        DataFrame: DataFrame containing computed product features.
    """

     # Merge df_order_product_prior with df_orders to get necessary columns
    df_order_product_prior = pd.merge(df_order_product_prior, df_orders[['order_id', 'user_id', 'date', 'order_number', 'order_number_reverse']], on='order_id', how='left')

    # Convert date column to datetime format
    df_order_product_prior['date'] = pd.to_datetime(df_order_product_prior['date'])

    product = df_order_product_prior.groupby('product_id').agg({
        'order_id': 'count',
        'reordered': 'mean',
        'add_to_cart_order': 'mean',
        'user_id': lambda x: x.nunique(),
        'order_number': "mean",
        'order_number_reverse': "mean",
        'date': "mean"
    }).reset_index()
    product.columns = ['product_id', 'p_total_purchases', 'p_reorder_ratio', 'p_avg_cart_position', 'p_unique_user_count',
                       'p_recency_order', 'p_recency_order_rev', 'p_recency_date']
    product = product.merge(df_products[['product_id', 'product_name', 'aisle_id', 'department_id']], on='product_id', how='left')

    # Product trend calculation
    products_trend = df_order_product_prior.query("order_number_reverse < 3").groupby(["product_id", "order_number_reverse"]).size().rename("p_size").reset_index()
    products_trend["p_trend_rt"] = products_trend["p_size"] / products_trend["p_size"].shift(-1)
    products_trend["p_trend_diff"] = products_trend["p_size"] - products_trend["p_size"].shift(-1)
    cond = products_trend["product_id"] != products_trend["product_id"].shift(-1)
    products_trend.loc[cond, ["p_trend_rt", "p_trend_diff"]] = np.nan
    products_trend = products_trend.query("order_number_reverse == 1").drop("order_number_reverse", 1)
    product = pd.merge(product, products_trend, how="left", on="product_id")

    # Product frequency calculation
    product_freq = df_order_product_prior.copy()
    product_freq = product_freq.sort_values(["user_id", "product_id", "order_number"])
    product_freq["p_freq_days"] = product_freq["date"].shift() - product_freq["date"]
    product_freq["p_freq_order"] = product_freq["order_number"] - product_freq["order_number"].shift()
    product_freq = product_freq.query("reordered == 1")
    product_freq = product_freq.groupby("product_id").agg({'p_freq_days': "mean", 'p_freq_order': "mean"}).reset_index()
    product = pd.merge(product, product_freq, how="left", on="product_id")

    return product



def get_user_product_features(df_order_product_prior, df_users, df_products, df_orders):
    """
    Computes user x product features based on prior order history, user data, product data, and order data.

    Args:
        df_order_product_prior (DataFrame): Users' prior order history DataFrame.
        df_users (DataFrame): User data DataFrame.
        df_products (DataFrame): Product data DataFrame.
        df_orders (DataFrame): Order data DataFrame.

    Returns:
        DataFrame: DataFrame containing computed user x product features.
    """

   # Merge df_order_product_prior with df_orders to get necessary columns
    df_order_product_prior = pd.merge(df_order_product_prior, df_orders[['order_id', 'user_id', 'date', 'order_number', 'order_number_reverse']], on='order_id', how='left')

# Convert 'date' column to datetime
    df_order_product_prior['date'] = pd.to_datetime(df_order_product_prior['date'])

# Define reference date
    reference_date = pd.to_datetime('2018-01-01')

# Compute 'add_to_cart_order_inverted': The inverse of 'add_to_cart_order'
    df_order_product_prior['add_to_cart_order_inverted'] = 1 / df_order_product_prior['add_to_cart_order']

# 'add_to_cart_order_relative': Normalize 'add_to_cart_order' by the maximum value
    df_order_product_prior['add_to_cart_order_relative'] = df_order_product_prior['add_to_cart_order'] / df_order_product_prior['add_to_cart_order'].max()

# Compute 'uxp_date_strike': 1 / 2 ** (days_since_reference / 7)
    days_since_reference = (df_order_product_prior['date'] - reference_date).dt.days
    df_order_product_prior['uxp_date_strike'] = 1 / (2 ** (days_since_reference / 7))

# Compute 'uxp_order_strike': 1 / 2 ** (order_number_reverse)
    df_order_product_prior['uxp_order_strike'] = 1 / (2 ** df_order_product_prior['order_number_reverse'])


    uxp = df_order_product_prior.groupby(['user_id', 'product_id']).agg({'order_id' : 'count',
                                                                         'reordered' : 'mean',
                                                                         'add_to_cart_order': 'mean',
                                                                         'add_to_cart_order_relative' : 'mean',
                                                                         'add_to_cart_order_inverted' :  'mean',
                                                                         'order_number_reverse' : ['min','max'],
                                                                         'date' :  ['min','max'],
                                                                         'uxp_date_strike' : 'sum',
                                                                         'uxp_order_strike' : 'sum'
                                                                        }).reset_index()
    uxp.columns = ['user_id','product_id','uxp_total_bought','uxp_reorder_ratio','uxp_avg_cart_position',
                   'uxp_add_to_cart_order_relative_mean','uxp_add_to_cart_order_inverted_mean','uxp_last_order_number','uxp_first_order_number',
                   'up_last_order_date','up_first_order_date','uxp_date_strike','uxp_order_strike']
    
    # Calculate additional features
    uxp["bool_reordered"] = (uxp["uxp_total_bought"] > 1).astype("int")
    users_fe1 = uxp.groupby('user_id')["bool_reordered"].agg(["mean", "size"]).reset_index()\
                                            .rename(index=str, columns={"mean": "u_reorder_ratio_bool", "size": "u_tot_active_prod"})
    user = pd.merge(df_users, users_fe1, on="user_id",how="left")
    
    product_fe1 = uxp.groupby('product_id')["bool_reordered"].agg(["mean", "size"]).reset_index()\
                                            .rename(index=str, columns={"mean": "p_reorder_ratio_bool", "size": "p_tot_active_usr"})
    product = pd.merge(df_products, product_fe1, on="product_id",how="left")
    
    uxp.drop(['bool_reordered'], axis=1, inplace=True)

    # Calculate recency feature
    df_order_product_prior['order_number_back'] = df_order_product_prior.groupby('user_id')['order_number'].transform(max) - df_order_product_prior.order_number + 1 
    last_five = df_order_product_prior[df_order_product_prior.order_number_back <= 5]
    last_five = last_five.groupby(['user_id','product_id'])[['order_id']].count()
    last_five.columns = ['uxp_bought_last5']
    uxp = uxp.merge(last_five, on=['user_id', 'product_id'], how='left')
    uxp.fillna(0, inplace=True)
    
    return uxp, user, product




