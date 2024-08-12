# =============================================================================
# Running Script
# =============================================================================
import dynamic_nesting_dt as dt
from genetic_algo import optimization
from nlip import NLIP
from saturation_demand_curve import sat_demand_cutoff

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from config import config
import math
import numpy as np

# input_data_folder = r"C:/Users/40107903/Downloads/Git/catexpert-space-optimization-delivery/Data/Input/"
output_data_folder = r"C:\Users\C736424\OneDrive - Anheuser-Busch InBev\workspace\office_work\Space Phase1\Test Data\Phase_1\Output/"

sales_data_df = pd.read_excel("Preprocessed Sales Data.xlsx")
style_interaction_data_df = pd.read_excel("columbia_style_interaction_matrix.xlsx")
dfProductDims = pd.read_excel("colombia_sku_dims_new.xlsx")
dfConstraintMaster = pd.read_excel("sku_cons_flag_new.xlsx")
dfShelfMaster = pd.read_excel("New_stadard_shelf_dimension.xlsx")
dfConstraintTypeMapping = pd.read_excel("constraint_type_mapping.xlsx")
intraweek_seasonality_data = pd.read_excel("colombia_intra_week_seasonality.xlsx")
# similarity_data = pd.read_csv("DT_Full_Data (10).csv")
similarity_data = pd.read_csv("similarity_dndt_output.csv")

# =============================================================================
# Creating unique poc list to be pasted in the config.json
# =============================================================================

# sales_data_df["poc_id"].unique()

# =============================================================================
# Demand transfer and similarity score calculation
# =============================================================================

dt_obj = dt.DynamicNestingDT(sales_data_df, style_interaction_data_df, dfProductDims)
dt_obj.preprocess_data()
# dt_obj.run()
dt_obj.similarity_data = similarity_data

# =============================================================================
# Saturation and Cut off
# =============================================================================

sat_obj = sat_demand_cutoff(dt_obj.sku_master_data, dt_obj.sales_data_df,dfProductDims,dfShelfMaster,intraweek_seasonality_data)#
sat_obj.func_7_90_demand_curves()

#============================================================
# NLIP Script
#============================================================

nlip_obj =NLIP(sat_obj.demand_curve_df,dt_obj.similarity_data,sat_obj.sku_master_data,dfConstraintMaster,dfConstraintTypeMapping,dfShelfMaster,dfProductDims)
nlip_obj.Run()

nlip_obj.lstRemovedConstraints
nlip_obj.lstConstraintCols
nlip_obj.lstConstraintTypeMappings
nlip_obj.lstSpaceConstraintPercentages

nlip_obj.dfSKUColConstraints.Solution.sum()

# nlip_obj.dfSKUColConstraints.to_clipboard()
# nlip_obj.dfSKUColConstraints.to_excel(output_data_folder+"nlip_Plano2.xlsx", index=False)

# =============================================================================
# Running 1 iteration for existing plano
# =============================================================================

# cat_exp_plano_output  = pd.read_excel(input_data_folder+"historical_plano.xlsx",sheet_name='Plano_output')[['sku','Columns']]
# cat_exp_plano_output.rename({"Columns":"cat_exp_cols"},inplace=True,axis=1)
# cat_exp_plano_output['sku'] = cat_exp_plano_output['sku'].astype(str)
# try:
#     nlip_obj.dfSKUColConstraints.drop("cat_exp_cols",axis=1,inplace=True)
# except:
#     pass
# nlip_obj.dfSKUColConstraints =nlip_obj.dfSKUColConstraints.merge(cat_exp_plano_output,on='sku',how='left')
# nlip_obj.dfSKUColConstraints["Solution"] = nlip_obj.dfSKUColConstraints["cat_exp_cols"]
# nlip_obj.dfSKUColConstraints["Max_Cols"] = nlip_obj.dfSKUColConstraints["cat_exp_cols"]

#=============================================================================

optimization_obj = optimization(dt_obj.sku_master_data ,dt_obj.sales_data_df, dt_obj.similarity_data, sat_obj.demand_curve_df, sat_obj.df_output,nlip_obj.dfSKUColConstraints,nlip_obj.fltPlanoTotalAugmentedSpace,nlip_obj.dfConstraintMaster,nlip_obj.lstConstraintCols,nlip_obj.lstSpaceConstraintPercentages,nlip_obj.lstConstraintTypeMappings,dfShelfMaster)
optimization_obj.optimization_run()

optimization_obj.cost_2.to_clipboard()
optimization_obj.cost_2['Columns'].sum()

optimization_obj.cost_2.to_excel(output_data_folder+"Previous Plano.xlsx", index=False)

#============================================================

# =============================================================================
# Run the below lines of code to create sku_master & sku_dims_dos file used in the excel automation script for template creation
# =============================================================================

dfProductDims.to_excel(output_data_folder+"Product_Dims_DOS.xlsx", index=False)

sku_master_template = optimization_obj.cat_test_df[['cluster_parent_sku','sku','sku_detail',"ptc_segment",'manufacturer','style','pack_size','pack_type','brand_family','pack_qty','price_per_ltr','Real_Height','Width','Real_Depth','Num_units_per_col','Actual_Space_Per_Column','Stacked_Units']].rename(columns={'Real_Height':'Height', 'Real_Depth':'Depth'})
# optimization_obj.cat_test_df.rename({'Actual_Space_Per_Column':'Space Per Column'},axis=1,inplace=True)
mb_list = ["CORONA","STELLA ARTOIS","BRAHMA","BUDWEISER","ANDES ORIGEN","QUILMES"]
sku_master_template['mega brand'] = np.where(sku_master_template['brand_family'].isin(mb_list),1,0)

sales_agg_template = dt_obj.sales_data_df.groupby(['sku']).agg({'revenue':['sum','mean'],'volume':['sum','mean'],'units_sold':['sum','mean']})
sales_agg_template.columns  =['_'.join(col).strip() for col in sales_agg_template.columns.values]
sales_agg_template['volume_ms'] = sales_agg_template['volume_sum']/sales_agg_template['volume_sum'].sum()
sales_agg_template['revenue_ms'] = sales_agg_template['revenue_sum']/sales_agg_template['revenue_sum'].sum()
sales_agg_template = sku_master_template.merge(sales_agg_template,on='sku')
sales_agg_template.to_excel(output_data_folder+"SKU_Master.xlsx", index=False)