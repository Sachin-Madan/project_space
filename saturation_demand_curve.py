import pandas as pd
import numpy as np
import math
from config import config



class sat_demand_cutoff:
    
    def __init__(self,sku_master_data,sales_data_df,product_dims,shelf_master,intraweek_seasonality_data):
        self.sku_master_data =sku_master_data
        self.sales_data_df = sales_data_df
        self.product_dims = product_dims
        self.shelf_master = shelf_master
        self.intraweek_seasonality_data = intraweek_seasonality_data

    def meet_demand_perc(self,df_1):
        df_1['units_sold_results'] = float('nan')
        for i in range(len(df_1)):
            df_1 = df_1.sort_values(by='units_sold').reset_index(drop=True)
            row_values = df_1['units_sold'].iloc[i:]
            df_1.at[i, 'units_sold_results'] = (row_values.sum() -  df_1.at[i,'units_sold'] * row_values.count())
        mode_per_group_units_sold = df_1.groupby('units_sold')['units_sold_results'].transform(lambda x: x.max())
        df_1['units_sold_results'] = df_1['units_sold_results'].fillna(mode_per_group_units_sold)   
        df_1['Meeting_Demand_units_perc'] = 1 - (df_1['units_sold_results']/df_1['units_sold'].sum())
        #     df_1['Meeting_Volume'] = df_1['units_sold'] * df_1['volume_per_unit'] * df_1['Meeting_Demand_units']
        # df_1['Total_Volume'] = df_1.groupby('cluster_parent_sku')['units_sold'].transform(lambda x: x.sum()) * df_1['volume_per_unit']
        
        return df_1 


    def perc_time(self,df_11):
        frequency_counts = df_11.groupby('cluster_parent_sku')['units_sold'].value_counts().reset_index(name='count')

        # Calculate the total number of rows for each 'group'
        group_counts = df_11['cluster_parent_sku'].value_counts().reset_index()
        group_counts.columns = ['cluster_parent_sku', 'total_count']

        # Merge the frequency counts and total counts DataFrames
        result = pd.merge(frequency_counts, group_counts, on='cluster_parent_sku')
        result = result.sort_values(by="units_sold")
        # Divide the 'count' column by the 'total_count'
        result['percentage_covered'] = result['count'] / result['total_count']
        result['percentage_times_met'] = result['percentage_covered'].cumsum()
        return (result)


    def linear_interpolation(self,df,inter_col):
        df_temp = df.sort_values(by=inter_col)
        org_data = [0] + df_temp.units_sold.tolist() 

        comp_data =  [n for n in range(min(org_data),max(org_data)+1)] 
        intp_points = [i for i in comp_data if i not in org_data]

        x = org_data
        y = [0]+df_temp[inter_col].tolist()

        # Values to interpolate
        interpolation_points = intp_points
        # Perform linear interpolation using numpy
        interpolated_values = np.interp(interpolation_points, x, y)
        final_dict = {}

        # Print interpolated values
        for point, value in zip(interpolation_points, interpolated_values):
            final_dict[point] = value

        df_output = pd.Series(final_dict).reset_index()
        df_output.columns = ['metric', '{}'.format(inter_col)]

        df_org = pd.DataFrame({'metric':org_data,'{}'.format(inter_col):y})
        df_output = pd.concat([df_output,df_org],axis=0).sort_values(by='metric')
        df_output['cluster_parent_sku'] = df.cluster_parent_sku.unique()[0]
        df_output = df_output.drop_duplicates()
        return df_output
        
        
    def identify_plateaus(self,time_series, threshold):
        plateaus = []
        plateau_start = None
        
        for i in range(len(time_series) - 1):
            if abs(time_series[i] - time_series[i + 1]) < threshold:
                if plateau_start is None:
                    plateau_start = i
            else:
                if plateau_start is not None:
                    plateaus.append((plateau_start, i))
                    plateau_start = None
        
        if plateau_start is not None:
            plateaus.append((plateau_start, len(time_series) - 1))
        return plateaus

    def cost_function(self,time_series, plateaus):
        try:
            plateau_start,plateau_end = [plateaus[-1]][0]
            dist_max = len(time_series) - plateau_start
            
            plateau_length = plateau_end-plateau_start
            plat_var = sum([np.var(time_series[start:end+1]) for start, end in plateaus])
            last_plat_var = [np.var(time_series[start:end+1]) for start, end in [plateaus][-1]][0]
            num_plats = len(plateaus)
            return last_plat_var*(1/dist_max)
        except:
            return float('inf')
        

    def grid_search_thresholds(self,time_series, thresholds,mean_of_differences):
        best_threshold = None
        best_cost = float('inf')
        
        for threshold_ip in thresholds:

            threshold = mean_of_differences/threshold_ip
            plateaus = self.identify_plateaus(time_series, threshold)
            current_cost = self.cost_function(time_series, plateaus)

            if current_cost < best_cost:
                best_cost = current_cost
                best_threshold = threshold
                

        return best_threshold, best_cost

    def plateau(self,df,sku,curve_column):

        interpol_df = df.copy()

        interpol_df = interpol_df.drop_duplicates()
        time_series = interpol_df[curve_column].tolist()


        differences = np.diff(time_series)
        mean_of_differences = np.mean(differences)


        input_thresholds = np.arange(0.1,20,0.5)
        best_threshold, best_cost = self.grid_search_thresholds(time_series, input_thresholds,mean_of_differences)
        plateau_segments = self.identify_plateaus(time_series, best_threshold)


    #   Visualize the identified plateaus

        # plt.figure(figsize=(20, 6))
        # plt.plot(bucket,time_series,  label='Time Series')

        if len(plateau_segments) == 0:
            plateau_segments = [(len(time_series)-1,len(time_series)-1)]
        sku_name = df.cluster_parent_sku.unique()[0]
    
        pl_points = {time_series[start]:time_series[end] for start,end in plateau_segments}

        df_output_plateau = pd.DataFrame({'plateau_start':pl_points.keys(),'plateau_end':pl_points.values()})
        df_output_plateau['sku_name'] = sku_name
        return df_output_plateau.iloc[[-1]],interpol_df
     
            
    def func_7_90_demand_curves(self):


        cut_off = config.cut_off
        curve_column =  config.curve_column 
#         store_id = config.store_id

        df = self.sales_data_df
        df['sku'] = df['sku'].astype('str')
        self.product_dims = self.product_dims.merge(self.intraweek_seasonality_data,on="DOS")
        df = df.merge(self.product_dims,on='sku')

        # if config.price_per_unit not in df.columns:
        #     df[config.price_per_unit] = df['units_sold']/df['revenue']
            
        # if 'volume_per_unit' not in df.columns:
        #     df['volume_per_unit'] = df['volume']/df['revenue']
            
        self.pre_intra_week = df.copy()
        
        df['units_sold'] = df['units_sold'] * df['Demand_Covered_percentage'] # Change in product dims and here
        df['units_sold'] = df['units_sold'].apply(math.ceil)
        df['revenue'] = df['units_sold'] * df[config.price_per_unit]
        # df['volume_per_unit'] = df['total_pack_size']/1000
        df['volume'] = df['units_sold'] * df['volume_per_unit']

        
        self.post_intra_week = df.copy()
        shelf_depth = self.shelf_master.Depth.min()
        self.sku_master_data['Num_units_per_col'] = (shelf_depth/self.sku_master_data[config.sku_depth]).apply(math.floor)

        df = df.groupby(['pack_size', 'pack_qty', 'yearweek','yearmonth' ,'cluster_parent_sku'],as_index=False)[['units_sold','revenue','volume','volume_per_unit']].mean()   # Changed things here - to elimate poc_id. Taking average of POCs here
        # print(df.shape)
        df['units_sold'] = df['units_sold'].apply(math.ceil)
        
        
        self.df_1 = df.copy()
        
        # self.sku_master_data['cluster_parent_sku']  = df_shelf_dim['cluster_parent_sku']
        df_shelf_dim = self.sku_master_data.copy()
        sku_list = df.cluster_parent_sku.unique()
        
        df_week = df.groupby(['cluster_parent_sku'],as_index=False).yearweek.nunique()
        df_week.columns = ['cluster_parent_sku', 'yearweek_count']

        df_sku_yw = df.groupby(['yearweek','cluster_parent_sku'],as_index=False).agg({'units_sold':'sum','volume':'sum','volume_per_unit':'mean'})
        

        perc_times_df = df_sku_yw.groupby('cluster_parent_sku',as_index=False).apply(self.perc_time).reset_index(drop=True)
        # print(perc_times_df.columns)
        self.df_sku_yw = df_sku_yw.copy()
        # perc_times_df = perc_times_df.loc[perc_times_df['percentage_times_met'] >= cut_off]
        
        self.perc_times_df_copy = perc_times_df.copy()
        perc_times_df['diff_cut_off'] = (perc_times_df['percentage_times_met']-cut_off).apply(abs)
        nearest_to_cut_off = perc_times_df.groupby('cluster_parent_sku')['diff_cut_off'].idxmin()
        result_nearest_to_cut_off = perc_times_df.loc[nearest_to_cut_off, ['cluster_parent_sku',"units_sold" ,'percentage_times_met']]
        result_nearest_to_cut_off.columns = ['cluster_parent_sku','units_sold_desired_dos_cut_off','percentage_times_met']
        result_nearest_to_cut_off = result_nearest_to_cut_off.merge(df_week,on='cluster_parent_sku',how='left')
        perc_times_df = perc_times_df.merge(result_nearest_to_cut_off[['cluster_parent_sku', 'units_sold_desired_dos_cut_off']],on='cluster_parent_sku',how='left')

        perc_times_df_1 = perc_times_df.copy()

        perc_times_df =  df_sku_yw.merge(perc_times_df,on=['cluster_parent_sku', 'units_sold'],how='left')
        
        self.perc_times_df_copy = perc_times_df.copy()
        
        df_collate = perc_times_df.groupby('cluster_parent_sku',as_index=False).apply(self.meet_demand_perc).reset_index(drop=True)

        df_collate['cluster_parent_sku'] = df_collate['cluster_parent_sku'].astype('str')
        df_collate['Total_Volume'] = df_collate.groupby(['cluster_parent_sku'],as_index=False)['volume'].transform('sum')
        
        self.df_collate_copy = df_collate.copy()
        df_lp1 = df_collate.groupby('cluster_parent_sku',as_index=False).apply(self.linear_interpolation,'Meeting_Demand_units_perc').reset_index(drop=True)
        df_lp2 = df_collate.groupby('cluster_parent_sku',as_index=False).apply(self.linear_interpolation,'percentage_times_met').reset_index(drop=True)
        self.df_lp_1 = df_lp1.copy()
        self.df_lp_2 = df_lp2.copy()
        df_lp = df_lp1.merge(df_lp2,on=['cluster_parent_sku','metric'],how='inner')
        

        df_desired_cut_off = df_collate.loc[df_collate['units_sold'] == df_collate['units_sold_desired_dos_cut_off']].drop_duplicates()
        
        
        
        self.df_collate = df_collate        
        self.df_desired_cut_off = df_desired_cut_off
        df_lp = df_lp.merge(df_collate[['cluster_parent_sku','volume_per_unit','Total_Volume']].round(2).drop_duplicates(),how='left',left_on='cluster_parent_sku',right_on='cluster_parent_sku')
        self.df_lp = df_lp.copy()
        
        
        
        df_lp['Volume_covered'] = df_lp['Total_Volume']*df_lp['Meeting_Demand_units_perc']
        demand_curve_df = df_lp.merge(df_shelf_dim[['cluster_parent_sku','Num_units_per_col']],how='left',on='cluster_parent_sku')
        
        demand_curve_df['max_units'] = demand_curve_df.groupby('cluster_parent_sku').metric.transform(lambda x:x.max())
        self.demand_curves_check = demand_curve_df.copy()
        demand_curve_df = demand_curve_df.loc[((demand_curve_df['metric']%demand_curve_df['Num_units_per_col'])==0) | (demand_curve_df['metric'] == demand_curve_df['max_units'])].reset_index(drop=True)
        demand_curve_df['Columns'] = (demand_curve_df['metric']/demand_curve_df['Num_units_per_col']).apply(math.ceil)
        demand_curve_df['max_columns'] = (demand_curve_df['max_units']/demand_curve_df['Num_units_per_col']).apply(math.ceil)
                
        demand_curve_df = demand_curve_df.drop_duplicates()
        demand_curve_df = demand_curve_df.drop('metric',axis=1)
        self.demand_curve_df = demand_curve_df

#          # df_export.to_excel('demand_curves_walmart_units_{}_{}.xlsx'.format(desired_DOS,cut_off),index=False)
        # df_export.to_excel('demand_curves_walmart_{}_{}.xlsx'.format(desired_DOS,cut_off),index=False)


#         img_path = r"{}\Sat_curves_{}_{}".format(file_path,desired_DOS,cut_off)
        
#         Path(img_path).mkdir(parents=True, exist_ok=True)
        df_plateau_collated = pd.DataFrame()
        df_sat = pd.DataFrame()

        for sku in sku_list:
            df_temp = df_lp.loc[df_lp.cluster_parent_sku==sku]
            df_output_plateau,interpol_df = self.plateau(df_temp,sku,curve_column)
            df_plateau_collated = pd.concat([df_plateau_collated,df_output_plateau],axis=0)
            interpol_df['cluster_parent_sku'] = sku
            df_sat = pd.concat([df_sat,interpol_df],axis=0)
        
        # print("=======================",df_sat.shape)
        df_plateau_collated.reset_index(drop=True).sort_values(by="plateau_start")

        df_saturation = df_plateau_collated.merge(df_lp,left_on=['plateau_start','sku_name'],right_on=[curve_column,'cluster_parent_sku'],how='left')
        df_saturation = df_saturation[['cluster_parent_sku','plateau_start','metric','percentage_times_met','Total_Volume','Volume_covered']]
        df_saturation.columns = ['cluster_parent_sku','plateau_start','Num_units_saturation','percentage_times_met_saturation','Total_Volume_saturation','Volume_covered_saturation']
        # df_saturation.columns = [i+'_saturation' for i in df_saturation.columns]


        self.df_desired_cut_off = df_desired_cut_off[['cluster_parent_sku','units_sold','percentage_times_met','Meeting_Demand_units_perc','Total_Volume']].drop_duplicates()
        self.df_desired_cut_off.columns = ["x_cutoff_"+i for i in self.df_desired_cut_off.columns]
        self.df_output = df_saturation.merge(self.df_desired_cut_off ,how='left',left_on='cluster_parent_sku',right_on='x_cutoff_cluster_parent_sku')

        self.df_saturation = df_saturation
        self.df_plateau_collated = df_plateau_collated
        self.intraweek_adjusted_sales_df = df.copy()

#         df_output.to_excel('saturation_curves_{}_{}_walmart.xlsx'.format(desired_DOS,cut_off),index=False)







