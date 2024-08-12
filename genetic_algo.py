# Importing the necessary packages and files
import pandas as pd
import numpy as np
import math
from deap import creator, base, tools, algorithms
import random
from functools import partial
from config import config
from gekko import GEKKO
from contextlib import redirect_stdout
from io import StringIO
from tqdm import tqdm 
import ast
import time
from scoop import futures
import datetime
import polars as pl
# Import the IPython display module
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
    

# df_baseline_soln = pd.read_excel('Baseline_Solution.xlsx')
# nlip_df = pd.read_excel('NLIP_Solution.xlsx',sheet_name='Manufacturer')
# nlip_df['cluster_parent_sku'] = nlip_df['primary_sku'].astype('str')+"_"+nlip_df['sku_detail']

# Discontinued product,Recent / Innovation product

# df_baseline_soln.columns
class optimization:


    """ 
    
    This class contains all the functions that are used in the optimization function.
    It has the functions which ingestes and prepares the data, the necessary functions used to configure the optimizer and 
    the optimization function itself
    
    """
    
    def __init__(self,sku_master_data,sales_data_df,similarity_data,df_sku_dc,df_sku_sat_curve,nlip_output_df,store_area,constraint_master_df,constraint_cols_list,space_constaint_perc_list,constraint_type_map_list,dfShelfMaster):
        self.df_cost_1 = None
        self.fit_func =  []
        self.fitness_lst = []
        self.cost_multiplier = 10e10
        self.sku_master_data =sku_master_data
        self.sales_data_df = sales_data_df
        self.similarity_data = similarity_data
        self.df_sku_dc = df_sku_dc
        self.df_sku_sat_curve = df_sku_sat_curve
        self.nlip_output_df = nlip_output_df
        self.store_area = store_area
        self.constraint_master_df = constraint_master_df
        self.constraint_cols_list = constraint_cols_list
        self.constraint_type_map_list = constraint_type_map_list
        self.space_constaint_perc_list = space_constaint_perc_list
        self.dfShelfMaster = dfShelfMaster


       
    def beyond_sat(self,extend_df):
        max_val_range = extend_df['Max_Cols_range'].max()
        max_val_dc = extend_df['max_columns'].max()
        if max_val_range > max_val_dc:
            df_temp = pd.DataFrame({'Extra_Columns':np.arange(max_val_dc+1,max_val_range+1,1)})

            df_repeated = pd.concat([extend_df] * len(df_temp), ignore_index=True)
            df_repeated = pd.concat([df_temp,df_repeated],axis=1)
            extend_df = pd.concat([extend_df,df_repeated],axis=0)
        return extend_df

    global cost_repo
    cost_repo = []
    global delist_perc_repo
    delist_perc_repo = []
    global individual_repo
    individual_repo = []
    
    def constraint_checker(self,individual):
        self.evaluate_df = self.cat_test_evaluate_df.copy()
        # individual_repo.append(individual)
        
        # Extract the SKU units and area allocation from the individual
        sku_units = individual
        sku_total_actual_area_list = [a*s for a,s in zip(self.actual_areas,sku_units)]
        sku_total_augmented_area_list = [ag*s for ag,s in zip(self.augmented_areas,sku_units)]
        total_actual_area = sum(sku_total_actual_area_list)        
        self.total_aug_area = sum(sku_total_augmented_area_list)  
        # print(self.total_aug_area)
        # print(self.store_area)
        self.delist_sku_perc = len([i for i in sku_units if i == 0])/len(sku_units)
        delist_perc_repo.append(self.delist_sku_perc)

        
        self.evaluate_df['Columns'] = individual
        self.evaluate_df['augmented_area'] = sku_total_augmented_area_list
        self.evaluate_df['actual_area'] = sku_total_actual_area_list
        
        
        if config.plano_constraints_flag:
            for i in range(len(self.constraint_type_map_list)):
                if self.constraint_type_map_list[i] == 'Overall':
                    self.evaluate_df[self.constraint_cols_list[i]] = (self.evaluate_df[self.constraint_cols_list[i]] * self.evaluate_df['actual_area'])/total_actual_area
                    
                if (self.constraint_type_map_list[i] == 'Shelf') | (self.constraint_type_map_list[i] == 'Shelf_Feasibility_Constraint_Less_Than'):
                    self.evaluate_df[self.constraint_cols_list[i]] = self.evaluate_df[self.constraint_cols_list[i]] * self.evaluate_df['augmented_area']
                    
            self.evaluate_cons_agg_df = self.evaluate_df[[i for i in self.constraint_cols_list if ((i.startswith('Constraint_')) | (i == 'Shelf'))]].sum(axis=0).reset_index()
            self.evaluate_cons_agg_df.columns = ['Contraint','Total_Space']
            
            # Comparing
            
            for i in range(len(self.constraint_cols_list)):
                
                
                if ((self.constraint_type_map_list[i] == 'Overall') | (self.constraint_cols_list[i] == 'Shelf')):
                    comp_val = float(self.evaluate_cons_agg_df.loc[self.evaluate_cons_agg_df['Contraint']==self.constraint_cols_list[i],'Total_Space'])
              
                    if (self.constraint_type_map_list[i] == 'Shelf') & ((comp_val > self.space_constaint_perc_list[i]) | (comp_val < self.space_constaint_perc_list[i]*(1-config.shelf_type_constraint_precision))):                    
                        # self.fitness_lst.append(-self.cost_multiplier)
                        return False,
                    
                    elif (self.constraint_type_map_list[i] == 'Overall') & (abs(comp_val-self.space_constaint_perc_list[i]) > config.overall_type_constraint_precision):
                        
                        # print(comp_val-self.space_constaint_perc_list[i])
                        # self.fitness_lst.append(-self.cost_multiplier)
                        return False
                
                    
                if 'Shelf_Feasibility_Constraint_Less_Than' in self.constraint_cols_list[i]:
                    # print('Shelf_Feasibility_Constraint_Less_Than')
                    comp_val = float(self.evaluate_cons_agg_df.loc[self.evaluate_cons_agg_df['Contraint']==self.constraint_cols_list[i],'Total_Space'])
                    if (comp_val <= self.space_constaint_perc_list[i]):
                        # self.fitness_lst.append(-self.cost_multiplier)
                        return False
                    
                    
        if config.overall_space_constraints:
            if  (round(self.total_aug_area,0) > round(self.store_area,0)) | (round(self.total_aug_area,0) < round(self.store_area*(1-config.total_planogram_space_constraint_precision),0)):
                # self.fitness_lst.append(-self.cost_multiplier)
                return False
            
        return True
    
    def evaluate(self,individual,constraint_checker_bool): #Added
        # print(self.constraint_checker(individual))
        if (not self.constraint_checker(individual)) & constraint_checker_bool:
            return self.cost_multiplier,
        # print("appending")
        # print(len(individual_repo))
        individual_repo.append(individual)

        # Extract the SKU units and area allocation from the individual
        sku_units = individual
        sku_total_actual_area_list = [a*s for a,s in zip(self.actual_areas,sku_units)]
        sku_total_augmented_area_list = [ag*s for ag,s in zip(self.augmented_areas,sku_units)]
        total_actual_area = sum(sku_total_actual_area_list)        
        self.total_aug_area = sum(sku_total_augmented_area_list)  
        # print(self.total_aug_area)
        # print(self.store_area)
        self.delist_sku_perc = len([i for i in sku_units if i == 0])/len(sku_units)
        delist_perc_repo.append(self.delist_sku_perc)

        
        
        # cost calculation
        lstDelistSKUs = [sku for sku, ele in zip(self.sku_names,individual) if int(ele) == 0]
        keep_SKUs = [sku for sku, ele in zip(self.sku_names,individual) if int(ele) > 0]
        # self.lstDelistSKUs_copy = lstDelistSKUs
        lstKeysToFilter = [str(sku) + "_" + str(int(ele)) for sku, ele in zip(self.sku_names,sku_units)]
        self.lstKeysToFilter_copy = lstKeysToFilter
        group_cols = ['primary_cluster_parent_sku']
        dfNewWalkAwayRatesAndWeightedSubPrice = self.similarity_evaluate_df_pl.filter(~(pl.col('cluster_secondary_sku').is_in(lstDelistSKUs) |pl.col('primary_cluster_parent_sku').is_in(lstDelistSKUs)))


        if  len(dfNewWalkAwayRatesAndWeightedSubPrice)==0:
            dfNewWalkAwayRatesAndWeightedSubPrice = self.similarity_evaluate_df_pl
            dfNewWalkAwayRatesAndWeightedSubPrice = dfNewWalkAwayRatesAndWeightedSubPrice.with_columns(pl.lit(1).alias('walkaway_rate'))
            dfNewWalkAwayRatesAndWeightedSubPrice = dfNewWalkAwayRatesAndWeightedSubPrice.with_columns(pl.lit(0).alias('weighted_avg_price_of_subs'))
            dfNewWalkAwayRatesAndWeightedSubPrice = dfNewWalkAwayRatesAndWeightedSubPrice.groupby(['primary_cluster_parent_sku']).agg(pl.col('weighted_avg_price_of_subs').mean(),pl.col('primary_volume').mean(),pl.col('walkaway_rate').mean())
            # dfNewWalkAwayRatesAndWeightedSubPrice = pd.DataFrame({'cluster_parent_sku':[],'walkaway_rate':[],'weighted_avg_price_of_subs':[]})
        else:
            dfNewWalkAwayRatesAndWeightedSubPrice = dfNewWalkAwayRatesAndWeightedSubPrice.groupby(['primary_cluster_parent_sku']).agg(pl.col('Weighted_Secondary_PPL').sum(),pl.col('primary_volume').mean(),pl.col('Demand_Transference_Volume').sum())


#           Calculate the sum of 'Weighted_Secondary_PPL' for each 'cluster_parent_sku'
            dfNewWalkAwayRatesAndWeightedSubPrice = dfNewWalkAwayRatesAndWeightedSubPrice.with_columns((1 - (dfNewWalkAwayRatesAndWeightedSubPrice['Demand_Transference_Volume'] / dfNewWalkAwayRatesAndWeightedSubPrice['primary_volume'])).alias('walkaway_rate'))

            # Calculate the sum of 'Weighted_Secondary_PPL' for each 'cluster_parent_sku'
            dfNewWalkAwayRatesAndWeightedSubPrice = dfNewWalkAwayRatesAndWeightedSubPrice.with_columns((dfNewWalkAwayRatesAndWeightedSubPrice['Weighted_Secondary_PPL']/dfNewWalkAwayRatesAndWeightedSubPrice['Demand_Transference_Volume']).alias('weighted_avg_price_of_subs'))
        self.dfNewWalkAwayRatesAndWeightedSubPrice_copy = dfNewWalkAwayRatesAndWeightedSubPrice
        
        dfDemandCurveData_new_wr_pl = self.df_sku_dc_pl.filter(pl.col('Key').is_in(lstKeysToFilter))
        dfDemandCurveData_new_wr_pl = dfDemandCurveData_new_wr_pl.join(dfNewWalkAwayRatesAndWeightedSubPrice, how='left',on='primary_cluster_parent_sku')

        
        dfDemandCurveData_new_wr_pl = dfDemandCurveData_new_wr_pl.with_columns(dfDemandCurveData_new_wr_pl['walkaway_rate'].fill_null(1))
        dfDemandCurveData_new_wr_pl = dfDemandCurveData_new_wr_pl.with_columns(dfDemandCurveData_new_wr_pl['weighted_avg_price_of_subs'].fill_null(dfDemandCurveData_new_wr_pl['primary_price_per_ltr']))

        # Calculate Total Revenue
        dfDemandCurveData_new_wr_pl = dfDemandCurveData_new_wr_pl.with_columns((dfDemandCurveData_new_wr_pl['Total_Volume'] * dfDemandCurveData_new_wr_pl['primary_price_per_ltr']).alias("Total_Revenue"))

        
        
        # Calculate Cost
        
        dfDemandCurveData_new_wr_pl = dfDemandCurveData_new_wr_pl.with_columns((((dfDemandCurveData_new_wr_pl['Total_Volume']-dfDemandCurveData_new_wr_pl['Volume_covered'])*(dfDemandCurveData_new_wr_pl['primary_price_per_ltr']*dfDemandCurveData_new_wr_pl['walkaway_rate']))+((dfDemandCurveData_new_wr_pl['Total_Volume']-dfDemandCurveData_new_wr_pl['Volume_covered'])*((1-dfDemandCurveData_new_wr_pl['walkaway_rate'])*(dfDemandCurveData_new_wr_pl['primary_price_per_ltr']-dfDemandCurveData_new_wr_pl['weighted_avg_price_of_subs'])))).alias('cost'))


        cost = dfDemandCurveData_new_wr_pl['cost'].sum()
        # cost= dfDemandCurveData_new_wr['cost'].sum() # comm
        
        self.df_cost_1 = dfDemandCurveData_new_wr_pl.clone()
        # self.df_cost_1 = dfDemandCurveData_new_wr.copy() #comm
       
        self.fitness_lst.append(cost)
        print(cost)
        return cost,

    
    def genetic_algo(self,iterations,population_size):
        # iterations = 1
        # population_size = 1

# ====================================================================================
        self.dfSimilarityData = self.similarity_data.copy()
        self.df_sku_dc = self.df_sku_dc.merge(self.dfSimilarityData[['primary_cluster_parent_sku','primary_price_per_ltr']].drop_duplicates(),left_on='cluster_parent_sku',right_on='primary_cluster_parent_sku') # Adding the column primary_price_per_ltr to the demand curve dataframe
        
        self.max_values_dc_df = self.df_sku_dc.groupby(['primary_cluster_parent_sku','cluster_parent_sku'],as_index=False)[['Total_Volume','Volume_covered','volume_per_unit','Num_units_per_col','max_units','max_columns','primary_price_per_ltr','Meeting_Demand_units_perc','percentage_times_met','Columns']].max() # Creating a dataframe with the max values in the demand curves for all the SKUs 
        
        self.dfSimilarityData["Weighted_Secondary_PPL"] = self.dfSimilarityData['Demand_Transference_Volume'] * self.dfSimilarityData["secondary_price_per_ltr"] # Weighted Secondary price per liter helps us to get a sense of what is the bearing the secondary sku has in demand transference based on the demand it aborbs from the primary SKU.
        self.dfSimilarityData['cluster_secondary_sku'] = self.dfSimilarityData["secondary_sku"].astype(str) +"_"+self.dfSimilarityData['secondary_sku_detail']
        
        self.cat_test_df = self.nlip_output_df.copy()     
        self.cat_test_df['cluster_parent_sku'] = self.cat_test_df['sku'].astype(str) + "_" +self.cat_test_df['sku_detail']
        
# =============================================================================
# The saturation dataframe gives us the number of units at saturation, here we are converting it into number of column and then rouding up the nearest mutiple of stacking units        
# =============================================================================
        self.cat_test_df = self.cat_test_df.merge(self.df_sku_sat_curve[['cluster_parent_sku','Num_units_saturation']],how='left',on='cluster_parent_sku')
        self.cat_test_df['cut_off_columns'] = (self.cat_test_df['Num_units_saturation']/self.cat_test_df['Num_units_per_col'])
        self.cat_test_df['cut_off_columns'] = np.ceil((self.cat_test_df['cut_off_columns']/self.cat_test_df['Stacked_Units'])*self.cat_test_df['Stacked_Units'])
        self.cat_test_df['cut_off_columns'] = self.cat_test_df[['cut_off_columns','Solution']].max(axis=1)
# In the nlip output for a SKU if the solution is beyond the saturation point, use the nlip solution as the max columns else keep max as sat points. This is to ensure that the solution can be beyond saturation point if needed. NLIP explores posibilities beyond saturation point. 

# =============================================================================

    # Creating a new dataframe for calculating SKUs that are pareto SKUs in terms of revenue market share. 
        df_cat_pareto =self.cat_test_df.sort_values(by=['primary_sku_rev_market_share'],ascending=False) # Sorting all the skus as per the market share in descending order. This is done to make sure all the top SKUs are in the top
        df_cat_pareto['ms_cum'] = df_cat_pareto['primary_sku_rev_market_share'].cumsum() # Creating a cumulative sum based on market share. The cumulative sum will help us compare the SKUs are big contributers and the ones that are marginal contributors. Example, if we set the pareto limit to 80%, all the SKUs that constribute to 80% of the market share will be flagged as pareto.
        self.cat_test_df =  self.cat_test_df.merge(df_cat_pareto[['cluster_parent_sku','ms_cum']],on='cluster_parent_sku',how='left') # Merging this dataframe with the cat_test_df with cumulative sum column.
        ms_cum_sum = self.cat_test_df['ms_cum'].tolist() # Creating list with respective cumulative sum market shares

        self.rev_market_share = self.cat_test_df['primary_sku_rev_market_share'].tolist() # Creating list with market shares
        self.sku_names = self.cat_test_df.cluster_parent_sku.tolist()
        self.sku_id = self.cat_test_df.sku.tolist()

        units_per_col = self.cat_test_df.Num_units_per_col.tolist() # Number of units per column
        self.min_col_1 = self.cat_test_df.Min_Cols_1.tolist() # A list of the min_cols_1 for every SKU
        self.actual_areas = self.cat_test_df.Actual_Space_Per_Column.tolist()  # A list of the actual area occupied by every SKU
        self.augmented_areas = self.cat_test_df.Augmented_Space_Per_Column.tolist()  # A list of the augmented area occupied by every SKU
    
        # Calculating_max_units for a sku in a shelf.
        self.cat_test_df['Max_Cols'] = self.cat_test_df[['Max_Cols', 'Initialization_Cols']].max(axis=1)
        if config.boolHalfShelfConstraint: # If this constraint is set to true then an SKU will only get as many columns that can be accomodated in the Shelf_width_threshold of the overall width of one shelf.
            shelf_width = self.dfShelfMaster.groupby(['Shelf'])['Width'].sum().min() # If there are shelfs with varying lengths, then the shelf with the least width will be taken
            self.cat_test_df['Max_cols_in_a_shelf'] = np.floor(((shelf_width*config.Shelf_width_threshold)/self.cat_test_df['Width']))*self.cat_test_df.Stacked_Units # Calculating the number of facings as per the condition. 
            self.cat_test_df['Max_Cols'] = self.cat_test_df[['Max_cols_in_a_shelf','Max_Cols']].min(axis=1) # Taking the min of the max cols and max facings in one shelf. If the max cols is less than max facings that can be kept in a shelf, then the value in max_cols can be retained
        
        
        self.cat_test_df["Initialization_Cols"] = self.cat_test_df['Solution'].astype(int) # The values in Solution will be the initialization point

 # If the solution provided by NLIP(Solution) or the custom_initilized value is beyond the max columns in the demnand curves, then the value in Initialization column will be the upper limit for that SKU
        self.cat_test_df['Min_Cols_2'] = self.cat_test_df[['Min_Cols_2','Max_Cols']].min(axis=1) # Sanity check in a case where the min_cols2 is greater than max_cols. Ex - Min_Cols_2 is the next multiple of stacked units after min_cols_1. Assuming min_cols_1 is 2, then min_cols_2 will be 4. But the max_columns can be just 2 for an sku. Then the min_cols_2 will be greater than max_cols. This check will check and take care of such instances by overwriting min_cols_2 as 2
        
        ranges = [(lb ,ub,stack) for lb,ub,stack in zip(self.cat_test_df['Min_Cols_2'],self.cat_test_df['Max_Cols'].tolist(),self.cat_test_df.Stacked_Units.tolist())] # Creating the ranges for every sku. A tuple like this (min_val, max_val, stacked_unit_multiple) will created
        print(ranges)

        self.cat_test_evaluate_df = self.cat_test_df[[i for i in self.constraint_cols_list if 'Half_Shelf_Constraint' not in i]] # This is a constraint taking into consideration the half_shelf_limit in NLIP. Since we are using a different method as shown in above for boolHalfShelfConstraint, it is removed from the cat_test_df


# =============================================================================
# Instances where the solution from NLIP or the custom initilization column us beyond the max values in the demand curves, the demand curve has to be extended until that point. The below snippet does this operation using the logic in the function beyond_sat        
# =============================================================================
        
        self.max_range_sku_df = self.cat_test_df[['cluster_parent_sku','Max_Cols']].drop_duplicates() # Subsetting cat_test_df dataframe as we require only Max_Cols, which is wither max columns on demand curves or max in initilization cols
        self.max_range_sku_df.columns = ['primary_cluster_parent_sku','Max_Cols_range'] # remanimg these columns to avoid confusion
        self.max_values_dc_df = self.max_values_dc_df.merge(self.max_range_sku_df,on='primary_cluster_parent_sku') # Merging the above dataframe "max_values_dc_df" with datafrem having max values from demand curves for comparision
        self.max_values_exteded_dc_df = self.max_values_dc_df.groupby(['primary_cluster_parent_sku'],as_index=False).apply(self.beyond_sat) # Grouping by primary_cluster_parent_sku to apply the function at this level
        self.max_values_exteded_dc_df = self.max_values_exteded_dc_df.dropna(axis=0) # Dropping nulls values. Nulls will be present for SKUs where the function did not return any values. This will happen for SKUs that are having max columns below the max values in demand curves
        if 'Extra_Columns' in self.max_values_exteded_dc_df.columns: # If there are no extra columns, which means no SKU is needed to go beyond the demand curves, the column "Extra_Columns" will not be created
            self.max_values_exteded_dc_df['Columns'] = self.max_values_exteded_dc_df['Extra_Columns']  # Re-asigning values the column "Columns"  with the values in "Extra Columns" since the extended values from the output of the function beyond_sat will be in the column "Extra_Columns"
            self.max_values_exteded_dc_df = self.max_values_exteded_dc_df.drop('Extra_Columns',axis=1) # Dropping the column , "Extra columns" since it will have the same values now as "Columns" and this column is not present in self.df_sku_dc
        self.df_sku_dc = pd.concat([self.df_sku_dc,self.max_values_exteded_dc_df.reset_index(drop=True)],axis=0) # It is to be made sure that the column names in the dataframe self.max_values_exteded_dc_df should have the same column names and the columns should be in the same order as the dataframe self.df_sku_dc.
        
        
        self.df_sku_dc["Key"] = self.df_sku_dc["primary_cluster_parent_sku"].astype(str) + "_" + self.df_sku_dc["Columns"].round(0).astype(int).astype(str) #The Key value will serve as a guide in the evaluate function to filetr out the primary sku and the given column in the individual from the demand curve dataframe - self.df_sku_dc

        self.similarity_evaluate_df = self.dfSimilarityData[['primary_cluster_parent_sku','Demand_Transference_Volume','Weighted_Secondary_PPL','primary_volume','cluster_secondary_sku']] # The subsequent processes require only the specified columns from the similarity dataframe. To prevent redundancy, filtering the necessary columns
        
        self.similarity_evaluate_df_pl = pl.from_dataframe(self.similarity_evaluate_df) # The objective function- evaluate, uses polar dataframes, hence converting the similarity_evaluate_df to polar datframe to make it suitable for consumption.
        self.df_sku_dc_pl = pl.from_dataframe(self.df_sku_dc)  # The objective function- evaluate, uses polar dataframes, hence converting the df_sku_dc to polar datframe to make it suitable for consumption.

        # Create the fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,)) # If the weights are +1, it is trying maximize the cost, -1, will minimize the cost
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()

        def main():

            pop = toolbox.population(n=population_size) # Registering the population size
            hof = tools.HallOfFame(30) # HallOfFame store the top individuals from all the generations

            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            start_time = datetime.datetime.now()
            # print(start_time)
            # toolbox.register("map", futures.map)
            # pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.5, ngen=iterations,stats=stats, halloffame=hof, verbose=False) # Uncomment this line if progress bar is not needed

# =============================================================================
# TQDM provides an graphic progress bar which gets updated in real time as the number of interations get completed. 
# =============================================================================
            with tqdm(total=iterations) as pbar:
                for self.iteration_num in range(iterations):
                    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.3, ngen=1,
                                        stats=stats, halloffame=hof, verbose=False) #cxpb is the probability of crossover and mutpb is the probability of mutation 

                    pbar.update(1)

# =============================================================================#
            
            return pop, logbook, hof 

        

        pre_initialization_ind = self.cat_test_df["Initialization_Cols"].tolist() # Instantiating the pre_initialization_ind with the initialization values from the cat_test_df dataframe. The list pre_initialization_ind will be provided to the optimization as the starting point

        # Initialization using NLIP Solution
        for i in range(len(self.sku_names)): # The initilization is done inidividually to every sku.      
            random.seed(42)
            attr_name = f"attr_int_{i}"
            init_list = [pre_initialization_ind[i]] # this list init_list stores the initilized value for every sku that is active in the current loop
            attr_generator = partial(random.choice,init_list) # The attr_generator is a package function that usses random.choice, since only one value is there in the init_list, that specif value will be selected
            toolbox.register(attr_name, attr_generator) # the tool box registers the initilized value for the sku in the index of the current iteration in the loop

        def init_individual():
            return creator.Individual([getattr(toolbox, f"attr_int_{i}")() for i in range(len(self.sku_names))]) 


        toolbox.register("individual", init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # print(ranges)
        
        
        def custom_crossover(ind1, ind2): 
            
            size = min(len(ind1), len(ind2)) # The min of the sizes of the two individuals are stored in this variable, this is to check that niether of them are not empty lists 
            # Storing the original individuals passed to the function to have a copy befor the crossover happens
            org_ind1 = ind1
            org_ind2 = ind2

            if size > 1: # Checking if both individuals are having values i.e, the number elements are more than 1
                # while self.constraints_checker_flag_crx == False:
                # cxpoint = random.randint(1, size - 1) # In case of a single point random crossover, a random point is selected and partion is made at the point or index in both the individuals. The alternative elements on alternative sides are swapped between the two parent individuals and child individuals are created
                for c in range(len(ind1)): # The custom logic here emplys a multi point point cross over, where the min of two elements are stored in one individual and max is stored in another individual
                    ind1[c] = max(ind1[c],ind2[c]) 
                    ind2[c] = min(ind1[c],ind2[c])
                # ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
                
                # Returning only individuals that are having better costs
                return_ind1 = ind1 if self.evaluate(ind1,True)[0] < self.evaluate(org_ind1,True)[0] else org_ind1 # Added
                return_ind2 = ind2 if self.evaluate(ind2,True)[0] < self.evaluate(org_ind2,True)[0] else org_ind2 # Added
                    # self.constraints_checker_flag = ((self.constraint_checker(ind1)) & (self.constraint_checker(ind2)))
            return return_ind1, return_ind2
        
        
        def custom_mutate(individual, indpb):

            for i in range(len(individual)):
                if random.random() <= indpb: # Mutation happens on a probailistic value for every sku. If the value of indpb is 0.15, it implies that there is 15% probability that the SKU will undergo mutation
                    min_range, max_range, stack_units = ranges[i]  #Selecting the maximum, minimum, and stacked values from the 'ranges' variable is necessary. This ensures that a series of numbers is provided for the mutated value to be obtained
                    # To control the range of mutation, we only give 2 twice the stacked units as the spectrum of numbers in which the mutation can happen. Ex - if the current value or facings for an sku is 8 and the stacked units is 2, the mutation can happen between the raneg [4,6,8,10,12]. If the min value is 2 and the max value is 30, the optimization might select value that are not aliginig with conditions of convergence or with the interest of better cost
                    # The multiple "2" for the stacked units is selected heuristically, can be altered to experiment for other casses
                    min_adj_val = individual[i] - stack_units * 2
                    max_adj_val = individual[i] + stack_units * 2 
                    # * math.ceil((self.rev_market_share[i]*100))
                    self.init_list_mut = [self.min_col_1[i]] + list(np.arange(max(min_range,min_adj_val), min(max_range+1,max_adj_val),stack_units)) # as shown above, the list provided for mutation will be appended with value in self.min_col_1 at the respective index of the iteration. if min_col_1 is 0, then the list will be [0,4,6,8,10,12]
                    choice_prob = [(prob+1)/(sum(self.init_list_mut)+len(self.init_list_mut)) for prob in self.init_list_mut] # This expression calculates the probability for each element prob in self.init_list_mut. (prob+1): This adds 1 to each element prob. This addition is done to avoid division by zero in the next step.sum(self.init_list_mut): This calculates the sum of all elements in self.init_list_mut.len(self.init_list_mut): This returns the length of self.init_list_mut, i.e., the number of elements in the list.sum(self.init_list_mut) + len(self.init_list_mut): This calculates the total sum of the elements in self.init_list_mut plus the length of the list.(prob+1)/(sum(self.init_list_mut) + len(self.init_list_mut)): This expression divides (prob+1) by the total sum calculated above, giving the probability of each element prob.
                    individual[i] = random.choices(self.init_list_mut,weights=choice_prob,k=1)[0] # random.choices is used to select a random variable from the muation list with each number having the probabilities as calculated in the previous line. Higher the number, the higher is the probability of that number getting selected
            return individual,
        
            
        toolbox.register("evaluate", partial(self.evaluate, constraint_checker_bool=True))
        toolbox.register("mate", tools.cxOnePoint)
        # toolbox.register("mate", custom_crossover)
        
        toolbox.register("mutate", custom_mutate, indpb = 0.15) # indpb is Probability of each individual to undergo mutation
        toolbox.register("select", tools.selTournament, tournsize=30)

        try:
            self.cost = self.df_cost_1['cost'].sum() # If the output of the optimization is a dataframe, then cost will be calculated as the sum of the cost column in the df_cost_1 dataframe. If the evaluate function exited in the constraint checker, then the cost multiplier will be returned. Therefore this operation will fail.
        except:
            self.df_cost_1 = self.cost_multiplier # When the cost_multiplier is returned, df_cost_1 holds the value of the cost_multiplier
            

        pop, logbook, hof = main()
        

# =============================================================================
#         The below code gets the least cost from the fitness_lst list and finds the individual in the same index of individual_repo
# =============================================================================
        if isinstance(self.df_cost_1, pl.DataFrame): # If df_cost_1 is a dataframe, i.e., all the constraints are satisfied, the repos for fitness and individuals are checked to get the best value

            min_cost_from_list = min([i for i in self.fitness_lst if i > 0]) # Select Min cost above zero since the cost multiplier is a very large negative value and therefore least value in the list will be the cost_multiplier. This is in the case when we are maximizing the fitness value. If the weights = -1, which is a minimization problem, all the fitness values will be above zero. This logic satisfies both the casses
            min_cost_index = self.fitness_lst.index(min_cost_from_list) # Finding the index of the least cost from the fitness_lst list
            self.individual_repo_1 = individual_repo
            sku_quantities = individual_repo[min_cost_index] # Getting the individual from the in the same index of the leat cost from the individual_repo
            best_individual = individual_repo
            fitness_value = self.evaluate(sku_quantities,True) # Running the evalute function again on the individual to get all the required dataframes in the function which will be used for further computations 

       
        if (config.iteration == 1) & (config.population == 1):# If the number of iteration is 1 and population is set to 1, the evaluate function will run with the initialized individual 
            best_individual = pre_initialization_ind
            sku_quantities = pre_initialization_ind
            fitness_value = self.evaluate(sku_quantities,True)
                
        if not isinstance(self.df_cost_1, pl.DataFrame):# When df_cost_1 is the cost_multiplier, i.e., all the constraints are not satisfied, an individual with all the SKUs with 0 columns/facings will be passed to the objective or the evaluate function.
            sku_quantities = [0]*len(self.sku_names) # If the passed input is all 0, and it doesn't satisfy the plano and overall constraints if set to true, the evaluate function will still return a cost multipler
            best_individual = sku_quantities
            fitness_value = self.evaluate(sku_quantities,False)

    #        df_output = pd.DataFrame()
        self.cost_repo = cost_repo
        self.individual_repo = individual_repo
        self.delist_perc_repo = delist_perc_repo

        df_final_output = pd.DataFrame({
    "cluster_parent_sku": self.sku_names,
    "Number of columns": sku_quantities,
    "Actual Area Occupied": [round(qty * act_area, 2) for qty, act_area in zip(sku_quantities, self.actual_areas)],
    "Augmented Area  Occupied": [round(qty * agm_area, 2) for qty, agm_area in zip(sku_quantities, self.augmented_areas)],
    "Units_per_col": units_per_col
})

        self.best_individual_output = best_individual
        return df_final_output,fitness_value,self.cat_test_df,self.df_cost_1,best_individual


    def optimization_run(self):
        
        if self.nlip_output_df[config.MIN_COLS_IF_KEPT].equals(self.nlip_output_df[config.MAX_COLS_IF_KEPT]):
            config.iteration = 1
            config.population = 1
            config.plano_constraints_flag = False
            config.overall_space_constraints = False
        
        output_df_1,fitness_value,df_cat,self.cost_2,sku_quantities = self.genetic_algo(config.iteration,config.population)

        if isinstance(self.cost_2, pl.DataFrame):
            self.cost_2 = self.cost_2.to_pandas()
            self.cost_2['volume_cost'] = self.cost_2['cost']/self.cost_2['primary_price_per_ltr']
            self.cost_2['sku'] = self.cost_2['cluster_parent_sku'].apply(lambda x:x.split("_")[0])
            self.cost_2 = self.cost_2.merge(self.cat_test_df[['sku','manufacturer','Constraint_1', 'Constraint_2', 'Constraint_3','Actual_Space_Per_Column', 'Augmented_Space_Per_Column']],on='sku',how='left')
            self.cost_2['Solution_Actual_Space_Occupied'] = self.cost_2['Actual_Space_Per_Column'] * self.cost_2['Columns']
            self.cost_2['Solution_Augmented_Space_Occupied'] = self.cost_2['Augmented_Space_Per_Column'] * self.cost_2['Columns']
           
            
            
        output_df_1['Num_units'] = output_df_1['Number of columns'] * output_df_1['Units_per_col']
        
        self.output_df_1 = output_df_1
        self.df_business_output_prep = (
                output_df_1[['cluster_parent_sku', 'Units_per_col', 'Number of columns','Num_units','Actual Area Occupied','Augmented Area  Occupied']].merge(
        self.sku_master_data[['cluster_parent_sku', 'sku', 'sku_detail', config.style, config.ptc_segment, config.manufacturer, config.brand, 'pack_size',
                          'pack_qty', 'pack_type','total_pack_size', 'price_per_ltr', 'overall_volume_market_share', 'Height',
                          'Width', 'Depth', 'revenue']],
        how='left',
        on='cluster_parent_sku'
        )
        )

        self.output_df_2 = self.output_df_1.merge(self.cat_test_df,on='cluster_parent_sku',how='left')
        self.df_business_output_prep = self.df_business_output_prep.merge(self.cat_test_df[['cluster_parent_sku','Min_Cols_1','Min_Cols_2','Max_Cols','Actual_Space_Per_Column','Augmented_Space_Per_Column','Actual_Space_Per_Column','Stacked_Units']],left_on='cluster_parent_sku',right_on='cluster_parent_sku',how='left')
        
        self.df_business_output_prep = self.df_business_output_prep.merge(self.df_sku_sat_curve,on='cluster_parent_sku',how='left')
        self.df_business_output_prep['Saturation_col_rounded_off'] = (self.df_business_output_prep['Num_units_saturation']/self.df_business_output_prep['Units_per_col']).apply(math.ceil)
        self.df_business_output_prep['Saturation_revenue'] = self.df_business_output_prep['Volume_covered_saturation']*self.df_business_output_prep['price_per_ltr']
        
        self.df_business_output_prep['x_cutoff_col_rounded_off'] = (self.df_business_output_prep['x_cutoff_units_sold']/self.df_business_output_prep['Units_per_col']).apply(math.ceil)
        self.df_business_output_prep['x_cutoff_volume_covered'] = self.df_business_output_prep['x_cutoff_Meeting_Demand_units_perc']*self.df_business_output_prep['x_cutoff_Total_Volume']
        
        self.df_business_output_prep['x_cutoff_revenue'] = self.df_business_output_prep['x_cutoff_volume_covered']*self.df_business_output_prep['price_per_ltr']
        
        self.df_business_output_prep = self.df_business_output_prep.merge(self.df_sku_dc[['cluster_parent_sku','Volume_covered','Columns','max_columns','percentage_times_met']],left_on=['cluster_parent_sku','Number of columns'],right_on=['cluster_parent_sku','Columns'],how='left')
        # print(self.df_business_output_prep.isnull().sum())
        self.df_business_output_prep['solution_Revenue_covered'] = self.df_business_output_prep['Volume_covered']*self.df_business_output_prep['price_per_ltr']
        # self.df_business_output_prep['Total_Volume'] = self.df_business_output_prep['max_columns']*self.df_business_output_prep['Units_per_col']*(self.df_business_output_prep['total_pack_size'])
        
        self.df_business_output_prep  = self.df_business_output_prep.merge(self.cost_2[['cluster_parent_sku','Total_Volume']],on='cluster_parent_sku',how='left')
        self.df_business_output_prep['Total_revenue'] = self.df_business_output_prep['Total_Volume']*self.df_business_output_prep['price_per_ltr']
        # print(self.df_business_output_prep.isnull().sum())
        self.df_business_output_prep['% Cols Met'] = self.df_business_output_prep['Number of columns']/np.where(self.df_business_output_prep['Saturation_col_rounded_off']==0,1,self.df_business_output_prep['Saturation_col_rounded_off'])
        # np.where(self.df_business_output_prep['Saturation_col_rounded_off']==0,0,self.df_business_output_prep['Number of columns']/self.df_business_output_prep['Saturation_col_rounded_off'])
        
        self.df_business_output_prep['% Demand Met'] = self.df_business_output_prep['Volume_covered']/self.df_business_output_prep['Volume_covered_saturation']
        self.df_business_output_prep['% Times met'] = self.df_business_output_prep['percentage_times_met']/self.df_business_output_prep['percentage_times_met_saturation']
                
        
        self.df_business_output_prep.rename(columns = { 'sku':'SKU ID', 'sku_detail':'SKU Name', config.manufacturer:'Manufacturer', config.brand:'Brand', 'pack_size':'Pack Size', 'pack_qty':'Pack Qty','pack_type':'Pack Type', 'price_per_ltr':'PPL', 'overall_volume_market_share':'Volume MS', 'Width':'Width', 'Units_per_col':'Units Per Col', 'Space_per_column':'Space Per Column', 'lower_bound_in_cols':'Min Cols Constraint', 'Number of columns':'Solution', 'x_cutoff_col_rounded_off':'x_90 Cols', 'Saturation_col_rounded_off':'Saturation Cols','max_columns':'Max Cols', 'Volume_covered':'Volume Covered Solution', 'x_cutoff_volume_covered':'x_90 Volume', 'Volume_covered_saturation':'Saturation Volume', 'Total_Volume':'Total Volume', 'solution_Revenue_covered':'Revenue Covered Solution', 'x_cutoff_revenue':'x_90 Revenue', 'Saturation_revenue':'Saturation Revenue','Total_revenue':'Total Revenue','% Cols Met':'% Cols Met','% Demand Met':'% Demand Met','% Times met':'% Times met', 'Depth':'Depth'},inplace=True)
        
        
        # self.df_business_output_prep['Space_Used'] =  self.df_business_output_prep['Space Per Column'] * self.df_business_output_prep['Solution']
        
        self.df_business_output = self.df_business_output_prep[['SKU ID'
        ,'SKU Name'
        ,'Manufacturer'
        ,'Brand'
        ,'Pack Size'
        ,'Pack Qty'
        ,'Pack Type'
        ,'PPL'
        ,'Volume MS'
        ,'Height'
        ,'Width'
        ,'Depth'
        ,'Units Per Col'
        ,'Stacked_Units'
        ,'Solution'
        ,'Actual_Space_Per_Column'
        ,'Actual Area Occupied'
        ,'Augmented_Space_Per_Column'
        ,'Augmented Area  Occupied'
        ,'Min_Cols_1'
        ,'Min_Cols_2'
        ,'x_90 Cols'
        ,'Saturation Cols'
        ,'Max Cols'
        ,'Volume Covered Solution'
        ,'x_90 Volume'
        ,'Saturation Volume'
        ,'Total Volume'
        ,'Revenue Covered Solution'
        ,'x_90 Revenue'
        ,'Saturation Revenue'
        ,'Total Revenue'
        ,'% Cols Met'
        ,'% Demand Met'
        ,'% Times met'
        ]]
        

        self.df_business_output = self.df_business_output.merge(self.constraint_master_df,left_on='SKU ID',right_on='sku')
        self.df_business_output[['% Demand Met','% Times met']] = self.df_business_output[['% Demand Met','% Times met']].fillna(0)
        
        self.df_plano_output_prep = self.sales_data_df.groupby(['sku','sku_detail','pack_qty',config.ptc_segment, config.style, 'pack_type','pack_size','Height', 'Width', 'Depth',config.brand],as_index=False)[['units_sold','revenue','volume']].sum()
        # self.df_plano_output_prep['poc_id'] = np.where(self.df_plano_output_prep['poc_id'].apply(lambda x: isinstance(x, str)), 1, self.df_plano_output_prep['poc_id'])

        self.df_plano_output_prep['cluster_name'] = 'Overall'
        self.df_plano_output_prep['category'] = 'Beer'

        df_mom_growth_ros = self.sales_data_df.groupby(['sku','yearmonth'],as_index=False)['units_sold'].sum()
        df_mom_growth_ros['growth_mom'] = df_mom_growth_ros.groupby(['sku'],as_index=False)['units_sold'].diff()
        df_mom_growth_ros['growth'] = df_mom_growth_ros.groupby(['sku'],as_index=False)['growth_mom'].transform('mean')
        df_mom_growth_ros['sku_presence_in_months'] = df_mom_growth_ros.groupby(['sku'],as_index=False)['yearmonth'].transform('nunique')
        df_mom_growth_ros['ros'] = df_mom_growth_ros.groupby(['sku'],as_index=False)['units_sold'].transform('mean')
        df_mom_growth_ros = df_mom_growth_ros[['sku','growth','ros','sku_presence_in_months']].drop_duplicates()
        self.df_plano_output_prep = self.df_plano_output_prep.merge(df_mom_growth_ros,on = 'sku',how='left')
        self.df_plano_output_prep['growth'] = np.where(self.df_plano_output_prep['sku_presence_in_months']==1,0,self.df_plano_output_prep['growth'])
        # self.df_plano_output_prep['sku'] = self.df_plano_output_prep['sku'].astype('int64')
        self.df_plano_output_prep = self.df_plano_output_prep.merge(self.df_business_output_prep[['SKU ID','Manufacturer','Num_units']],left_on = 'sku',right_on = 'SKU ID',how='left')
        

        self.df_plano_output_prep.rename({'Num_units':'facing','Height':'sku_height','Width':'sku_width','Depth':'sku_depth','Manufacturer':'manufacturer'},inplace=True,axis=1)
        self.df_plano_output_prep['count_in_cluster'] = ''
        self.df_plano_output_prep['innovation_promotion_flag'] = ''
        self.df_plano_output_prep['door_position_from_left'] = ''
        self.df_plano_output_prep['shelf_position_from_top'] = ''
        # print(df_plano_output_prep.columns)

        self.df_plano_opt_output = self.df_plano_output_prep[['sku','sku_detail','pack_qty','cluster_name',config.ptc_segment, config.style,'pack_type','pack_size','category','manufacturer','sku_width','sku_height','sku_depth','units_sold','revenue','volume','growth','ros','count_in_cluster','sku_presence_in_months','innovation_promotion_flag','facing','door_position_from_left','shelf_position_from_top',config.brand]]

        self.output_df_1=output_df_1
        self.fitness_value=fitness_value

        self.sku_quantities=sku_quantities


