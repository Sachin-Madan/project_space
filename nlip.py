# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:03:55 2024

@author: 40107964
"""

import pandas as pd
import numpy as np
from gekko import GEKKO
import re
from config import config


class NLIP:
    
    def __init__(self, dfDemandCurveData, dfSimilarityData, dfSKUMaster, dfConstraintMaster, dfConstraintTypeMapping, dfShelfMaster, dfProductDims):
        
        self.dfDemandCurveData = dfDemandCurveData
        self.dfSimilarityData = dfSimilarityData
        self.dfSKUMaster = dfSKUMaster
        self.dfConstraintMaster = dfConstraintMaster
        self.dfShelfMaster = dfShelfMaster
        self.dfProductDims = dfProductDims
        self.dfConstraintTypeMapping = dfConstraintTypeMapping
        # self.cat_exp_plano_output = cat_exp_plano_output
        
    def __calculate_new_walkaway_rates(self, group):
        dictNewFeatures = {}
        dictNewFeatures['walkaway_rate'] = 1 - (group['Demand_Transference_Volume'].sum() / group['primary_'+config.VOLUME].mean())
        dictNewFeatures['weighted_avg_price_of_subs'] = group["Weighted_Secondary_PPL"].sum() / group['Demand_Transference_Volume'].sum()
        return pd.Series(dictNewFeatures)

    def __ObjectiveFunction(self, x):
        setUniqueKeys = set(self.dfDemandCurveData["Key"])
        lstQueryList = list(zip(self.lstUniqueSKUs, [i.value for i in x]))
        self.lstQueryList_1 = lstQueryList
        # print(lstQueryList)
        lstDelistSKUs = [sku for sku, ele in lstQueryList if ele.value == [0.0]]
        # print(lstDelistSKUs)
        setKeysToFilter = set([str(sku) + "_" + str(int(round(float(str(ele.value).replace("[","").replace("]","")), 0))) for sku, ele in lstQueryList])
        lstKeysToCreate = list(setKeysToFilter - setUniqueKeys)
        if len(lstKeysToCreate) != 0:
            for key in lstKeysToCreate:
                dfToExtendDemandCurve = self.dfMaxColKeys.loc[self.dfMaxColKeys[config.CLUSTER_PARENT_SKU]==int(key.split("_")[0])].copy()
                dfToExtendDemandCurve["Key"] = key
                dfToExtendDemandCurve["Columns"] = int(key.split("_")[1])
                self.dfDemandCurveData = pd.concat([self.dfDemandCurveData, dfToExtendDemandCurve], ignore_index=True)
        dfNewWalkAwayRatesAndWeightedSubPrice = self.dfSimilarityData[~((self.dfSimilarityData["secondary_"+config.CLUSTER_PARENT_SKU].isin(lstDelistSKUs)) | (self.dfSimilarityData["primary_"+config.CLUSTER_PARENT_SKU].isin(lstDelistSKUs)))]
        if  len(dfNewWalkAwayRatesAndWeightedSubPrice)==0:
            dfNewWalkAwayRatesAndWeightedSubPrice = pd.DataFrame({"primary_"+config.CLUSTER_PARENT_SKU:[],'walkaway_rate':[],'weighted_avg_price_of_subs':[]})
        else:
            dfNewWalkAwayRatesAndWeightedSubPrice = dfNewWalkAwayRatesAndWeightedSubPrice.groupby(["primary_"+config.CLUSTER_PARENT_SKU], as_index=False).apply(self.__calculate_new_walkaway_rates).reset_index(drop=True)
        dfIterationOutput = self.dfDemandCurveData[self.dfDemandCurveData["Key"].isin(setKeysToFilter)].rename(columns={config.CLUSTER_PARENT_SKU:"primary_"+config.CLUSTER_PARENT_SKU})
        dfIterationOutput = dfIterationOutput.merge(dfNewWalkAwayRatesAndWeightedSubPrice[["primary_"+config.CLUSTER_PARENT_SKU, 'walkaway_rate', 'weighted_avg_price_of_subs']],how='left',on="primary_"+config.CLUSTER_PARENT_SKU)
        dfIterationOutput['walkaway_rate'].fillna(1,inplace=True)
        # dfIterationOutput['weighted_avg_price_of_subs'].fillna(dfIterationOutput['primary_price_per_ltr'],inplace=True)
        dfIterationOutput['weighted_avg_price_of_subs'].fillna(0,inplace=True)
        dfIterationOutput['cost'] = ((dfIterationOutput['Total_Volume'] - dfIterationOutput['Volume_covered'])*dfIterationOutput['primary_'+config.PRICE_PER_LTR]*dfIterationOutput['walkaway_rate'])+((dfIterationOutput['Total_Volume'] - dfIterationOutput['Volume_covered'])*(1-dfIterationOutput['walkaway_rate'])*(dfIterationOutput['primary_'+config.PRICE_PER_LTR]-dfIterationOutput['weighted_avg_price_of_subs']))
        return dfIterationOutput['cost'].sum()
    
    def __RecursiveSolver(self):
        # print("Recursive Solver Started")
        self.m.solve(debug=False)
        # print("Appstatus",self.m.options.APPSTATUS)
        if self.m.options.APPSTATUS == 0:
            # print("If Block Started")
            with open(self.m.path+'\\infeasibilities.txt', 'r') as InfeasibilityFile:
                contents = InfeasibilityFile.read()
            srEquationResiduals = pd.Series([], dtype=float)
            for i in range(len(self.dfNLIPConstraintIndexMapping)):
                srEquationResiduals[i] = abs(float(re.findall(r"\S+\s+\S+\s+ss.Eqn.{}.".format(i+1),contents)[1].split()[0]))
            # print(srEquationResiduals)
            intConstraintIndexToDelete = int(self.dfNLIPConstraintIndexMapping.loc[self.dfNLIPConstraintIndexMapping["NLIP_Constraint_Index"]==srEquationResiduals.idxmax(), "List_Constraint_Index"].iloc[0])
            # print("Dropping Constraint {}".format(intConstraintIndexToDelete))
            self.lstRemovedConstraints.append(self.lstConstraintCols[intConstraintIndexToDelete])
            lstNLIPConstraintsToBeRemoved = self.dfNLIPConstraintIndexMapping.loc[self.dfNLIPConstraintIndexMapping["List_Constraint_Index"]==intConstraintIndexToDelete, "NLIP_Constraint_Index"].to_list()
            if len(lstNLIPConstraintsToBeRemoved) == 1:
                del self.m._equations[lstNLIPConstraintsToBeRemoved[0]]
            elif len(lstNLIPConstraintsToBeRemoved) == 2:
                del self.m._equations[lstNLIPConstraintsToBeRemoved[0]:lstNLIPConstraintsToBeRemoved[1]+1]
            del self.dfSKUColConstraints[self.lstConstraintCols[intConstraintIndexToDelete]], self.lstConstraintCols[intConstraintIndexToDelete], self.lstSpaceConstraintPercentages[intConstraintIndexToDelete], self.lstConstraintTypeMappings[intConstraintIndexToDelete]
            self.dfNLIPConstraintIndexMapping = self.dfNLIPConstraintIndexMapping[self.dfNLIPConstraintIndexMapping["List_Constraint_Index"]!=intConstraintIndexToDelete].reset_index(drop=True)
            self.dfNLIPConstraintIndexMapping['NLIP_Constraint_Index'] = range(0, len(self.dfNLIPConstraintIndexMapping))
            for index, row in self.dfSKUColConstraints.iterrows():
                x[index].value = np.ceil(row["Initialization_Cols"] / row["Stacked_Units"])
            # print("If Block Ended")
            return self.__RecursiveSolver()
        else:
            # print("Else Block Started, Appstatus",self.m.options.APPSTATUS)
            return
        
    def __PreProcessData(self):
        # print(config.lstSKUsToRemove)
        self.dfDemandCurveData[config.CLUSTER_PARENT_SKU] = self.dfDemandCurveData['cluster_parent_sku'].apply(lambda x:x.split("_")[0])
        self.dfDemandCurveData = self.dfDemandCurveData[~self.dfDemandCurveData[config.CLUSTER_PARENT_SKU].isin(config.lstSKUsToRemove)].reset_index(drop=True)
        self.dfSimilarityData = self.dfSimilarityData[~self.dfSimilarityData["primary_sku"].isin(config.lstSKUsToRemove)].reset_index(drop=True)
        self.dfSimilarityData = self.dfSimilarityData[~self.dfSimilarityData["secondary_sku"].isin(config.lstSKUsToRemove)].reset_index(drop=True)
        self.dfSKUMaster = self.dfSKUMaster[~self.dfSKUMaster[config.CLUSTER_PARENT_SKU].isin(config.lstSKUsToRemove)].reset_index(drop=True)
        self.dfConstraintMaster = self.dfConstraintMaster[~self.dfConstraintMaster[config.CLUSTER_PARENT_SKU].isin(config.lstSKUsToRemove)].reset_index(drop=True)
            
        self.dfShelfMaster[config.CONSTRAINT_MAPPING] = self.dfShelfMaster[config.CONSTRAINT_MAPPING].fillna("Shelf_Unconstrained")
        self.dfProductDims = self.dfProductDims[[config.CLUSTER_PARENT_SKU, config.SKU_HEIGHT, config.SKU_WIDTH, config.SKU_DEPTH, config.DELISTING_NOT_ALLOWED_FLAG, config.MIN_COLS_IF_KEPT, config.MAX_COLS_IF_KEPT]]
        self.dfProductDims[config.DELISTING_NOT_ALLOWED_FLAG] = self.dfProductDims[config.DELISTING_NOT_ALLOWED_FLAG].fillna(0)
        self.dfProductDims[config.MIN_COLS_IF_KEPT] = self.dfProductDims[config.MIN_COLS_IF_KEPT].fillna(1)
        
        self.dfSKUMaster[config.CLUSTER_PARENT_SKU] = self.dfSKUMaster[config.CLUSTER_PARENT_SKU].astype(str)
        self.dfProductDims[config.CLUSTER_PARENT_SKU] = self.dfProductDims[config.CLUSTER_PARENT_SKU].astype(str)
        
        # print(self.dfSKUMaster.isnull().sum())
        
        self.dfSKUMaster = self.dfSKUMaster[[i for i in self.dfSKUMaster.columns if i not in ['Height', 'Width', 'Depth']]]

        # self.dfProductDims = self.dfProductDims[['sku','DOS','Delisting_Not_Allowed','Min_Cols_If_Kept']]
        self.dfSKUMaster = pd.merge(self.dfSKUMaster, self.dfProductDims, how="left", on=config.CLUSTER_PARENT_SKU)
        
        self.dfDemandCurveData["Columns"] = self.dfDemandCurveData["Columns"].astype(int)
        
        # =============================================================================
        # Checking if any constraint flag is 0 for the entire column i.e. sum of the flag is 0.
        # Removing such constrinats
        # =============================================================================

        # Creating a list to track unmet constraints
        self.lstRemovedConstraints = []

        self.dfConstraintMaster = self.dfConstraintMaster[[config.CLUSTER_PARENT_SKU]+[i for i in self.dfConstraintMaster.columns if i.startswith('Constraint_')]]

        for i in [i for i in self.dfConstraintMaster.columns if i.startswith('Constraint_')]:
            if self.dfConstraintMaster[i].sum()==0:
                self.lstRemovedConstraints.append(i)
                del self.dfConstraintMaster[i]
                self.dfShelfMaster[config.CONSTRAINT_MAPPING] = self.dfShelfMaster[config.CONSTRAINT_MAPPING].replace(i, "Shelf_Unconstrained")
                self.dfConstraintTypeMapping = self.dfConstraintTypeMapping[self.dfConstraintTypeMapping[config.CONSTRAINT_ID]!=i].reset_index(drop=True)
        
        self.dfShelfMaster_copy_1 = self.dfShelfMaster.copy()
        
        # =============================================================================
        # Shelf & SKU height adjustment
        # =============================================================================

        self.dfShelfMaster[config.SHELF_DOOR_HEIGHT] = self.dfShelfMaster[config.SHELF_DOOR_HEIGHT] - config.fltShelfDoorBufferHeight
        self.dfShelfMaster[config.SHELF_DOOR_WIDTH] = self.dfShelfMaster[config.SHELF_DOOR_WIDTH] - config.fltShelfDoorBufferWidth
        dfConstraintMinMaxHeights = self.dfShelfMaster.groupby([config.CONSTRAINT_MAPPING]).agg({config.SHELF_DOOR_HEIGHT:[min, max], config.SHELF_DOOR_DEPTH:min}).reset_index()

        dfConstraintMinMaxHeights.columns = [config.CONSTRAINT_MAPPING, "Min_Shelf_Height", "Max_Shelf_Height", "Min_Shelf_Depth"]
        
        self.dfShelfMaster = pd.merge(self.dfShelfMaster, dfConstraintMinMaxHeights, how="left", on=config.CONSTRAINT_MAPPING)
        self.dfShelfMaster["Augmented_Space"] = self.dfShelfMaster["Max_Shelf_Height"]*self.dfShelfMaster[config.SHELF_DOOR_WIDTH]*self.dfShelfMaster["Min_Shelf_Depth"]
        
        self.dfShelfMaster_copy_2 = self.dfShelfMaster.copy()
        
        self.dfConstraintMaster["Min_Shelf_Height"] = np.nan
        self.dfConstraintMaster["Max_Shelf_Height"] = np.nan
        self.dfConstraintMaster["Min_Shelf_Depth"] = np.nan
        
        
        self.dfConstraintMaster_copy_1 = self.dfConstraintMaster.copy()
        
        self.dfConstraintMaster["Shelf_Unconstrained"] = (self.dfConstraintMaster[self.dfConstraintTypeMapping.loc[self.dfConstraintTypeMapping[config.CONSTRAINT_TYPE]==config.CONSTRAINT_TYPE_SHELF, config.CONSTRAINT_ID]].max(axis=1).fillna(0) - 1)*-1
        
        self.dfConstraintTypeMapping_1 = self.dfConstraintTypeMapping.copy()
        self.dfConstraintMaster_copy_2 = self.dfConstraintMaster.copy()
        
        for index, row in dfConstraintMinMaxHeights.iterrows():
            print(row[config.CONSTRAINT_MAPPING])
            self.dfConstraintMaster["Min_Shelf_Height"] = np.where(self.dfConstraintMaster[row[config.CONSTRAINT_MAPPING]]==1, row["Min_Shelf_Height"], self.dfConstraintMaster["Min_Shelf_Height"])
            self.dfConstraintMaster["Max_Shelf_Height"] = np.where(self.dfConstraintMaster[row[config.CONSTRAINT_MAPPING]]==1, row["Max_Shelf_Height"], self.dfConstraintMaster["Max_Shelf_Height"])
            self.dfConstraintMaster["Min_Shelf_Depth"] = np.where(self.dfConstraintMaster[row[config.CONSTRAINT_MAPPING]]==1, row["Min_Shelf_Depth"], self.dfConstraintMaster["Min_Shelf_Depth"])
        
        self.dfConstraintMaster_copy_3 = self.dfConstraintMaster.copy()    
        
        self.dfConstraintMaster[config.CLUSTER_PARENT_SKU] = self.dfConstraintMaster[config.CLUSTER_PARENT_SKU].astype(str)

        self.dfSKUMaster = pd.merge(self.dfSKUMaster, self.dfConstraintMaster, how="left", on=config.CLUSTER_PARENT_SKU)
        
        self.dfSKUMaster["sku_height_too_large_flag"] = np.where(self.dfSKUMaster[config.SKU_HEIGHT]>self.dfSKUMaster["Max_Shelf_Height"], 1, 0)

        self.dfSKUMaster["possibility_of_stacking"] = np.where((self.dfSKUMaster[config.PACK_TYPE].isin(config.lstStackingPackType))&(self.dfSKUMaster[config.PACK_QTY]>=config.intStackingPackQty), 1, 0)

        self.dfSKUMaster["Real_Height"] = self.dfSKUMaster[config.SKU_HEIGHT]
        self.dfSKUMaster["Real_Depth"] = self.dfSKUMaster[config.SKU_DEPTH]

        if self.dfSKUMaster[config.MIN_COLS_IF_KEPT].equals(self.dfSKUMaster[config.MAX_COLS_IF_KEPT]):
            # Running a special case (No optimization only getting the cost dataframe)
            self.dfSKUMaster["Stacked_Units"] = 1
        else:
            # Calculate stacked units (Equivalent to vertical facings). This will be 1 or more.
            self.dfSKUMaster["Stacked_Units"] = np.where(self.dfSKUMaster["possibility_of_stacking"] == 1, np.floor(self.dfSKUMaster["Min_Shelf_Height"] /self.dfSKUMaster["Real_Height"]), 1)
            self.dfSKUMaster["Stacked_Units"] = np.where(self.dfSKUMaster["Stacked_Units"] == 0, 1, self.dfSKUMaster["Stacked_Units"])
        
        # Replacing all heights according to the logic to take care of wasted whitespace above the product
        self.dfSKUMaster[config.SKU_HEIGHT] = self.dfSKUMaster["Max_Shelf_Height"] / self.dfSKUMaster["Stacked_Units"]

        # Useful to calculate total width of kept SKUs asjusted for stacking. Used in setting half shelf constraint in NLIP.
        self.dfSKUMaster["Stacking_Adjusted_Width"] = self.dfSKUMaster[config.SKU_WIDTH] / self.dfSKUMaster["Stacked_Units"]

        # Calculate depth units. This will be 1 or more
        self.dfSKUMaster["Depth_Units"] = np.floor(self.dfSKUMaster["Min_Shelf_Depth"] /self.dfSKUMaster["Real_Depth"])

        # Replacing all depths according to the logic to take care of wasted whitespace behind the product
        self.dfSKUMaster[config.SKU_DEPTH] = self.dfSKUMaster["Min_Shelf_Depth"] / self.dfSKUMaster["Depth_Units"]

        # Calculating Min Cols 1 & 2 based on stacking & if the SKU is allowed to be delisted or not
        self.dfSKUMaster["Min_Cols_If_Kept_Stacking_Adjusted"] = np.ceil(self.dfSKUMaster[config.MIN_COLS_IF_KEPT] / self.dfSKUMaster["Stacked_Units"]) * self.dfSKUMaster["Stacked_Units"]
        self.dfSKUMaster["Min_Cols_1"] = np.where(self.dfSKUMaster[config.DELISTING_NOT_ALLOWED_FLAG]==1, self.dfSKUMaster["Min_Cols_If_Kept_Stacking_Adjusted"], 0)
        self.dfSKUMaster["Min_Cols_2"] = np.where(self.dfSKUMaster[config.DELISTING_NOT_ALLOWED_FLAG]==1, self.dfSKUMaster["Min_Cols_If_Kept_Stacking_Adjusted"] + self.dfSKUMaster["Stacked_Units"], self.dfSKUMaster["Min_Cols_If_Kept_Stacking_Adjusted"])
        
        # =============================================================================
        # Shelf Infeasibility Constraint Prep
        # =============================================================================

        self.dfSKUShelfInfeasibilityConstraint = pd.pivot_table(self.dfShelfMaster, values="Augmented_Space", index = [config.CONSTRAINT_MAPPING, config.SHELF_DOOR_HEIGHT], aggfunc=sum).reset_index().sort_values(by=[config.CONSTRAINT_MAPPING, config.SHELF_DOOR_HEIGHT], ascending=False)
        self.dfSKUShelfInfeasibilityConstraint["Augmented_Space_Cumsum_Less_Than"] = self.dfSKUShelfInfeasibilityConstraint.groupby(config.CONSTRAINT_MAPPING)["Augmented_Space"].transform("cumsum")
        self.dfSKUShelfInfeasibilityConstraint = self.dfSKUShelfInfeasibilityConstraint.sort_values(by=[config.CONSTRAINT_MAPPING, config.SHELF_DOOR_HEIGHT])
        self.dfSKUShelfInfeasibilityConstraint["Augmented_Space_Cumsum_Greater_Than"] = self.dfSKUShelfInfeasibilityConstraint.groupby(config.CONSTRAINT_MAPPING)["Augmented_Space"].transform("cumsum")
        self.dfSKUShelfInfeasibilityConstraint["Augmented_Space_Cumsum_Less_Than"] = self.dfSKUShelfInfeasibilityConstraint.groupby([config.CONSTRAINT_MAPPING])["Augmented_Space_Cumsum_Less_Than"].shift(-1)
        self.dfSKUShelfInfeasibilityConstraint = self.dfSKUShelfInfeasibilityConstraint.dropna(subset="Augmented_Space_Cumsum_Less_Than")
        self.dfSKUShelfInfeasibilityConstraint["ID"] = self.dfSKUShelfInfeasibilityConstraint.groupby([config.CONSTRAINT_MAPPING]).cumcount()+1
        self.dfSKUShelfInfeasibilityConstraint = self.dfSKUShelfInfeasibilityConstraint.drop(columns=["Augmented_Space"])

        # =============================================================================
        self.dfDemandCurveData["cluster_parent_sku"] = self.dfDemandCurveData["cluster_parent_sku"].astype(str)
        self.dfDemandCurveData[config.CLUSTER_PARENT_SKU] = self.dfDemandCurveData["cluster_parent_sku"].apply(lambda x: x.split("_")[0]).astype(str)
        self.dfDemandCurveData["Key"] = self.dfDemandCurveData[config.CLUSTER_PARENT_SKU].astype(str) + "_" + self.dfDemandCurveData["Columns"].astype(str)
        
        self.dfSimilarityData['primary_'+config.CLUSTER_PARENT_SKU] = self.dfSimilarityData['primary_'+config.CLUSTER_PARENT_SKU].astype(str)
        
        
        print("Before merge  ",self.dfDemandCurveData.shape)
        
        self.dfDemandCurveData = self.dfDemandCurveData.merge(self.dfSimilarityData[['primary_'+config.CLUSTER_PARENT_SKU,'primary_'+config.PRICE_PER_LTR]].drop_duplicates().rename(columns={'primary_'+config.CLUSTER_PARENT_SKU: config.CLUSTER_PARENT_SKU}), how='inner', on=config.CLUSTER_PARENT_SKU)
        
        print("After merge  ",self.dfDemandCurveData.shape)
        
        
        self.dfSimilarityData["Weighted_Secondary_PPL"] = self.dfSimilarityData['Demand_Transference_Volume'] * self.dfSimilarityData["secondary_"+config.PRICE_PER_LTR]

        # =============================================================================
        # Creating dfSKUColConstraints dataframe which has information about all the constraints that need to be set on the optimizer
        # =============================================================================

        # self.dfSKUColConstraints = self.dfDemandCurveData.groupby(config.CLUSTER_PARENT_SKU,as_index=False)["Columns", "Num_units_per_col"].max().rename(columns={"Columns":"Max_Cols"})
        
        
        
        self.dfSKUColConstraints = self.dfDemandCurveData.groupby(config.CLUSTER_PARENT_SKU,as_index=False)["Columns"].max().rename(columns={"Columns":"Max_Cols"})
        
        # print("1====>",self.dfSKUColConstraints.isnull().sum())

        dfRevMarketShares = self.dfSimilarityData[["primary_"+config.CLUSTER_PARENT_SKU, "primary_"+config.REVENUE]].drop_duplicates().reset_index(drop=True).rename(columns={"primary_"+config.CLUSTER_PARENT_SKU:config.CLUSTER_PARENT_SKU})
        dfRevMarketShares["primary_sku_rev_market_share"] = dfRevMarketShares["primary_"+config.REVENUE] / dfRevMarketShares["primary_"+config.REVENUE].sum()
        
        # print("1====>",self.dfSKUColConstraints.isnull().sum())

        self.dfSKUColConstraints = pd.merge(self.dfSKUColConstraints, dfRevMarketShares[[config.CLUSTER_PARENT_SKU, "primary_sku_rev_market_share"]], on=config.CLUSTER_PARENT_SKU, how="left")
        del dfRevMarketShares
        self.dfSKUColConstraints = self.dfSKUColConstraints.sort_values(by="primary_sku_rev_market_share", ascending=False)
        self.dfSKUColConstraints = pd.merge(self.dfSKUColConstraints, self.dfSKUMaster, on=config.CLUSTER_PARENT_SKU, how="left")
        self.dfSKUColConstraints["Max_Cols"] = self.dfSKUColConstraints[config.MAX_COLS_IF_KEPT].fillna(self.dfSKUColConstraints["Max_Cols"])
        
        # print("2====>",self.dfSKUColConstraints.isnull().sum())
        
        self.dfSKUColConstraints["Actual_Space_Per_Column"] = self.dfSKUColConstraints["Real_Height"] * self.dfSKUColConstraints[config.SKU_WIDTH] * self.dfSKUColConstraints["Real_Depth"] * self.dfSKUColConstraints["Depth_Units"]
        self.dfSKUColConstraints["Augmented_Space_Per_Column"] = self.dfSKUColConstraints[config.SKU_HEIGHT] * self.dfSKUColConstraints[config.SKU_WIDTH] * self.dfSKUColConstraints[config.SKU_DEPTH] * self.dfSKUColConstraints["Depth_Units"]

        # Initialization based on market share:
        if config.boolMaxColInitialization:
            self.dfSKUColConstraints["initialization_space"] = 0
            self.dfSKUColConstraints["Initialization_Cols"] = self.dfSKUColConstraints["Max_Cols"]
        else:
            self.dfSKUColConstraints["initialization_space"] = self.dfSKUColConstraints["primary_sku_rev_market_share"] * self.dfShelfMaster["Augmented_Space"].sum()
            self.dfSKUColConstraints["Initialization_Cols"] = np.ceil(self.dfSKUColConstraints["initialization_space"] / self.dfSKUColConstraints["Augmented_Space_Per_Column"])
        
        self.dfSKUColConstraints["Initialization_Cols"] = self.dfSKUColConstraints[["Initialization_Cols", "Min_Cols_1"]].max(axis=1)
        self.dfSKUColConstraints["Initialization_Cols"] = np.where(self.dfSKUColConstraints["sku_height_too_large_flag"]==1, 0, self.dfSKUColConstraints["Initialization_Cols"])
        # self.cat_exp_plano_output['sku'] = self.cat_exp_plano_output['sku'].astype(str)
        # self.dfSKUColConstraints = self.dfSKUColConstraints.merge(self.cat_exp_plano_output ,on='sku')
        # self.dfSKUColConstraints["Initialization_Cols"] = self.dfSKUColConstraints["cat_exp_cols"]
        
        # print("3====>",self.dfSKUColConstraints.isnull().sum())
        
        self.lstUniqueSKUs = self.dfSKUColConstraints[config.CLUSTER_PARENT_SKU].tolist()
        self.dfMaxColKeys = self.dfDemandCurveData.loc[self.dfDemandCurveData.groupby(config.CLUSTER_PARENT_SKU)["Columns"].transform(max) == self.dfDemandCurveData["Columns"]]
        

        
    def __RunOptimization(self):

        self.m = GEKKO(remote=False)

        global x
        x = []

        for index, row in self.dfSKUColConstraints.iterrows():
            if row["sku_height_too_large_flag"] == 1:
                # Note that, "Initialization_Cols" have been already been made zero if "sku_height_too_large_flag" == 1
                x.append(self.m.Var(value=np.ceil(row["Initialization_Cols"] / row["Stacked_Units"]), integer=True, lb=0, ub=0))
            elif row[config.MAX_COLS_IF_KEPT] == row[config.MAX_COLS_IF_KEPT]:
                x.append(self.m.Var(value=np.ceil(row["Initialization_Cols"] / row["Stacked_Units"]), integer=True, lb=np.ceil(row["Min_Cols_1"] / row["Stacked_Units"]), ub = np.ceil(row[config.MAX_COLS_IF_KEPT] / row["Stacked_Units"])))
            else:
                # x.append(self.m.Var(value=row["Initialization_Cols"], integer=True, lb=row["Min_Cols_1"]))
                x.append(self.m.Var(value=np.ceil(row["Initialization_Cols"] / row["Stacked_Units"]), integer=True, lb=np.ceil(row["Min_Cols_1"] / row["Stacked_Units"])))
        
        y = []
        
        for index, row in self.dfSKUColConstraints.iterrows():       
                y.append(self.m.Intermediate(x[index]*row["Stacked_Units"]))
        
        # =============================================================================
        # Adding constraints
        # =============================================================================

        # These 3 lists will have infomation about all the constraints added
        self.lstConstraintCols = []
        self.lstSpaceConstraintPercentages = []
        self.lstConstraintTypeMappings = []

        # Adding "Shelf" & "Overall" type constraints to the 3 lists
        for index, row in self.dfConstraintTypeMapping.iterrows():
            if row[config.CONSTRAINT_TYPE] == config.CONSTRAINT_TYPE_SHELF:
                self.lstConstraintCols.append(row[config.CONSTRAINT_ID])
                self.lstConstraintTypeMappings.append(row[config.CONSTRAINT_TYPE])
                dfTempFilterShelfMaster = self.dfShelfMaster[self.dfShelfMaster[config.CONSTRAINT_MAPPING]==row[config.CONSTRAINT_ID]]
                self.lstSpaceConstraintPercentages.append(dfTempFilterShelfMaster["Augmented_Space"].sum())
            if row[config.CONSTRAINT_TYPE] == config.CONSTRAINT_TYPE_OVERALL:
                self.lstConstraintCols.append(row[config.CONSTRAINT_ID])
                self.lstConstraintTypeMappings.append(row[config.CONSTRAINT_TYPE])
                self.lstSpaceConstraintPercentages.append(row[config.CONSTRAINT_SPACE_PERC])

        # Identifying if there is a need for sku-shelf infeasibility constraint
        self.dfSKUShelfInfeasibilityConstraint["Flag"] = 0
        for con in self.dfSKUShelfInfeasibilityConstraint[config.CONSTRAINT_MAPPING].unique():
            dfTempFilteredSKUShelfInfeasibilityConstraint = self.dfSKUShelfInfeasibilityConstraint[self.dfSKUShelfInfeasibilityConstraint[config.CONSTRAINT_MAPPING]==con].copy().reset_index(drop=True)
            intSumDefaultConstraintFlag = sum(self.dfSKUColConstraints[con])
            for index, row in dfTempFilteredSKUShelfInfeasibilityConstraint.iterrows():
                intSumToCompare = sum((self.dfSKUColConstraints[con]==1) & (self.dfSKUColConstraints["Real_Height"]>row[config.SHELF_DOOR_HEIGHT]))
                if intSumToCompare == 0:
                    continue
                if intSumToCompare < intSumDefaultConstraintFlag:
                    self.dfSKUShelfInfeasibilityConstraint.loc[(self.dfSKUShelfInfeasibilityConstraint[config.CONSTRAINT_MAPPING]==con)&(self.dfSKUShelfInfeasibilityConstraint["ID"]==row["ID"]), "Flag"] = 1
                    intSumDefaultConstraintFlag = intSumToCompare

        self.dfSKUShelfInfeasibilityConstraint = self.dfSKUShelfInfeasibilityConstraint[self.dfSKUShelfInfeasibilityConstraint["Flag"]==1].reset_index(drop=True)
        self.dfSKUShelfInfeasibilityConstraint["ID"] = self.dfSKUShelfInfeasibilityConstraint.groupby([config.CONSTRAINT_MAPPING]).cumcount()+1
        self.dfSKUShelfInfeasibilityConstraint["ID"] = self.dfSKUShelfInfeasibilityConstraint[config.CONSTRAINT_MAPPING] + "_" + self.dfSKUShelfInfeasibilityConstraint["ID"].astype(str)

        for index, row in self.dfSKUShelfInfeasibilityConstraint.iterrows():
            self.dfSKUColConstraints[row["ID"]+"_Less_Than"] = np.where((self.dfSKUColConstraints[row[config.CONSTRAINT_MAPPING]]==1) & (self.dfSKUColConstraints["Real_Height"]>row[config.SHELF_DOOR_HEIGHT]), 1, 0)
            self.lstConstraintCols.append(row["ID"]+"_Less_Than")
            self.lstSpaceConstraintPercentages.append(row["Augmented_Space_Cumsum_Less_Than"])
            self.lstConstraintTypeMappings.append("Shelf_Feasibility_Constraint_Less_Than")
            
            # Provision to add "Greater than type constraints"
            # self.dfSKUColConstraints[row["ID"]+"_Greater_Than"] = np.where((self.dfSKUColConstraints[row[config.CONSTRAINT_MAPPING]]==1) & (self.dfSKUColConstraints["Real_Height"]<=row["Height"]), 1, 0)
            # self.lstConstraintCols.append(row["ID"]+"_Greater_Than")
            # self.lstSpaceConstraintPercentages.append(row["Augmented_Space_Cumsum_Greater_Than"])
            # self.lstConstraintTypeMappings.append("Shelf_Feasibility_Constraint_Greater_Than")

        # If we are running a special case (No optimization only getting the cost dataframe), this entire constraint setting section will be skipped
        if not self.dfSKUColConstraints[config.MIN_COLS_IF_KEPT].equals(self.dfSKUColConstraints[config.MAX_COLS_IF_KEPT]):
            
            lstConstraintNamesForDF = []
            lstConstraintNLIPIndexForDF = []
            lstConstraintListIndexForDF = []
            lstConstraintTypeForDF = []
            intConstraintCounterForDF = 0

            for i in range(len(self.lstConstraintCols)):
                
                if self.lstConstraintTypeMappings[i] == config.CONSTRAINT_TYPE_SHELF:
                    # self.m.Equation(abs(sum(map(lambda a, b: a * b, y, list(self.dfSKUColConstraints["Augmented_Space_Per_Column"] * self.dfSKUColConstraints[self.lstConstraintCols[i]]))) - self.lstSpaceConstraintPercentages[i]) / self.lstSpaceConstraintPercentages[i] <= config.shelf_type_constraint_precision)
                    lstConstraintNamesForDF.extend([self.lstConstraintCols[i], self.lstConstraintCols[i]])
                    lstConstraintNLIPIndexForDF.extend([intConstraintCounterForDF, intConstraintCounterForDF+1])
                    intConstraintCounterForDF += 2
                    lstConstraintListIndexForDF.extend([i, i])
                    lstConstraintTypeForDF.extend([self.lstConstraintTypeMappings[i], self.lstConstraintTypeMappings[i]])
                    self.m.Equation(sum(map(lambda a, b: a * b, y, list(self.dfSKUColConstraints["Augmented_Space_Per_Column"] * self.dfSKUColConstraints[self.lstConstraintCols[i]]))) <= self.lstSpaceConstraintPercentages[i])
                    self.m.Equation(sum(map(lambda a, b: a * b, y, list(self.dfSKUColConstraints["Augmented_Space_Per_Column"] * self.dfSKUColConstraints[self.lstConstraintCols[i]]))) >= self.lstSpaceConstraintPercentages[i]*(1-config.shelf_type_constraint_precision))

                if self.lstConstraintTypeMappings[i] == "Shelf_Feasibility_Constraint_Less_Than":
                    lstConstraintNamesForDF.extend([self.lstConstraintCols[i]])
                    lstConstraintNLIPIndexForDF.extend([intConstraintCounterForDF])
                    intConstraintCounterForDF += 1
                    lstConstraintListIndexForDF.extend([i])
                    lstConstraintTypeForDF.extend([self.lstConstraintTypeMappings[i]])
                    self.m.Equation(sum(map(lambda a, b: a * b, y, list(self.dfSKUColConstraints["Augmented_Space_Per_Column"] * self.dfSKUColConstraints[self.lstConstraintCols[i]]))) <= self.lstSpaceConstraintPercentages[i])
                
                # Provision to add "Greater than type constraints"
                # if self.lstConstraintTypeMappings[i] == "Shelf_Feasibility_Constraint_Greater_Than":
                #     self.m.Equation(sum(map(lambda a, b: a * b, y, list(self.dfSKUColConstraints["Augmented_Space_Per_Column"] * self.dfSKUColConstraints[self.lstConstraintCols[i]]))) >= self.lstSpaceConstraintPercentages[i])
                
                if self.lstConstraintTypeMappings[i] == config.CONSTRAINT_TYPE_OVERALL:
                    lstConstraintNamesForDF.extend([self.lstConstraintCols[i]])
                    lstConstraintNLIPIndexForDF.extend([intConstraintCounterForDF])
                    intConstraintCounterForDF += 1
                    lstConstraintListIndexForDF.extend([i])
                    lstConstraintTypeForDF.extend([self.lstConstraintTypeMappings[i]])
                    self.m.Equation(abs((sum(map(lambda a, b: a * b, y, list(self.dfSKUColConstraints["Actual_Space_Per_Column"] * self.dfSKUColConstraints[self.lstConstraintCols[i]]))) / sum(map(lambda a, b: a * b, y, self.dfSKUColConstraints["Actual_Space_Per_Column"].to_list()))) - self.lstSpaceConstraintPercentages[i]) <= config.overall_type_constraint_precision)
            
            self.dfNLIPConstraintIndexMapping = pd.DataFrame({"Shelf_Constraint_Name":lstConstraintNamesForDF, "NLIP_Constraint_Index":lstConstraintNLIPIndexForDF, "List_Constraint_Index":lstConstraintListIndexForDF, "Constraint_Type":lstConstraintTypeForDF})
            # =============================================================================
            # Adding half shelf constraint if boolHalfShelfConstraint == True
            # =============================================================================
    
            # if config.boolHalfShelfConstraint == True:
    
            #     lstIntermediateWidths = []
                
            #     for index, row in self.dfSKUColConstraints.iterrows():
            #         lstIntermediateWidths.append(self.m.Intermediate(row["Stacking_Adjusted_Width"]*y[index]))
                
            #     mx = lstIntermediateWidths[0] # max
            #     for i in range(1,len(lstIntermediateWidths)):
            #         mx = self.m.max3(mx,lstIntermediateWidths[i])
            #     fltTotalPlanoWidth = self.dfShelfMaster.groupby(['Shelf'])['Width'].sum().max()
            #     self.m.Equation(mx<=fltTotalPlanoWidth/2)
            #     self.lstConstraintCols.append("Half_Shelf_Constraint")
            #     self.lstSpaceConstraintPercentages.append(fltTotalPlanoWidth/2)
            #     self.lstConstraintTypeMappings.append("Half_Shelf_Constraint")

            # =============================================================================
    
            # Adding overall space constraint (Note that this is always the last equation / constraint added)
            # self.m.Equation(abs(sum(map(lambda a, b: a * b, y, self.dfSKUColConstraints["Augmented_Space_Per_Column"].to_list())) - self.dfShelfMaster["Augmented_Space"].sum()) / self.dfShelfMaster["Augmented_Space"].sum() <= config.total_planogram_space_constraint_precision)
            self.m.Equation(sum(map(lambda a, b: a * b, y, self.dfSKUColConstraints["Augmented_Space_Per_Column"].to_list())) <= self.dfShelfMaster["Augmented_Space"].sum())
            self.m.Equation(sum(map(lambda a, b: a * b, y, self.dfSKUColConstraints["Augmented_Space_Per_Column"].to_list())) >= self.dfShelfMaster["Augmented_Space"].sum() * (1-config.total_planogram_space_constraint_precision))

        # =============================================================================
        # Setting constraints finished
        # =============================================================================

        self.m.Minimize(self.__ObjectiveFunction(y))

        # We need to use SOLVER = 1 which is APOPT. This is the solver used for Mixed integer programming.
        # 1: APOPT, 2: BPOPT, 3: IPOPT
        self.m.options.SOLVER = 1
        # self.m.options.COLDSTART = 2
        # self.m.open_folder()
        
        try:
            self.__RecursiveSolver()
            self.dfSKUColConstraints["Solution"] = [int(i.value[0]) for i in y]
        except:
            self.dfSKUColConstraints["Solution"] = 0

        dfDemandCurveSolutionExtention = self.dfSKUColConstraints.loc[self.dfSKUColConstraints["Solution"] > self.dfSKUColConstraints["Max_Cols"], [config.CLUSTER_PARENT_SKU, "Solution"]].rename(columns={"Solution":"Columns"})
        dfDemandCurveSolutionExtention = pd.merge(dfDemandCurveSolutionExtention, self.dfMaxColKeys.drop(columns=["Columns"]), on=config.CLUSTER_PARENT_SKU, how="left")
        dfDemandCurveSolutionExtention["Key"] = dfDemandCurveSolutionExtention[config.CLUSTER_PARENT_SKU].astype(str) + "_" + dfDemandCurveSolutionExtention["Columns"].astype(str)
        self.dfDemandCurveData = pd.concat([self.dfDemandCurveData[self.dfDemandCurveData["Columns"]<=self.dfDemandCurveData["max_columns"]].reset_index(drop=True), dfDemandCurveSolutionExtention], ignore_index=True)
        self.dfSKUColConstraints = pd.merge(self.dfSKUColConstraints, self.dfDemandCurveData[[config.CLUSTER_PARENT_SKU, 'Meeting_Demand_units_perc', 'percentage_times_met', 'Columns', 'Total_Volume', 'Volume_covered', 'primary_'+config.PRICE_PER_LTR]].rename(columns={"Columns":"Solution"}), on=[config.CLUSTER_PARENT_SKU, 'Solution'], how='left')

        self.dfSKUColConstraints["Max_Cols"] = np.ceil(self.dfSKUColConstraints[["Max_Cols", "Solution", "Min_Cols_If_Kept_Stacking_Adjusted"]].max(axis=1) / self.dfSKUColConstraints["Stacked_Units"]) * self.dfSKUColConstraints["Stacked_Units"]
        self.dfSKUColConstraints["Solution Actual Space Occupied"] = self.dfSKUColConstraints["Solution"]*self.dfSKUColConstraints["Actual_Space_Per_Column"]
        self.dfSKUColConstraints["Solution Augmented Space Occupied"] = self.dfSKUColConstraints["Solution"]*self.dfSKUColConstraints["Augmented_Space_Per_Column"]
        self.fltPlanoTotalAugmentedSpace = self.dfShelfMaster["Augmented_Space"].sum()


    def Run(self):
        
        self.__PreProcessData()
        self.__RunOptimization()