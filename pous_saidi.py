"""
- processes and corrects raw POUS data
- aggregates outages and customerstracked to county level and corrects county-level customers tracked
- Output: time-series of county level CustomersOut and CustomersTracked

srun -n 1 -c 2 --partition=cpu-preempt --mem=60G  --pty bash

@author: Zeal Shah
@date: December 17, 2022
"""
import os
import sys
import glob
import code
import json
import scipy
import difflib
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from statsmodels import robust
from shapely.ops import unary_union
from scipy.stats.mstats import winsorize


def POUS_county_daily_outages(input_dir):
	"""
	- performs basic processing of the CountyByUtility daily summary dataset
	- winsorizes customers tracked
	- aggregates customers out and tracked to county level
	- output: daily customers out and tracked in each county
	"""
	raw_filename = "POUS_Export_CountyByUtility_Daily_2017-2021.csv"
	dm = pd.read_csv(os.path.join(input_dir, raw_filename), encoding='utf-16')
	dc = dm.copy() #create a backup
	dc.loc[:,"RecordDate"] = pd.to_datetime(dc.loc[:,"RecordDate"], format="%Y-%m-%d")
	dc.loc[:,"year"] = dc.loc[:,"RecordDate"].dt.year
	#
	dc.loc[:,"StateName"] = dc["StateName"].str.replace(r'[^A-Za-z]+', '', regex=True).str.lower()
	dc.loc[:,"UtilityName"] = dc["UtilityName"].str.replace(r'[^A-Za-z]+', '', regex=True).str.lower()
	dc.loc[:,"CountyName"] = dc["CountyName"].str.replace(r'[^A-Za-z]+', '', regex=True).str.lower()
	"""
	calculate customers tracked
	- winsorize over a year
	- output: it only changes daily CT values when they are beyond the winsorization extremes
	"""
	# noise removal
	def winsorize_series(group):
	    return winsorize(group, limits=[0.1,0.1]) #initially it was [0.1,0.2]
	# winsorize year by year
	dc.loc[:,"CustomersTracked_clean"] = dc.groupby(["StateName","CountyName","UtilityName","year"])["CustomersTracked"].transform(winsorize_series)
	"""
	calculate total daily customer-outage-hours and customers tracked in each county 
	- Basically adds up all customer outage-hours and customers tracked in a county and eliminates utility name
	"""
	dc = dc.groupby(["StateName","CountyName","RecordDate","year"])[["CustomersTracked_clean","CustomerHoursOutTotal"]].sum().reset_index()
	dc = dc[(dc["CustomerHoursOutTotal"] <= dc["CustomersTracked"]*24) & (dc["CustomerHoursOutTotal"] <= dc["CustomersTracked_clean"]*24)]
	# dc["SAIDI"] = dc["CustomerHoursOutTotal"].div(dc["CustomersTracked_clean"])
	return dc


#### Functions used for Resource Economics Work [Old and deprecated on June 17, 2023]
def get_stormevents_db_old(stormdata_dir, counties_shapefile):
	""" 
	processes and combines storm events files into a single dataframe. 
	columns for future work: ["DAMAE_PROPERTY","INJURIES_DIRECT","INJURIES_INDIRECT","EPISODE_NARRATIVE","BEGIN_LAT","BEGIN_LON","END_LAT","END_LON"]
	"""
	print("--*--"*30, flush=True)
	print("Loading and processing storms data.", flush=True)
	#
	cols_of_interest = ["BEGIN_DATE_TIME","END_DATE_TIME","CZ_TIMEZONE","EPISODE_ID","EVENT_ID","STATE_FIPS","CZ_FIPS","STATE","CZ_NAME","CZ_TYPE","EVENT_TYPE"]
	"""
	STEP 0: use county shapefile and NWPS Zones file to map zone numbers to countyFIPS
	Reason: some entries in the storm dataset have zone numbers whereas some entries have countyFIPS. For consistency, we convert everything to countyFIPS.
	"""
	crs = "EPSG:4326"
	# process zones dataframe (updated z_22mr22.shp to z_08mr23.shp)
	zones = gpd.read_file(stormdata_dir + "/NWPS_zones/z_08mr23.shp")[["STATE_ZONE","geometry"]]
	zones = zones.to_crs(crs)
	# some state_zones have multiple entries i.e. different geometries. They can be merged into one single geom
	zones = zones.groupby(["STATE_ZONE"])["geometry"].apply(lambda x: np.array(x)).reset_index()
	zones.loc[:,"geometry"] = zones["geometry"].apply(lambda x: unary_union(list(x)))
	# obtain centroids of each zone
	zones.loc[:,"geometry"] = zones["geometry"].apply(lambda x: x.centroid)
	zones = gpd.GeoDataFrame(zones, geometry="geometry", crs="EPSG:4326")
	# sjoin with counties data
	counties = gpd.read_file(counties_shapefile)[["STATEFP","COUNTYFP","geometry"]]
	counties.crs = "EPSG:4326"
	counties.loc[:,"CountyFIPS"] = counties.loc[:,"STATEFP"] + counties.loc[:,"COUNTYFP"]
	z2c = (
	    gpd.sjoin(zones, counties, op="within", how="left")
	    .drop(columns=["index_right","geometry"])
	    .drop_duplicates()
	    .dropna()
	    .reset_index(drop=True)
	)
	#
	# reformat STATEZONE column from: (statename + zone number) format ---> (statefips + zone number) format
	z2c["ZoneFIPS"] = z2c.apply(lambda x: int(str(x["STATEFP"]) + x["STATE_ZONE"][2:]), axis=1)
	z2c["State"] = z2c["STATE_ZONE"].apply(lambda x: x[0:2])
	# NV gets 04 (AZ) STATEFP instead of 32; MO Missouri gets 05 (AR) instead of 29 in some cases. Centroids of these confused souls lie in an out-of-state county. We ignore these points for simplicity.
	condition1 = ((z2c["State"]=="NV") & (z2c["STATEFP"]=="04"))
	condition2 = ((z2c["State"]=="MO") & (z2c["STATEFP"]=="05"))
	z2c = z2c[(~condition1) & (~condition2)].reset_index(drop=True)
	#
	"""
	STEP 1: merge all the stormevents csvs [cols_of_interest]
	"""
	all_files = glob.glob(stormdata_dir + "/*.csv")
	ds = []
	for file in all_files:
	    temp = pd.read_csv(file)[cols_of_interest]
	    ds.append(temp)
	    del(temp)
	ds = pd.concat(ds)
	ds.loc[:,"STATE"] = ds["STATE"].str.replace(r'\W', '', regex=True).str.lower()
	#
	"""
	STEP 2: assign countyFIPS to all entries in CZ_TYPE == "z" in the storms dataframe.
	NOTE: we use both, CZ_TYPE == "c" and "z" in this work, where "c" represents county and "z" represents zone.
	"""
	# create a CountyFIPS column in storms data using CZ_FIPS and STATE_FIPS
	ds[["CZ_FIPS","STATE_FIPS"]] = ds[["CZ_FIPS","STATE_FIPS"]].astype(str)
	ds.loc[:,"CZ_FIPS"] = ds["CZ_FIPS"].apply(lambda x: x.zfill(3))
	ds.loc[:,"CZ_FIPS"] = (ds["STATE_FIPS"] + ds["CZ_FIPS"]).astype(int)
	ds = pd.merge(ds, z2c[["CountyFIPS","ZoneFIPS"]], left_on="CZ_FIPS", right_on="ZoneFIPS", how="left")
	ds["CountyFIPS"] = ds.apply(lambda x: x["CZ_FIPS"] if x["CZ_TYPE"]=="C" else x["CountyFIPS"] if x["CZ_TYPE"]=="Z" else np.nan, axis=1)
	# we lose 22k datapoints out of 300K. This data loss is related to zones without mapping to countyFIPS not present in the shapefile. Two possible causes: (1) county wasn't included in orig county shpfile. (2) zone centroid did not fall in any of the counties. remaining len(ds) ~ 287K
	ds = (
		ds
	    .dropna(subset=["CountyFIPS"])
	    .drop(columns=["ZoneFIPS","CZ_TYPE","CZ_FIPS","CZ_NAME"])
	)
	#
	"""
	STEP 3: 
	- convert CZ_TIMEZONE to native Python format. 
	- convert local ts to UTC (POUS is in UTC).
	"""
	timezones = {"EST-5":"US/Eastern","EDT-4":"US/Eastern","CST-6":"US/Central","CDT-5":"US/Central","MST-7":"US/Mountain","PST-8":"US/Pacific","PDT-7":"US/Pacific","AKST-9":"US/Alaska","HST-10":"US/Hawaii"}
	ds.loc[:,"tz"] = ds["CZ_TIMEZONE"].apply(lambda x: timezones[x] if x in timezones.keys() else np.nan)
	# we drop nans and remove virgin islands
	ds = ds.dropna()
	# #
	# """
	# STEP 4: process string timestamps and convert tz to UTC
	# """
	ds.loc[:,["BEGIN_DATE_TIME","END_DATE_TIME"]] = ds.loc[:,["BEGIN_DATE_TIME","END_DATE_TIME"]].apply(pd.to_datetime, format='%d-%b-%y %H:%M:%S')
	ds = ds.sort_values(by=["CountyFIPS","BEGIN_DATE_TIME"])
	# iterating using groupby because some countyFIPS have two tz. County-wise iteration helps with nonexistent and ambigous timestamps due to DST
	ds_utc = []
	for idx, grouped in ds.groupby(["CountyFIPS","tz"]):
	    grouped.loc[:,"BEGIN_DATE_TIME"] = grouped.apply(lambda row: row["BEGIN_DATE_TIME"].tz_localize(row['tz'], nonexistent='shift_forward').tz_convert('UTC').tz_localize(None), axis=1)
	    grouped.loc[:,"END_DATE_TIME"] = grouped.apply(lambda row: row["END_DATE_TIME"].tz_localize(row['tz'], nonexistent='shift_forward', ambiguous='NaT').tz_convert('UTC').tz_localize(None), axis=1)
	    ds_utc.append(grouped)
	ds_utc = pd.concat(ds_utc).dropna().reset_index(drop=True) # we lose only 1 ambiguous reading here
	ds_utc = (
	    ds_utc[["EVENT_ID","EVENT_TYPE","BEGIN_DATE_TIME","END_DATE_TIME","CountyFIPS"]]
	    .rename(columns={"BEGIN_DATE_TIME":"event_start", "END_DATE_TIME":"event_end", "EVENT_ID":"event_id","EVENT_TYPE":"event_type"}) 
	)
	"""
	For each (event_id, event_type, CountyFIPS) tuple, create one row per day of the event. 
	"""
	db = []
	for _, row in ds_utc.iterrows():
		event_start = row["event_start"].date()
		event_end = row["event_end"].date()
		#
		date_range = pd.date_range(event_start, event_end, freq="1D")
		#
		temp = pd.DataFrame({
			"event_id":row["event_id"],
			"event_type":row["event_type"],
			"CountyFIPS":row["CountyFIPS"],
			"event_date":date_range,
		})
		db.append(temp)
	#
	db = pd.concat(db).reset_index(drop=True)
	#
    # # remove duplicates and remove droughts
    # ds_utc = ds_utc.drop(columns="event_id").drop_duplicates() # event ids with same times
    # if droughts == False:
    #     ds_utc = ds_utc[(ds_utc["event_type"]!="Drought")].reset_index(drop=True)
    # #
    # """
    # STEP 5: Merge overlapping storms
    # (1) Merging level 1: combine time intervals that overlap and use one big interval instead
    # Reference: https://stackoverflow.com/questions/46784482/merging-time-ranges-in-python
    
    # (2) Merging level 2: merge smaller non-overlapping intervals. If consecutive storms lie within 2 hours of one another, just label them as one.
    # """
    # # Merging level 1
    # def merge_times(times):
    #     times = iter(times)
    #     merged = next(times).copy()
    #     for entry in times:
    #         start, end = entry['event_start'], entry['event_end']
    #         if start <= merged['event_end']:
    #             # overlapping, merge
    #             merged['event_end'] = max(merged['event_end'], end)
    #         else:
    #             # distinct; yield merged and start a new copy
    #             yield merged
    #             merged = entry.copy()
    #     yield merged
    # #
    # ds_utc_grouped = ds_utc.groupby(["CountyFIPS"])
    # ds_utc_arr = []
    # for CountyFIPS, group in ds_utc_grouped:
    #     # merge level 1 (takes care of overlapping intervals)
    #     storm_times = [{"event_start":i[0], "event_end":i[1]} for i in zip(group["event_start"], group["event_end"])]
    #     storm_times = sorted(storm_times, key=itemgetter('event_start', 'event_end'))
    #     storm_times_merged = list(merge_times(storm_times))
    #     storm_times_merged = pd.DataFrame(storm_times_merged)
    #     # merge level 2 (takes care of non-overlapping but nearby intervals)
    #     storm_times_merged.loc[:,"prev_event_end"] = storm_times_merged.loc[:,"event_end"].shift(1)
    #     storm_times_merged.loc[:,"duration_since_last_event"] = storm_times_merged.loc[:,"event_start"] - storm_times_merged.loc[:,"prev_event_end"]
    #     storm_times_merged.loc[:,"outage_group"] = storm_times_merged.loc[:,"duration_since_last_event"].apply(lambda x: True if x <=pd.Timedelta(hours=2) else False)
    #     storm_times_merged.loc[:,"outage_group"] = (storm_times_merged.loc[:,"outage_group"] == False).cumsum()
    #     storm_times_merged = storm_times_merged.groupby(["outage_group"]).agg({"event_start":"min","event_end":"max"}).reset_index()
    #     storm_times_merged.columns = ["outage_group","event_start","event_end"]
    #     storm_times_merged["CountyFIPS"] = CountyFIPS
    #     ds_utc_arr.append(storm_times_merged)
    # #
    # ds_utc = pd.concat(ds_utc_arr).drop(columns=["outage_group"]).reset_index(drop=True)
    # del(ds_utc_grouped)
    # del(ds_utc_arr)
    # """
    # STEP 6: generate unique storm ids
    # """
    # ds_utc.loc[:,"storm_id"] = ds_utc.index.map(lambda x: "storm_{}".format(x))
    # ds_utc.loc[:,"duration"] = ds_utc.loc[:,"event_end"] - ds_utc.loc[:,"event_start"]
    # print("--*--"*30, flush=True)
    # print("Storm dataset is ready.", flush=True)
    # return ds_utc
	return db


def process_EIA_old(input_dir):
	#
	states = []
	usa = []
	for year in np.arange(2016,2022):
		df = pd.read_excel(os.path.join(input_dir, "Reliability_{}.xlsx".format(year)))
		df = df.iloc[:,0:16]
		df = df.drop(df.index[-1])
		# dataset's format got updated in 2021
		if year != 2021:
			df.columns = df.iloc[0]
			df = df.drop(df.index[0])
			df = df[["Data Year", "Utility Number", "Utility Name", "State", "SAIDI With MED", "SAIDI Without MED", "Number of Customers"]]
			df = df.rename(columns={"Data Year":"year", "Utility Number":"utility_number","Utility Name":"utility_name","State":"state","SAIDI With MED":"saidi_w_storm","SAIDI Without MED":"saidi_wo_storm","Number of Customers":"customers"})
		else:
			df.columns = df.iloc[1]
			df = df.drop(df.index[[0,1]])
			df.columns = ["year", "utility_number", "utility_name", "state", "_", "saidi_w_storm", "_", "_", "saidi_wo_storm", "_", "_", "_", "_", "_", "customers", "_"]
			df = df.drop(columns=["_"])
		#
		df["saidi_w_storm"] = df["saidi_w_storm"].apply(lambda x: x if (isinstance(x, float)==True) | (isinstance(x, int)==True) else np.nan)
		df["saidi_wo_storm"] = df["saidi_wo_storm"].apply(lambda x: x if (isinstance(x, float)==True) | (isinstance(x, int)==True) else np.nan)
		df["customers"] = df["customers"].apply(lambda x: x if (isinstance(x, float)==True) | (isinstance(x, int)==True) else np.nan)
		# compute outage duration 
		df["outage_hours_w_storm"] = df["saidi_w_storm"].mul(df["customers"])/60
		df["outage_hours_wo_storm"] = df["saidi_wo_storm"].mul(df["customers"])/60
		df["outage_hours_only_storm"] = df["outage_hours_w_storm"] - df["outage_hours_wo_storm"]
		"""
		state-level
		"""
		# total outage duration and total customers in state
		df = df.groupby(["state","year"])[["outage_hours_w_storm","outage_hours_wo_storm","outage_hours_only_storm","customers"]].sum().reset_index()
		# calculate saidi - storm and no storm
		df.loc[:,"saidi"] = df["outage_hours_w_storm"].div(df["customers"]) # overall SAIDI
		df.loc[:,"saidi_wo_storm"] = df["outage_hours_wo_storm"].div(df["customers"]) # SAIDI without major events
		df.loc[:,"saidi_w_storm"] = df["outage_hours_only_storm"].div(df["customers"]) # SAIDI with major events
		#
		states.append(df)
		"""
		USA-level
		"""
		# total outage duration and total customers in state
		c = df.groupby(["year"])[["outage_hours_w_storm","outage_hours_wo_storm","outage_hours_only_storm","customers"]].sum().reset_index()
		# calculate saidi - storm and no storm
		c.loc[:,"saidi"] = c["outage_hours_w_storm"].div(c["customers"]) # overall SAIDI
		c.loc[:,"saidi_wo_storm"] = c["outage_hours_wo_storm"].div(c["customers"]) # SAIDI without major events
		c.loc[:,"saidi_w_storm"] = c["outage_hours_only_storm"].div(c["customers"]) # SAIDI with major events
		#
		usa.append(c)
	#
	states = pd.concat(states).reset_index(drop=True)
	usa = pd.concat(usa).reset_index(drop=True)
	return states, usa


#### Functions used for Resource Economics Work [Updated function created on June 17, 2023]
#### It saves processed storm data pickle file
def get_stormevents_db(stormdata_dir, counties_shapefile, use_pickled=False):
	#
	output_dir = "/gypsum/eguide/projects/zshah/data/USA_storms_data/StormEvents_processed"
	output_filename = "StormEvents_c2022.pkl" #2022 version
	#
	if use_pickled == True:
		db = pd.read_pickle(os.path.join(output_dir, output_filename))
		return db
	""" 
	processes and combines storm events files into a single dataframe. 
	columns for future work: ["DAMAE_PROPERTY","INJURIES_DIRECT","INJURIES_INDIRECT","EPISODE_NARRATIVE","BEGIN_LAT","BEGIN_LON","END_LAT","END_LON"]
	"""
	print("--*--"*30, flush=True)
	print("Loading and processing storms data.", flush=True)

	"""
	STEP 0: Load extra data
	"""
	# load mapping of county names and FIPS to states
	county2fips = pd.read_csv("/gypsum/eguide/projects/zshah/data/USA_fips2county/fips2county.tsv", delimiter="\t")
	county2fips = county2fips[["CountyFIPS","CountyName","StateName"]].drop_duplicates()
	county2fips["StateName"] = county2fips["StateName"].str.replace(r'[^A-Za-z]+', '', regex=True).str.lower()
	county2fips["CountyName"] = county2fips["CountyName"].str.replace(r'[^A-Za-z]+', '', regex=True).str.lower()
	county2fips["CountyFIPS"] = county2fips["CountyFIPS"].apply(lambda x: str(x).zfill(5))

	# dictionary of state to list of county names
	state2county = county2fips.groupby(["StateName"])
	state2county = {key: group["CountyName"].tolist() for key, group in state2county}

	"""
	STEP 1: use county shapefile and NWPS Zones file to map zone numbers to countyFIPS
	Reason: some entries in the storm dataset have zone numbers whereas some entries have countyFIPS. For consistency, we convert everything to countyFIPS.
	"""
	# process zones dataframe (updated z_22mr22.shp to z_08mr23.shp)
	crs = "EPSG:4326" # NOTE: works only in usa_env conda environment. Newer versions of geopandas might raise an error
	zones = gpd.read_file(stormdata_dir + "/NWPS_zones/z_08mr23.shp")[["STATE","STATE_ZONE","geometry"]]
	zones = zones.to_crs(crs)

	# sjoin with counties data (one zone can intersect with multiple counties)
	counties = gpd.read_file(counties_shapefile)[["STATEFP","COUNTYFP","geometry"]]
	counties = counties.to_crs(crs)
	counties.loc[:,"CountyFIPS"] = counties.loc[:,"STATEFP"] + counties.loc[:,"COUNTYFP"]
	counties["county_geom"] = counties["geometry"]

	# Perform spatial join using 'intersects'
	z2c = gpd.sjoin(zones, counties, op="intersects", how="inner")

	# Calculate the area of the intersection between each zone and county
	z2c["intersection_area"] = z2c.geometry.intersection(z2c["county_geom"]).area

	# Filter out matches with minimal overlap/intersection (threshold was manually selected looking at Los Angeles county and the corresponding zones)
	z2c = z2c[z2c.intersection_area >= 1e-3]

	# Add zones that lie within a county but might be eliminated due to small self-area
	z2c_within = gpd.sjoin(zones, counties, op="within", how="inner")

	# combine the two dataframes
	z2c = pd.concat([z2c, z2c_within]).reset_index(drop=True)

	# find direct mapping of state shortname to state fips.
	# since some zones intersect multiple states, stateFP and state_zone mapping needs to be cleaned first
	state2fips = z2c[["STATE","STATEFP"]].drop_duplicates()
	state2fips = state2fips.groupby("STATE")["STATEFP"].apply(lambda x: x.mode()[0])

	# merge this with the z2c dataframe
	z2c = z2c[["STATE","STATE_ZONE","CountyFIPS"]].drop_duplicates()
	z2c = pd.merge(z2c, state2fips, on=["STATE"], how="left")

	# reformat STATEZONE column from: (statename + zone number) format ---> (statefips + zone number) format
	# this is to ensure compatibility with the NOAA storms database
	z2c["ZoneFIPS"] = z2c.apply(lambda x: x["STATEFP"] + x["STATE_ZONE"][2:], axis=1)
	z2c["CZ_TYPE"] = "Z"

	"""
	STEP 2: Load storm data and add the mapping of zone to county
	"""
	cols_of_interest = ["BEGIN_DATE_TIME","END_DATE_TIME","CZ_TIMEZONE","EPISODE_ID","EVENT_ID","STATE_FIPS","CZ_FIPS","STATE","CZ_NAME","CZ_TYPE","EVENT_TYPE","BEGIN_LAT","BEGIN_LON","END_LAT","END_LON"]
	all_files = glob.glob(stormdata_dir + "/*_c2022*.csv") #use the most updated files (they were updated in 2023). 
	ds = []
	for file in all_files:
	    temp = pd.read_csv(file)[cols_of_interest]
	    ds.append(temp)
	    print("File = {} | rows = {}".format(file, len(temp)))
	    # display(temp.head())
	    del(temp)
	ds = pd.concat(ds)
	ds.loc[:,"STATE"] = ds["STATE"].str.replace(r'\W', '', regex=True).str.lower()

	print("total episodes = {} | total events = {}".format(ds["EPISODE_ID"].nunique(), ds["EVENT_ID"].nunique()))

	ds.loc[:,"CZ_FIPS"] = ds.apply(lambda x: str(x["STATE_FIPS"]).zfill(2) + str(x["CZ_FIPS"]).zfill(3), axis=1)

	# perform the join only for "Zones"
	ds = pd.merge(ds, z2c[["CountyFIPS","ZoneFIPS","CZ_TYPE"]], left_on=["CZ_TYPE","CZ_FIPS"], right_on=["CZ_TYPE","ZoneFIPS"], how="left")

	# wherever the dataset states "C", the CZ_FIPS column gives us county FIPS --> leave that as is
	# wherever the dataset states "Z", the CZ_FIPS gives the zone FIPS --> use mapping of zone FIPS to CountyFIPS
	ds["CountyFIPS"] = ds.apply(lambda x: x["CZ_FIPS"] if x["CZ_TYPE"]=="C" else x["CountyFIPS"] if x["CZ_TYPE"]=="Z" else np.nan, axis=1)

	# check if any of the "zones" have CZ_FIPS same as county FIPS
	countyfips = counties["CountyFIPS"].unique()
	ds["CountyFIPS"] = ds.apply(lambda x: x["CZ_FIPS"] if ((x["CZ_TYPE"]=="Z") & (pd.isnull(x["CountyFIPS"]) & (x["CZ_FIPS"] in countyfips))) else x["CountyFIPS"], axis=1)

	# many of the rows with missing county FIPS correspond to marine zones, which are not covered in the NWPS zones
	# since marine zones are limited to the beach/coastal areas, we would hardly find overlap between a county and marine zone
	# so instead we will use BEGIN_LAT/LON and END_LAT/LON
	ds_missing_CountyFIPS = ds[(pd.isnull(ds["CountyFIPS"]))].copy().drop(columns=["CountyFIPS"])

	ds = (
	    ds
	    .dropna(subset=["CountyFIPS"])
	    .drop(columns=["ZoneFIPS","CZ_TYPE","CZ_FIPS","CZ_NAME","STATE_FIPS","STATE","BEGIN_LAT","BEGIN_LON","END_LAT","END_LON"])
	)

	print("total episodes = {} | total events = {}".format(ds["EPISODE_ID"].nunique(), ds["EVENT_ID"].nunique()))

	"""
	STEP 3: Deal with rows that are missing CountyFIPS (ds_missing_CountyFIPS)
	"""
	ds_missing_CountyFIPS["begin_geometry"] = ds_missing_CountyFIPS.apply(lambda x: Point(x["BEGIN_LON"], x["BEGIN_LAT"]), axis=1)
	ds_missing_CountyFIPS["end_geometry"] = ds_missing_CountyFIPS.apply(lambda x: Point(x["END_LON"], x["END_LAT"]), axis=1)
	ds_missing_CountyFIPS = ds_missing_CountyFIPS.drop(columns = ["BEGIN_LAT","BEGIN_LON","END_LAT","END_LON"])

	ds_missing_CountyFIPS_begin = ds_missing_CountyFIPS.drop(columns=["end_geometry"]).rename(columns={"begin_geometry":"geometry"})
	ds_missing_CountyFIPS_end = ds_missing_CountyFIPS.drop(columns=["begin_geometry"]).rename(columns={"end_geometry":"geometry"})

	ds_missing_CountyFIPS = pd.concat([ds_missing_CountyFIPS_begin, ds_missing_CountyFIPS_end]).reset_index(drop=True)
	ds_missing_CountyFIPS = gpd.GeoDataFrame(ds_missing_CountyFIPS, geometry="geometry", crs="EPSG:4269")
	ds_missing_CountyFIPS = ds_missing_CountyFIPS.to_crs("EPSG:4326")
	ds_missing_CountyFIPS = gpd.sjoin(ds_missing_CountyFIPS, counties.drop(columns=["county_geom","STATEFP","COUNTYFP"]), op="within", how="left")

	del(ds_missing_CountyFIPS_begin, ds_missing_CountyFIPS_end)

	ds_other = ds_missing_CountyFIPS[(pd.isnull(ds_missing_CountyFIPS["CountyFIPS"]))].drop(columns=["index_right","CountyFIPS"]).reset_index(drop=True)
	ds_missing_CountyFIPS = ds_missing_CountyFIPS[(~pd.isnull(ds_missing_CountyFIPS["CountyFIPS"]))].drop(columns=["STATE_FIPS","STATE","geometry","index_right","ZoneFIPS","CZ_TYPE","CZ_FIPS","CZ_NAME"]).reset_index(drop=True)

	"""
	STEP 4: Deal with rows that are STILL missing CountyFIPS (ds_other)

	What is ds_other?
	(4A) it contains points (lon, lat) that do not belong to any county or forecast zone
	(4B) it contains points with empty lat, lon, and whose CZ_FIPS don't seem to be associated with any zone or counties. It almost seems like the FIPS are wrong.
	    - we cannot do much in this case. If the CZ_NAME contains a county name we use it, else we discard the rows
	"""

	ds_other_coords = ds_other[(pd.notnull(ds_other["geometry"].x) & pd.notnull(ds_other["geometry"].y))]
	ds_other_nocoords = ds_other[(pd.isnull(ds_other["geometry"].x) | pd.isnull(ds_other["geometry"].y))]

	#### STEP 4A
	# Deal with ds_other_nocoords by using county name-based regex expressions
	ds_other_nocoords["counties"] = ds_other_nocoords["STATE"].apply(lambda x: state2county[x] if x in state2county else np.nan)

	# drop rows with states like atlanticsouth or gulfofmexico
	ds_other_nocoords = ds_other_nocoords.dropna(subset=["counties"])

	# obtain the actual county names from the CZ_NAME statements. 
	ds_other_nocoords["CZ_NAME_regex"] = ds_other_nocoords["CZ_NAME"].str.replace(r'[^A-Za-z]+', '', regex=True).str.lower()
	ds_other_nocoords["CountyName"] = ds_other_nocoords.apply(lambda x: [y for y in x["counties"] if y in x["CZ_NAME_regex"]], axis=1)

	# some events would be spread across multiple counties --> we find multiple matches --> we explode each county it into an individual row.
	ds_other_nocoords = ds_other_nocoords.explode(column="CountyName").reset_index(drop=True)

	# we will be left with entries where no match was found, and so we eliminate those rows
	ds_other_nocoords = ds_other_nocoords[(~pd.isnull(ds_other_nocoords["CountyName"]))]

	# add countyfips corresponding to that county
	ds_other_nocoords = pd.merge(ds_other_nocoords, county2fips, left_on=["STATE","CountyName"], right_on=["StateName","CountyName"], how="left")
	ds_other_nocoords = ds_other_nocoords.dropna(subset=["CountyFIPS"]).reset_index(drop=True)

	# only extract the columns of interest
	ds_other_nocoords = ds_other_nocoords[["BEGIN_DATE_TIME","END_DATE_TIME","CZ_TIMEZONE","EPISODE_ID","EVENT_ID","EVENT_TYPE","CountyFIPS"]].drop_duplicates().reset_index(drop=True)


	#### STEP 4B
	# Deal with ds_other_coords by finding the closest county to the lat lon of the event
	def find_nearest_county(points_df, polygons_df):
	    # Build the spatial index for polygons_df
	    polygons_index = polygons_df.sindex

	    # Create an empty list to store the nearest polygon ID for each point
	    nearest_polygon_ids = []

	    # Iterate over each point in points_df
	    for index, point in points_df.iterrows():
	        # Get the bounds of the point geometry
	        point_bounds = point.geometry.bounds

	        # Find the nearest polygon using the spatial index
	        nearest_index = polygons_index.nearest(point_bounds)

	        # Get the ID of the nearest polygon
	        nearest_polygon_id = polygons_df.iloc[nearest_index]["CountyFIPS"].values
	        # print(nearest_polygon_id)

	        # Add the nearest polygon ID to the list
	        nearest_polygon_ids.append(nearest_polygon_id)

	    # # Add the nearest polygon ID column to points_df
	    points_df["CountyFIPS"] = nearest_polygon_ids
	    points_df = points_df.explode(column="CountyFIPS").reset_index(drop=True)
	    return points_df
	    
	ds_other_coords = find_nearest_county(points_df=ds_other_coords.copy(), polygons_df=counties[["CountyFIPS","geometry"]].drop_duplicates().reset_index(drop=True))
	# only extract the columns of interest
	ds_other_coords = ds_other_coords[["BEGIN_DATE_TIME","END_DATE_TIME","CZ_TIMEZONE","EPISODE_ID","EVENT_ID","EVENT_TYPE","CountyFIPS"]].drop_duplicates().reset_index(drop=True)

	"""
	STEP 5: Combine all the dataframes together
	"""
	ds = pd.concat([ds, ds_missing_CountyFIPS, ds_other_coords, ds_other_nocoords]).reset_index(drop=True)

	"""
	STEP 6: 
	- convert CZ_TIMEZONE to native Python format. 
	- convert local ts to UTC (POUS is in UTC).
	"""
	timezones = {"EST-5":"US/Eastern","EDT-4":"US/Eastern","CST-6":"US/Central","CDT-5":"US/Central","MST-7":"US/Mountain","PST-8":"US/Pacific","PDT-7":"US/Pacific","AKST-9":"US/Alaska","HST-10":"US/Hawaii"}
	ds.loc[:,"tz"] = ds["CZ_TIMEZONE"].apply(lambda x: timezones[x] if x in timezones.keys() else np.nan)
	# we drop nans and remove virgin islands
	ds = ds.dropna().reset_index(drop=True)

	# convert timestamps
	ds.loc[:,"BEGIN_DATE_TIME"] = pd.to_datetime(ds["BEGIN_DATE_TIME"], format='%d-%b-%y %H:%M:%S')
	ds.loc[:,"END_DATE_TIME"] = pd.to_datetime(ds["END_DATE_TIME"], format='%d-%b-%y %H:%M:%S')
	ds = ds.sort_values(by=["CountyFIPS","BEGIN_DATE_TIME"])

	# iterating using groupby because some countyFIPS have two tz. County-wise iteration also helps with nonexistent and ambigous timestamps due to DST.
	ds_utc = []
	for idx, grouped in ds.groupby(["CountyFIPS","tz"]):
	    grouped.loc[:,"BEGIN_DATE_TIME"] = grouped.apply(lambda row: row["BEGIN_DATE_TIME"].tz_localize(row['tz'], nonexistent='shift_forward').tz_convert('UTC').tz_localize(None), axis=1)
	    grouped.loc[:,"END_DATE_TIME"] = grouped.apply(lambda row: row["END_DATE_TIME"].tz_localize(row['tz'], nonexistent='shift_forward', ambiguous='NaT').tz_convert('UTC').tz_localize(None), axis=1)
	    ds_utc.append(grouped)
	ds_utc = pd.concat(ds_utc).dropna().reset_index(drop=True) # we lose only 1 ambiguous reading here
	ds_utc = (
	    ds_utc[["BEGIN_DATE_TIME","END_DATE_TIME","EPISODE_ID","EVENT_ID","EVENT_TYPE","CountyFIPS"]]
	    .rename(columns={"BEGIN_DATE_TIME":"event_start", "END_DATE_TIME":"event_end", "EVENT_ID":"event_id","EVENT_TYPE":"event_type","EPISODE_ID":"episode_id"}) 
	)

	"""
	STEP 7: For each (event_id, event_type, CountyFIPS) tuple, create one row per day of the event. 
	"""
	db = []
	for _, row in ds_utc.iterrows():
	    event_start = row["event_start"].date()
	    event_end = row["event_end"].date()

	    date_range = pd.date_range(event_start, event_end, freq="1D")

	    temp = pd.DataFrame({
	        "episode_id":row["episode_id"],
	        "event_id":row["event_id"],
	        "event_type":row["event_type"],
	        "CountyFIPS":row["CountyFIPS"],
	        "event_date":date_range,
	    })
	    db.append(temp)
	#
	db = pd.concat(db).reset_index(drop=True)

	#save the storms dataframe
	db.to_pickle(os.path.join(output_dir, output_filename))

	print("File saved @ {}".format(datetime.datetime.now()), flush=True)

	return db


#### February 10, 2023 (considers SAIDI calculated using the IEEE standard as well as other standards)
def process_EIA(input_dir):
	#
	states = []
	usa = []
	for year in np.arange(2016,2022):
		df = pd.read_excel(os.path.join(input_dir, "Reliability_{}.xlsx".format(year)))
		# df = df.iloc[:,0:16]
		df = df.drop(df.index[-1])
		# dataset's format got updated in 2021
		if year != 2021:
			df.columns = df.iloc[0]
			df = df.drop(df.index[0])
			# Only 2019 data has an extra column called "Short Form". We remove it for consistency.
			if year == 2019:
				df = df.drop(columns=["Short Form"])
			#
			df.columns = ['year', 'utility_number', 'utility_name', 'state', '_', 'saidi_w_med_ieee', '_', '_','saidi_wo_med_ieee', '_', '_','_', '_', '_', 'customers_ieee', '_', '_', 'saidi_w_med_other', '_', '_', 'saidi_wo_med_other', '_', '_', 'customers_other', '_', '_', '_', '_']
			df = df.drop(columns=["_"])
		else:
			df.columns = df.iloc[1]
			df = df.drop(df.index[[0,1]])
			df.columns = ['year', 'utility_number', 'utility_name', 'state', '_','saidi_w_med_ieee', '_', '_', 'saidi_wo_med_ieee', '_', '_', '_', '_', '_', 'customers_ieee', '_', '_', 'saidi_w_med_other', '_', '_', 'saidi_wo_med_other', '_', '_', 'customers_other', '_', '_', '_', '_']
			df = df.drop(columns=["_"])
		# replace "." with nans
		for col in ["saidi_w_med_ieee","saidi_wo_med_ieee","customers_ieee", "saidi_w_med_other","saidi_wo_med_other","customers_other"]:
			df[col] = df[col].apply(lambda x: x if (isinstance(x, float)==True) | (isinstance(x, int)==True) else np.nan)

		# fill nans in IEEE standard columns with values from the other standard columns (if available)
		df.loc[:,"saidi_w_med_ieee"] = df["saidi_w_med_ieee"].fillna(df["saidi_w_med_other"])
		df.loc[:,"saidi_wo_med_ieee"] = df["saidi_wo_med_ieee"].fillna(df["saidi_wo_med_other"])
		df.loc[:,"customers_ieee"] = df["customers_ieee"].fillna(df["customers_other"])

		# remove all the other standard columns
		df = df.drop(columns = ["saidi_w_med_other", "saidi_wo_med_other", "customers_other"])

		# rename IEEE columns
		df = df.rename(columns={"saidi_w_med_ieee":"saidi_w_med", "saidi_wo_med_ieee":"saidi_wo_med", "customers_ieee":"customers"})

		# compute outage duration 
		df["customer_outage_hours_w_med"] = df["saidi_w_med"].mul(df["customers"])/60
		df["customer_outage_hours_wo_med"] = df["saidi_wo_med"].mul(df["customers"])/60
		df["customer_outage_hours_only_med"] = df["customer_outage_hours_w_med"] - df["customer_outage_hours_wo_med"]

		"""
		state-level
		"""
		# total outage duration and total customers in state
		df = df.groupby(["state","year"])[["customer_outage_hours_w_med","customer_outage_hours_wo_med","customer_outage_hours_only_med","customers"]].sum().reset_index()
		# calculate saidi - storm and no storm
		df.loc[:,"saidi_w_med"] = df["customer_outage_hours_w_med"].div(df["customers"]) # overall SAIDI (includes major events)
		df.loc[:,"saidi_wo_med"] = df["customer_outage_hours_wo_med"].div(df["customers"]) # SAIDI without major events
		df.loc[:,"saidi_only_med"] = df["customer_outage_hours_only_med"].div(df["customers"]) # SAIDI only major events
		#
		states.append(df)

		"""
		USA-level
		"""
		# total outage duration and total customers in state
		c = df.groupby(["year"])[["customer_outage_hours_w_med","customer_outage_hours_wo_med","customer_outage_hours_only_med","customers"]].sum().reset_index()
		# calculate saidi - storm and no storm
		c.loc[:,"saidi_w_med"] = c["customer_outage_hours_w_med"].div(c["customers"]) # overall SAIDI (includes major events)
		c.loc[:,"saidi_wo_med"] = c["customer_outage_hours_wo_med"].div(c["customers"]) # SAIDI without major events
		c.loc[:,"saidi_only_med"] = c["customer_outage_hours_only_med"].div(c["customers"]) # SAIDI only major events
		#
		usa.append(c)
	#
	states = pd.concat(states).reset_index(drop=True)
	usa = pd.concat(usa).reset_index(drop=True)
	return states, usa


def storms_and_pous(pous, name2fips, storm, output_dir, output_version):
	#
	def q75(x):
	    return x.quantile(0.75)
	#
	# add countyfips column to daily SAIDI POUS data
	name2fips = name2fips[["StateName","CountyName","CountyFIPS"]].drop_duplicates()
	pous = pd.merge(pous, name2fips, on=["StateName","CountyName"], how="left")
	# add event type and event id
	storm = storm.groupby(["CountyFIPS","event_date"])["event_type"].apply(lambda x: np.unique(np.array(x))).reset_index()
	db = pd.merge(pous, storm, left_on=["CountyFIPS","RecordDate"], right_on=["CountyFIPS","event_date"], how="left")
	db["year"] = db["RecordDate"].dt.year
	db["month"] = db["RecordDate"].dt.month
	#
	db.loc[:,"CustomerHoursOutTotal_only_storm"] = db.apply(lambda x: 0 if pd.isnull(x["event_date"])==True else x["CustomerHoursOutTotal"], axis=1)
	db.loc[:,"CustomerHoursOutTotal_no_storm"] = db.apply(lambda x: 0 if pd.isnull(x["event_date"])==False else x["CustomerHoursOutTotal"], axis=1)
	# code.interact(local=locals())
	"""
	SAIDI (overall and storm) (monthly and yearly)
	- county level
	- state level
	- country level
	"""
	def calculate_saidi(dy, spatial_agg_level, temporal_agg_level):
		#
		if temporal_agg_level == "yearly": 
			if spatial_agg_level == "county":
				agg_cols = ["StateName","CountyName","CountyFIPS","year"]
			elif spatial_agg_level == "state":
				# calculate total customer outage hours and CT in each state on each day
				dy = dy.groupby(["StateName","RecordDate","year"])[["CustomerHoursOutTotal","CustomerHoursOutTotal_only_storm","CustomerHoursOutTotal_no_storm","CustomersTracked_clean"]].sum().reset_index()
				agg_cols = ["StateName","year"]
			elif spatial_agg_level == "country":
				# calculate total customer outage hours and CT in the country on each day
				dy = dy.groupby(["RecordDate","year"])[["CustomerHoursOutTotal","CustomerHoursOutTotal_only_storm","CustomerHoursOutTotal_no_storm","CustomersTracked_clean"]].sum().reset_index()
				agg_cols = ["year"]	
			else:
				None
		elif temporal_agg_level == "monthly":
			if spatial_agg_level == "county":
				agg_cols = ["StateName","CountyName","CountyFIPS","year","month"]
			elif spatial_agg_level == "state":
				# calculate total customer outage hours and CT in each state on each day
				dy = dy.groupby(["StateName","RecordDate","year","month"])[["CustomerHoursOutTotal","CustomerHoursOutTotal_only_storm","CustomerHoursOutTotal_no_storm","CustomersTracked_clean"]].sum().reset_index()
				agg_cols = ["StateName","year","month"]
			elif spatial_agg_level == "country":
				# calculate total customer outage hours and CT in the country on each day
				dy = dy.groupby(["RecordDate","year","month"])[["CustomerHoursOutTotal","CustomerHoursOutTotal_only_storm","CustomerHoursOutTotal_no_storm","CustomersTracked_clean"]].sum().reset_index()
				agg_cols = ["year","month"]	
			else:
				None
		else:
			None	
		#
		dy = dy.groupby(agg_cols).agg({"CustomerHoursOutTotal":"sum", "CustomerHoursOutTotal_only_storm":"sum","CustomerHoursOutTotal_no_storm":"sum","CustomersTracked_clean":q75}).reset_index()
		dy.columns = agg_cols + ["CustomerHoursOutTotal", "CustomerHoursOutTotal_only_storm", "CustomerHoursOutTotal_no_storm", "CustomersTracked"]
		dy.loc[:,"saidi"] = dy["CustomerHoursOutTotal"].div(dy["CustomersTracked"])
		dy.loc[:,"saidi_only_storm"] = dy["CustomerHoursOutTotal_only_storm"].div(dy["CustomersTracked"])
		dy.loc[:,"saidi_no_storm"] = dy["CustomerHoursOutTotal_no_storm"].div(dy["CustomersTracked"])
		return dy
	#
	# YEARLY
	dy_county = calculate_saidi(dy=db.copy(), spatial_agg_level="county", temporal_agg_level="yearly")
	dy_state = calculate_saidi(dy=db.copy(), spatial_agg_level="state", temporal_agg_level="yearly")
	dy_country = calculate_saidi(dy=db.copy(), spatial_agg_level="country", temporal_agg_level="yearly")
	# MONTHLY
	dm_county = calculate_saidi(dy=db.copy(), spatial_agg_level="county", temporal_agg_level="monthly")
	dm_state = calculate_saidi(dy=db.copy(), spatial_agg_level="state", temporal_agg_level="monthly")
	dm_country = calculate_saidi(dy=db.copy(), spatial_agg_level="country", temporal_agg_level="monthly")
	# Save all files
	dy_county.to_csv(os.path.join(output_dir, "yearly_county_saidi_v{}.csv".format(output_version)))
	dy_state.to_csv(os.path.join(output_dir, "yearly_state_saidi_v{}.csv".format(output_version)))
	dy_country.to_csv(os.path.join(output_dir, "yearly_country_saidi_v{}.csv".format(output_version)))
	#
	dm_county.to_csv(os.path.join(output_dir, "monthly_county_saidi_v{}.csv".format(output_version)))
	dm_state.to_csv(os.path.join(output_dir, "monthly_state_saidi_v{}.csv".format(output_version)))
	dm_country.to_csv(os.path.join(output_dir, "monthly_country_saidi_v{}.csv".format(output_version)))
	#
	print("Files saved @ {}".format(datetime.datetime.now()), flush=True)
	return None	


#### PAPER/SI PLOT
def total_utilities_covered(pous_data_dir, eia_data_dir):
	"""
	Better if viewed in jupyter notebook
	Compares total number of utilities covered in EIA with POUS
	"""
	#
	# EIA
	utils = []
	for year in np.arange(2016,2022):
	    df = pd.read_excel(os.path.join(eia_data_dir, "Reliability_{}.xlsx".format(year)))
	    df = df.iloc[:,0:16]
	    df = df.drop(df.index[-1])
	    # dataset's format got updated in 2021
	    if year != 2021:
	        df.columns = df.iloc[0]
	        df = df.drop(df.index[0])
	        df = df[["Data Year", "Utility Number", "Utility Name", "State", "SAIDI With MED", "SAIDI Without MED", "Number of Customers"]]
	        df = df.rename(columns={"Data Year":"year", "Utility Number":"utility_number","Utility Name":"utility_name","State":"state","SAIDI With MED":"saidi_w_storm","SAIDI Without MED":"saidi_wo_storm","Number of Customers":"customers"})
	    else:
	        df.columns = df.iloc[1]
	        df = df.drop(df.index[[0,1]])
	        df.columns = ["year", "utility_number", "utility_name", "state", "_", "saidi_w_storm", "_", "_", "saidi_wo_storm", "_", "_", "_", "_", "_", "customers", "_"]
	        df = df.drop(columns=["_"])
	    #
	    df = df[["year","state","utility_name"]].drop_duplicates()
	    df.loc[:,"utility_name"] = df["utility_name"].str.replace(r'[^A-Za-z]+', '', regex=True).str.lower()
	    df.loc[:,"state"] = df["state"].str.replace(r'[^A-Za-z]+', '', regex=True).str.lower()
	    utils.append(df)
	utils = pd.concat(utils).reset_index(drop=True).rename(columns={"utility_name":"eia_utility"})
	#
	# POUS
	raw_filename = "POUS_Export_CountyByUtility_Daily_2017-2021.csv"
	dc = pd.read_csv(os.path.join(pous_data_dir, raw_filename), encoding='utf-16')
	dc.loc[:,"RecordDate"] = pd.to_datetime(dc.loc[:,"RecordDate"], format="%Y-%m-%d")
	dc.loc[:,"year"] = dc.loc[:,"RecordDate"].dt.year
	dc = dc[["year","StateName","UtilityName"]].drop_duplicates()                                                                             
	#
	dc.loc[:,"StateName"] = dc["StateName"].str.replace(r'[^A-Za-z]+', '', regex=True).str.lower()
	dc.loc[:,"UtilityName"] = dc["UtilityName"].str.replace(r'[^A-Za-z]+', '', regex=True).str.lower()
	dc = dc[["year","StateName","UtilityName"]].drop_duplicates().rename(columns={"UtilityName":"pous_utility"}) 
	#
	# Compare the two at country-year level 
	u1 = utils.groupby(["year"])[["eia_utility"]].nunique().reset_index()
	c1 = dc.groupby(["year"])[["pous_utility"]].nunique().reset_index()
	db1 = pd.merge(u1, c1, on=["year"])
	db1 = db1[(db1["year"].isin([2017,2018,2019,2020,2021]))]
	#
	# plot
	n = db1["year"].values
	x = db1["eia_utility"].values
	y = db1["pous_utility"].values
	#
	fs = 16
	fig, ax = plt.subplots(figsize=(7,6))
	ax.scatter(x, y, c="r", marker="*")
	ax.set_xlim([400,1200])
	ax.set_ylim([400,800])
	for i, txt in enumerate(n):
		ax.annotate(txt, (x[i]+0.12, y[i]+0.12), fontsize=fs)
	ax.set_xlabel("Total utilities covered - EIA", fontsize=fs)
	ax.set_ylabel("Total utilities covered - POUS", fontsize=fs)
	ax.tick_params(axis='both', which='major', labelsize=fs)
	#
	plt.grid()
	plt.tight_layout()
	plt.show()
	return None


# created on: June 16, 2023
def pous_county_to_MajorUtility_mapping():
	return None


def main():
	#-----unity
	pous_data_dir = "/gypsum/eguide/projects/zshah/data/POUS"
	stormdata_dir = "/gypsum/eguide/projects/zshah/data/USA_storms_data/"
	counties_shapefile = "/gypsum/eguide/projects/zshah/data/shapefiles/USA/cb_2018_us_county_500k/cb_2018_us_county_500k.shp"
	eia_data_dir = "/gypsum/eguide/projects/zshah/data/EIA_reliability_2016_2021"
	output_dir = "/gypsum/eguide/projects/zshah/data/POUS_and_storms"

	#----local
	# stormdata_dir = "/Users/zealshah/Documents/Poweroutages-inequity/poweroutages-inequity/data/storm_data",
	# counties_shapefile = "/Users/zealshah/Documents/Poweroutages-inequity/poweroutages-inequity/data/shapefiles/cb_2018_us_county_500k/cb_2018_us_county_500k.shp"
	# eia_data_dir = "/Users/zealshah/Documents/Poweroutages-inequity/poweroutages-inequity/data/eia_data"
	#
	#------
	print("-*-"*30, flush=True)
	print("Started @ {}".format(datetime.datetime.now()), flush=True)

	pous = POUS_county_daily_outages(input_dir=pous_data_dir)
	print("POUS loaded @ {}".format(datetime.datetime.now()), flush=True)
	#
	name2fips = pd.read_pickle(os.path.join(pous_data_dir, "state_county_city_utility.pkl")) #maps countyname to fips
	name2fips["CountyFIPS"] = name2fips["CountyFIPS"].apply(lambda x: str(int(x)).zfill(5))
	print("name2fips loaded @ {}".format(datetime.datetime.now()), flush=True)
	#
	storm = get_stormevents_db(
		stormdata_dir = stormdata_dir,
		counties_shapefile = counties_shapefile,
		use_pickled = True
	)
	print("Storm event data loaded @ {}".format(datetime.datetime.now()), flush=True)
	#
	states, usa = process_EIA(input_dir=eia_data_dir)
	print("EIA data loaded @ {}".format(datetime.datetime.now()), flush=True)
	#
	storms_and_pous(pous=pous, name2fips=name2fips, storm=storm, output_dir=output_dir, output_version=2) ##<---- use this function to save the monthly and yearly SAIDI values
	#
	print("Ended @ {}".format(datetime.datetime.now()), flush=True)
	print("-*-"*30, flush=True)


if __name__ == '__main__':
	main()