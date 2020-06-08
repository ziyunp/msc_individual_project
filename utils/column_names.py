"""
Summary:
    This file is a mapping of column names to variables that can used to enable tab completion and ensure we are
    referring data with the same name throughout the code.

Project:
    HEADS

Created:
    July 2019

Dependencies:

Support Documentation:
    Confluence - https://andotce.royalmailgroup.com/pages/viewpage.action?pageId=57771370

Repository:
    BitBucket - https://bitbucket.org/rmgdigital/heads

"""
# Isotrack extract columns
EVENT_LAT = "Event_Lat"
EVENT_LONG = "Event_Long"
DUTY_ID = 'Duty_Id'
RSLDUTY_ID = 'rslDutyId'
EVENT_ID = 'Event_Id'
SRC_EVENT_CD = 'Src_Event_Cd'
EVENT_DTTM = 'Event_Dttm'
PLN_START_TIME = 'Pln_Strt_Tm'
ACT_START_TIME = 'Act_Strt_Tm'
ACT_END_TIME = 'Act_End_Tm'
SPEED = 'Speed'
FROM_DEPOT = 'from_depot'
TO_DEPOT = 'to_depot'

# OS Motorway-Junctions data columns
OS_MOTORWAY = 'os_motorway'
OS_JUNCTION = 'os_junction'
OS_LAT = 'os_latitude'
OS_LONG = 'os_longitude'
OS_SECTION = 'os_section'
OS_E = "E"
OS_N = "N"

# Generated columns
CLUSTER = 'cluster'
ROAD = 'road'
NEXT_ROAD = 'next_road'
ROAD_TYPE = 'road_type'
LINK_NAME = 'link_name'
C_WAY = "c_way"
JUNCTION = 'junction'
BOUND = 'bound'
SECTION = 'section'
ORDERED_SECTION = 'ordered_section'
NEXT_SECTION = 'next_section'
SECT_MIN_JCT = 'section_min_jct'
NEXT_SECT_MAX_JCT = 'next_section_max_jct'
MISSING_SECT = 'missing_sect'
LEG_ID = 'leg_id'
DATE_STR = 'date_str'
TIMESTAMP = 'timestamp'
OS_NEXT_JCT = 'os_next_jct'
OS_NEXT_SECT = 'os_next_section'
ARTIFICIAL_PING = 'artificial_ping'
INDEX = 'ping_index'
NEXT_LAT = 'next_lat'
SECTION_GAP = 'section_gap'
DIRECTION = 'direction'
START_TIME = "start_time"
END_TIME = "end_time"
ENTITY = "entity"
ENTITY_LIST = "entity_list"
ENTITIES = "entities"
ROAD_RANK = "road_rank"
REPRESENTATIVE = "reduce_flag"
REDUCE_FLAG = "reduce_flag"
CLUSTER_ID = "cluster_id"
FROM_TO = "from_to"
RUNTIME = "runtime"
MIN_RUNTIME = "min_runtime"
MAX_RUNTIME = "max_runtime"
MEDIAN_RUNTIME = "median_runtime"
MEAN_RUNTIME = "mean_runtime"
COUNT_RUNTIME = "count_runtime"
GAPFILLED = "gapfilled"
ENTITY_LIST_AS_STR = "entity_list_as_str"
COUNT_LEG_ID = "count_leg_id"
HOUR_DEPARTED = 'Hour_departed'
OVERNIGHT = 'Overnight'
MIN_RUNTIME_FT = "min_runtime_filter"
MAX_RUNTIME_FT = "max_runtime_filter"
MEDIAN_RUNTIME_FT = "median_runtime_filter"
MEAN_RUNTIME_FT = "mean_runtime_filter"
COUNT_RUNTIME_FT = "count_runtime_filter"
COUNT_LEG_ID_FT = "count_leg_id_filter"
UNIQUE = "unique"
TODAYS_DATA = "todays_data"
TEMP_CLUSTER_ID = "temp_cluster_id"
TEMP_CLUSTER = "temp_cluster"
TEMP_LAST_24H = "temp_last_24"
TRAVERSED_24H = "traversed_24h"

# ALL RMG LOCATIONS columns
DEPOT = "location_name"
LAT = "lat"
LONG = "lon"
