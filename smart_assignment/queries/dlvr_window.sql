WITH tw_data AS (
    WITH days AS (
        SELECT 'M' AS day
        UNION
        SELECT 'T' AS day
        UNION
        SELECT 'W' AS day
        UNION
        SELECT 'R' AS day
        UNION
        SELECT 'F' AS day
        UNION
        SELECT 'S' AS day
        UNION
        SELECT 'U' AS day
    ),

    co_filter AS (
        SELECT *
        FROM dw.dim_operatingcompany AS co
        WHERE 1 = 1
            AND co.operatingcompanynumber IN ('067') -- {OPCO}  -- TODO: make this a parameter
    )

    SELECT tw.operatingcompanyid
         , co.operatingcompanynumber AS co_nbr
         , tw.regionid
         , tw.srcstopid AS cust_nbr
         , tw.days
         , days.day AS tw_days
         , TO_CHAR(tw.opentime, 'HH24:MI:SS') AS bsns_open_tm
         , TO_CHAR(tw.closetime, 'HH24:MI:SS') AS bsns_close_tm
         , TO_CHAR(tw.tw1opentime, 'HH24:MI:SS') AS tw1_open_tm
         , TO_CHAR(tw.tw1closetime, 'HH24:MI:SS') AS tw1_close_tm
         , tw.tw1opentime  AS tw1_open_tm_tm
         , tw.tw1closetime AS tw1_close_tm_tm
         , TO_CHAR(tw.tw2opentime, 'HH24:MI:SS') AS tw2_open_tm
         , TO_CHAR(tw.tw2closetime, 'HH24:MI:SS') AS tw2_close_tm
         , tw.effectivethrudate
         , tw.effectivefromdate
         , tw.usermodified
         -- , CASE WHEN tw.datemodified > to_date({DATE_OUTREACH}, 'YYYYMMDD', FALSE) THEN 1 ELSE 0 END AS updated_recently
         , CASE WHEN tw.datemodified > (current_date - 6) THEN 1 ELSE 0 END AS updated_recently  -- TODO: make this a parameter
    FROM dw.dim_locationtimewindowoverride AS tw
             JOIN days
                  ON tw.days LIKE '%' || days.day || '%' --explosion of tw_days into individual day
             JOIN co_filter AS co
                  ON tw.operatingcompanyid = co.operatingcompanyid
    WHERE 1 = 1
      AND tw.iscurrentversion = 1
      AND lower(tw.scenario) = 'delivery'
      AND tw.type = 'SIT'
      AND len(tw.srcstopid) <= 6
)
SELECT twor.co_nbr || '-' || twor.cust_nbr as co_cust_nbr
       , ploc.description
       , ploc.deliverydays
       , ploc.region1 as cty_nm
       , LEFT(ploc.postalcode,5) as zip_cd
       , decode(twor.tw_days, 'M', 'Monday', 'T', 'Tuesday', 'W', 'Wednesday', 'R', 'Thursday', 'F', 'Friday', 'S',
                        'Saturday', 'U', 'Sunday', 'NA') as dlvry_day_nm
       , CASE WHEN ploc.deliverydays LIKE '%' || twor.tw_days || '%' THEN 1
           ELSE 0 END AS scheduled_day_ind
       , twor.operatingcompanyid
       , twor.regionid
       , twor.days
       , twor.tw_days
       , twor.bsns_open_tm
       , twor.bsns_close_tm
       , twor.tw1_open_tm
       , twor.tw1_close_tm
       , twor.tw1_open_tm_tm
       , twor.tw1_close_tm_tm
       , twor.tw2_open_tm
       , twor.tw2_close_tm
       , twor.effectivefromdate
       , twor.effectivethrudate
       , twor.usermodified
       , twor.updated_recently
FROM dw.dim_planneddeliverylocation AS ploc

JOIN tw_data AS twor
    ON ploc.srcregionid = twor.regionid
        AND ploc.srclocationid = twor.cust_nbr

WHERE 1 = 1
    AND ploc.iscurrentversion = 1
    AND ploc.type = 'SIT'




