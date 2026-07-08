WITH date_range AS (
    SELECT
        MAX(datekey) AS end_date
        , MIN(datekey) AS start_date
    FROM dw.dim_timebase AS timebase
    WHERE 1 = 1
        -- AND dateid between {start_date} AND {end_date}
        -- AND dateid between '20260701' AND '20260707'
        AND dateid BETWEEN
            TO_CHAR(CURRENT_DATE - INTERVAL '28 days', 'YYYYMMDD')::INT AND TO_CHAR(CURRENT_DATE, 'YYYYMMDD')::INT  -- TODO: make this a parameter
)

   , fiscal_cal AS (
    SELECT dateid
         , datekey
         , fiscalyear
         , fiscalweekid
         , fiscalperiod
         , daynameshort
    FROM dm.dim_time
        INNER JOIN date_range
            ON datekey BETWEEN date_range.start_date AND date_range.end_date
)

   , co AS (
    SELECT co.*
    FROM dw.dim_operatingcompany AS co
    WHERE co.operatingcompanynumber
    IN ('067') -- {OPCO}  -- TODO: make this a parameter
    )

SELECT
    co.operatingcompanynumber as co_nbr
    , dd1.srcstopid as cust_nbr
    , ploc.description AS cust_nm_
    , ploc.deliverydays AS cust_dlvry_day_
    , co.operatingcompanynumber || '-' || dd1.srcstopid as co_cust_nbr
--     , cust.acct_typ_cd
    , route.srcrouteid AS route_id
    , route.description as route_nm
    , routes.weightcapacity as route_weight_capacity
    , routes.cubecapacity as route_cube_capacity
    , plnd_dlvry_stp.routestartdateid AS route_start_date
    , TO_CHAR(TO_DATE(plnd_dlvry_stp.deliverydaysdateid, 'YYYYMMDD', FALSE),'Day') AS dlvry_day_nm
--     , fiscal_cal.daynameshort as route_start_day
    -- planned info:
    , plnd_dlvry_stp.weight AS weight
    , plnd_dlvry_stp.cube as cubes
    , plnd_dlvry_stp.cases as cases
    , plnd_dlvry_stp.sequencenumber AS planned_stop_seq
    , plnd_dlvry_stp.traveltime/60 planned_trvl_tm
    , plnd_dlvry_stp.servicetime/60 planned_srvc_tm
    , trips.planlocationminutes
    , plnd_dlvry_stp.arrivaldatetime as planned_arrive_time
    , plnd_dlvry_stp.arrivaldatetime + ((plnd_dlvry_stp.servicetime / 60.0) * interval '1 minute') as planned_depart_time
    , plnd_dlvry_stp.stoptype
    , CASE WHEN plnd_dlvry_stp.stoptype = 'L' THEN plnd_dlvry_stp.servicetime/60 end as fix_service_time
    , dd1.type
    , LOWER(ploc.region1) as city
    , ploc.longitude  -- customer long
    , ploc.latitude -- customer lat
FROM dm.fact_dailyplanneddeliverystops AS plnd_dlvry_stp
    JOIN fiscal_cal
        ON plnd_dlvry_stp.routestartdateid = fiscal_cal.dateid

    JOIN co
        ON plnd_dlvry_stp.operatingcompanyid = co.operatingcompanyid

    LEFT JOIN dm.dim_deliverystop AS dd1
        ON plnd_dlvry_stp.operatingcompanyid = dd1.operatingcompanyid
            AND plnd_dlvry_stp.deliverystopid = dd1.deliverystopid

    LEFT JOIN dm.fact_dailydeliverytrips AS trips
        ON plnd_dlvry_stp.operatingcompanyid = trips.operatingcompanyid
            AND plnd_dlvry_stp.deliverystopid = trips.deliverystopid
            AND plnd_dlvry_stp.routestartdateid = trips.tripstartdateid
            AND plnd_dlvry_stp.deliveryrouteid=trips.deliveryrouteid

    LEFT JOIN dm.dim_deliveryroute AS route
        ON plnd_dlvry_stp.operatingcompanyid = route.operatingcompanyid
            AND plnd_dlvry_stp.deliveryrouteid = route.deliveryrouteid

    LEFT JOIN dm.fact_dailyplanneddeliveryroutes as routes
        ON plnd_dlvry_stp.operatingcompanyid = routes.operatingcompanyid
            AND plnd_dlvry_stp.routestartdateid = routes.startdateid
            AND plnd_dlvry_stp.deliveryrouteid = routes.deliveryrouteid
            AND routes.stopcount is not null
    LEFT JOIN dw.dim_custmuanationalid AS cust
        ON plnd_dlvry_stp.operatingcompanyid = cust.operatingcompanyid
            AND plnd_dlvry_stp.deliverystopid = cust.deliverystopid
            AND cust.curr_rec_ind IN ('Y')
            and current_date between cust.cust_ship_to_rec_eff_dt and cust_ship_to_rec_trm_dt

    LEFT JOIN dw.dim_planneddeliverylocation AS ploc
        ON plnd_dlvry_stp.operatingcompanyid = ploc.operatingcompanyid
            AND dd1.srcstopid = ploc.srclocationid
            AND dd1.type = ploc.type
            AND ploc.iscurrentversion = 1
WHERE 1 = 1
    AND plnd_dlvry_stp.stoptype IN ('STP')
    AND plnd_dlvry_stp.cases is not null
    AND dd1.type = 'SIT'
ORDER BY route_id, plnd_dlvry_stp.routestartdateid, planned_stop_seq


