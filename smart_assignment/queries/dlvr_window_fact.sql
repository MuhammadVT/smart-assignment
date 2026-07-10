WITH
     date_range AS (

    SELECT
        MAX(datekey) AS end_date
        , MIN(datekey) AS start_date
    FROM dw.dim_timebase AS timebase
    WHERE 1 = 1
        -- AND dateid between {start_date} AND {end_date}
       AND dateid BETWEEN
            TO_CHAR(CURRENT_DATE - INTERVAL '28 days', 'YYYYMMDD')::INT AND TO_CHAR(CURRENT_DATE, 'YYYYMMDD')::INT  -- TODO: make this a parameter
)

   , fiscal_cal AS (
    SELECT dateid
         , datekey
         , fiscalyear
         , fiscalweekid
         , fiscalperiod
    FROM dm.dim_time
        INNER JOIN date_range
            ON datekey BETWEEN date_range.start_date AND date_range.end_date
)

   , co AS (
    SELECT co.*
    FROM dw.dim_operatingcompany AS co
    WHERE co.operatingcompanynumber
    IN ('067') -- {OPCO}  TODO: make this a parameter
    )

SELECT
    co.operatingcompanynumber as co_nbr
    , dd1.srcstopid as cust_nbr
    , co.operatingcompanynumber || '-' || dd1.srcstopid as co_cust_nbr
    , cust.acct_typ_cd
    , cust.dist_id AS district
    , cust.terr_cd AS territory
    , ploc.description AS cust_nm_
    , ploc.deliverydays AS cust_dlvry_day_
    , route.srcrouteid AS route_id
    , route.description as route_nm
    , fiscal_cal.fiscalyear
    , fiscal_cal.fiscalweekid
    , fiscal_cal.fiscalperiod
    , plnd_dlvry_stp.sequencenumber AS stp_seq_nbr
    , plnd_dlvry_stp.deliverystopid AS dlvry_stp_id
    , LOWER(ploc.region1) as city
    , ploc.longitude  -- customer long
    , ploc.latitude -- customer lat
    , substring(ploc.postalcode,0,6) as postalcode
    , plnd_dlvry_stp.offdaydelivery::text AS off_day_dlvry
    , plnd_dlvry_stp.routestartdateid AS rte_strt_dt
    , plnd_dlvry_stp.cases as cases
    , plnd_dlvry_stp.cube as cubes
    , plnd_dlvry_stp.weight as weights
    , plnd_dlvry_stp.distance/100 as dist_from_prev_stop
    , plnd_dlvry_stp.traveltime/60 trvl_tm
    , plnd_dlvry_stp.tw1opendatetime
    , plnd_dlvry_stp.tw1closedatetime
    , ploc.srcservicetimetypeid
    , dpt.latitude as dpt_lat
    , dpt.longitude as dpt_long
    , dpt.description as dpt_description
    , dpt.srclocationid as dpt_origin_id
    , routes.cubecapacity AS cubecapacity

--  planned info:
    , TO_CHAR(TO_DATE(plnd_dlvry_stp.deliverydaysdateid, 'YYYYMMDD', FALSE),'Day') AS dlvry_day_nm
    , TRIM(TO_CHAR(plnd_dlvry_stp.routestartdatetime, 'Day')) AS route_delivery_day
    , plnd_dlvry_stp.routestartdatetime
    , TO_CHAR(plnd_dlvry_stp.routestartdatetime, 'HH24:MI:SS') AS rte_strt_tm
    , TO_CHAR(plnd_dlvry_stp.arrivaldatetime, 'HH24:MI:SS') AS schdld_arvl_tm
    , plnd_dlvry_stp.arrivaldatetime
    , TO_CHAR(plnd_dlvry_stp.arrivaldatetime, 'HH24:MI:SS') AS schdld_dprt_tm

--  actual info:
    , datediff(minute, act_dlvry_stp.arrivaldatetime, act_dlvry_stp.departuredatetime) as actl_srvc_tm
    , TO_CHAR(act_dlvry_stp.arrivaldatetime, 'HH24:MI:SS') AS arvl_tm
    , act_dlvry_stp.arrivaldatetime AS actual_arrivaldatetime
    , TRIM(TO_CHAR(act_dlvry_stp.arrivaldatetime, 'Day')) AS arvl_day_nm
    , CASE WHEN act_dlvry_stp.arrivaldatetime IS NULL THEN 'telogis' ELSE 'sts' END AS arvl_tm_src
    , TO_CHAR(act_dlvry_stp.departuredatetime, 'HH24:MI:SS') AS dprtr_tm

    , CASE WHEN ploc.srctimewindowtypeid IN ('KEY') THEN 'Key Drop' END AS key_drop_flag

FROM dm.fact_dailyplanneddeliverystops AS plnd_dlvry_stp

    JOIN fiscal_cal
        ON plnd_dlvry_stp.routestartdateid = fiscal_cal.dateid

    JOIN co
        ON plnd_dlvry_stp.operatingcompanyid = co.operatingcompanyid

    LEFT JOIN dm.dim_deliverystop AS dd1
        ON plnd_dlvry_stp.operatingcompanyid = dd1.operatingcompanyid
            AND plnd_dlvry_stp.deliverystopid = dd1.deliverystopid

    LEFT JOIN dm.dim_deliveryroute AS route
        ON plnd_dlvry_stp.operatingcompanyid = route.operatingcompanyid
            AND plnd_dlvry_stp.deliveryrouteid = route.deliveryrouteid

    LEFT JOIN dm.fact_dailyplanneddeliveryroutes as routes
        ON plnd_dlvry_stp.operatingcompanyid = routes.operatingcompanyid
            AND plnd_dlvry_stp.routestartdateid = routes.startdateid
            AND plnd_dlvry_stp.deliveryrouteid = routes.deliveryrouteid
            AND routes.stopcount is not null

    LEFT JOIN dw.dim_planneddeliverylocation AS dpt
        ON plnd_dlvry_stp.operatingcompanyid = dpt.operatingcompanyid
            AND routes.srclocationidorigin = dpt.srclocationid
            AND dpt.iscurrentversion = 1
            AND dpt.type = 'DPT'

    JOIN dm.fact_dailyactualdeliveryroutes AS act_routes
        ON plnd_dlvry_stp.operatingcompanyid = act_routes.operatingcompanyid
            AND plnd_dlvry_stp.deliverydaysdateid = act_routes.scheduleddateid
            AND plnd_dlvry_stp.deliveryrouteid = act_routes.deliveryrouteid

    LEFT JOIN dw.dim_planneddeliverylocation AS ploc
        ON plnd_dlvry_stp.operatingcompanyid = ploc.operatingcompanyid
            AND dd1.srcstopid = ploc.srclocationid
            AND dd1.type = ploc.type
            AND ploc.iscurrentversion = 1

    LEFT JOIN dw.dim_custmuanationalid AS cust
        ON plnd_dlvry_stp.operatingcompanyid = cust.operatingcompanyid
            AND plnd_dlvry_stp.deliverystopid = cust.deliverystopid
            AND cust.curr_rec_ind IN ('Y')

    LEFT JOIN (SELECT

            act_dlvry_stp.operatingcompanyid
            , act_dlvry_stp.deliveryrouteid
            , act_dlvry_stp.scheduleddateid
            , dd2.deliverystopid

        --      agg columns:
            , listagg(dd1.srcstopid, ',') as sub_cust_nbr_list
            , listagg(act_dlvry_stp.stopsequencenumber, ',') as stopsequencenumber_list
            , min(act_dlvry_stp.arrivaldatetime) AS arrivaldatetime
            , max(act_dlvry_stp.departuredatetime) AS departuredatetime
            , listagg(act_dlvry_stp.srcrouteid, ',') AS routeid_list

        FROM dm.fact_dailyactualdeliverystops AS act_dlvry_stp

        JOIN fiscal_cal
            ON act_dlvry_stp.scheduleddateid = fiscal_cal.dateid

        JOIN co
            ON act_dlvry_stp.operatingcompanyid = co.operatingcompanyid

        LEFT JOIN dw.dim_deliverystop AS dd1
            ON act_dlvry_stp.operatingcompanyid = dd1.operatingcompanyid
                AND act_dlvry_stp.deliverystopid = dd1.deliverystopid

        LEFT JOIN dw.dim_deliverystop AS dd2
            ON act_dlvry_stp.operatingcompanyid = dd2.operatingcompanyid
                AND (case when dd1.deliverystopid = dd1.masterdeliverystopid then dd1.deliverystopid else dd1.masterdeliverystopid end) = dd2.deliverystopid

        WHERE 1 = 1

        GROUP BY 1, 2, 3, 4) AS act_dlvry_stp
        ON plnd_dlvry_stp.operatingcompanyid = act_dlvry_stp.operatingcompanyid
            AND plnd_dlvry_stp.deliverydaysdateid = act_dlvry_stp.scheduleddateid
            AND plnd_dlvry_stp.deliveryrouteid = act_dlvry_stp.deliveryrouteid
            AND plnd_dlvry_stp.deliverystopid = act_dlvry_stp.deliverystopid

WHERE 1 = 1
    AND plnd_dlvry_stp.stoptype IN ('STP')
    AND plnd_dlvry_stp.cases is not null
    AND plnd_dlvry_stp.offdaydelivery = 0 -- Excluding off-day deliveries
    AND dd1.type = 'SIT'

