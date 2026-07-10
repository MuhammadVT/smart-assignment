WITH eval_period AS (
    select distinct fisc_mo_id, fisc_mo_skey
    from edwp.cal_day_dim
    where fisc_mo_end_dt <= current_date and
    fisc_mo_skey > (select max(fisc_mo_skey) from edwp.cal_day_dim where fisc_mo_end_dt <= current_date) - 12
)
select distinct
    co.co_nbr || '-' || cust.cust_nbr as co_cust_nbr,
    -- co.co_nbr, cust.cust_nbr,
--     case when (UPPER(sfdc.sysco_your_way_desc) = 'ENROLLED') then 'Y' else 'N' end as sysco_your_way_ind,
--     case when (sfdc.sysco_perks_sts_desc in ('Enrolled', 'Enrolled free trial')
--         and sfdc.perks_enrl_dt <= CURRENT_DATE
--         and (sfdc.perks_expir_dt > CURRENT_DATE or sfdc.perks_expir_dt is null)) then 1
--     else 0 end as perks_enrolled
    case when (sfdc.sysco_perks_sts_desc in ('Enrolled', 'Enrolled free trial')
        and sfdc.perks_enrl_dt <= CURRENT_DATE
        and (sfdc.perks_expir_dt > CURRENT_DATE or sfdc.perks_expir_dt is null)) then 'Perks'
    else 'Non-Perks' end as cust_tier
from  edwp.cust_ship_to_dim as cust
join edwp.org_co_dim AS co ON co.co_skey = cust.co_skey
    AND co.bus_unit_nm = 'USBL'
    AND co.mkt_lvl <> ''
    AND co.sts_ind = 'A'
JOIN edwp.sale_oblig_dtl_fact AS oblig  ON cust.cust_skey = oblig.cust_skey
JOIN edwp.cal_day_dim as cal on cal.day_dt = oblig.oblig_dt
JOIN eval_period as eval on eval.fisc_mo_id = cal.fisc_mo_id
join edwp.sfdc_account_dim as sfdc on sfdc.acct_id = (co.co_nbr||'-'||cust.cust_nbr)
        and sfdc.curr_rec_ind = 'Y'
where cust.curr_rec_ind = 'Y'
    and cust.src_sys_cd = 'SUS'
    and cust.acct_typ_cd IN ('TRS', 'LCC', 'CMU', 'OTH')
    and co.co_skey in (67)  -- {OPCO}  -- TODO: make this a parameter
    and cust_tier = 'Perks'

UNION

select  LEFT(css.co_cust_nbr, 3) || '-' || css.cust_nbr as co_cust_nbr
    , css.tier as cust_tier
from s_eat_cust_seg.cust_lf_cyc_seg_aggr_fact_vw css
where 1 = 1
    and css.tier in ('4', '5')
    and css.co_skey IN ('067')-- {OPCO}  -- Make this a parameter

-- limit 10;