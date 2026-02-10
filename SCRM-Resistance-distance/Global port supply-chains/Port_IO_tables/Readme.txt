Output of the connection between the link between the transport model and the multi-regional input-output tables (MRIO). 

-------
The port import coefficient estimates the maritime imports needed to meet the final demand in an economy, either via exports or domestic consumption.

id: node ID of port
name: node name of port (port name_country)
iso3: iso3 of country go port
Continent_Code: continent code of port
Import_export: maritime imports through port needed for exports
Total_export: total exports in domestic economy 
Import_demand: maritime imports through port needed for domestic demand
Total_demand: total domestic demand in domestic economy
import_req_dom: Import_export + Import_demand
import_multiplier: import_req_dom/(Total_export+Total_demand)
import_multiplier_scale: import_multiplier*1000 (imports per 1000 USD increase in final demand)

-------
The port output coefficient estimates the industry output and final consumption which is linked to the trade-flows going through a port. 

id: node ID of port
name: node name of port (port name_country)
iso3: iso3 of country go port
multiplier: total economic multiplier (total economic value influenced / trade flow)
multiplier_dom: domestic economic multiplier (domestic economic value influenced / trade flow)
multiplier_row: rest of world economic multiplier (rest of world economic value influenced / trade flow)
frac_bw_fw: fraction of backward and forward economic value influenced (-)
frac_row_dom: fraction of rest of world and domestic economic value influenced (-)
C: total value of direct consumption influenced
Dind_total: total value of industry output influenced
D_economic_total: total economic value influenced (C + Dind_total)
dom_ind_share: share of domestic industry output dependent on port
glob_ind_share: share of global industry output dependent on port
dom_C_com_share: share of domestic final consumption dependent on port
glob_C_com_share: share of global final consumption dependent on port


