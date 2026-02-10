The freight statistics and port-to-port trade network, which is the output of the maritime transport model


-----
The file freight_cost_aggregate contains freight flows between two countries per sector, including the share of that being allocated on the maritime transport and hinterland transport networks. 

iso3_O: origin iso3
iso3_D: destination iso3
q_sea_flow: trade flow in weight terms (tonnes)
v_sea_flow: trade flow in value terms (x1000 USD)
cost_freight_total: total freight costs in USD
tonnes_km: freight-km of trade flow on both maritime and hinterland transport network (tonnes x km)
tonnes_km: freight-km of trade flow on maritime transport network (tonnes x km)
share_mar: stare of trade flow being maritime between countries
Industries: sector classification (1-11) according to EORA-MRIO tables. 

-----
The port_trade_network file contains the allocated freight flows through ports, including the origin and destination countries of the trade flow. 

iso3_O: origin iso3
iso3_D: destination iso3
flow: import, export of transhipment
q_sea_flow: trade flow through port in weight terms (tonnes)
v_sea_flow: trade flow through port in value terms (x1000 USD)
q_sea_flow_sum: trade flow between origin and destination country in weight terms (tonnes)
v_sea_flow_sum: trade flow between origin and destination country in value terms (x1000 USD)
Industries: sector classification (1-11) according to EORA-MRIO tables. 
share_mar: stare of trade flow being maritime between countries
q_share_port: share of maritime trade flow through port in weight terms (tonnes)
v_share_port: share of maritime trade flow through port in value terms (x1,000 USD)
q_trade_port: share of total trade flow (all modes) through port in weight terms (tonnes)
v_trade_port: share of total trade flow (all modes) through port in value terms (x1,000 USD)
-----
