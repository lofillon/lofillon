Aggregate port statistics as output of the maritime transport model

-------
The port_locations_value file contains the value of the throughput flowing through port

id: node ID of port
name: node name of port (port name_country)
iso3: iso3 of country go port
geometry: geometry (lat/lon of port centroid)
lat: latitude of port centroid
lon: longitude of port centroid
export: export flow through port (in USD)
import: import flow through port (in USD)
trans: transhipment flow through port (in USD)
throughput: total throughput through port (import+export+trans)(in USD)

-------
The port_locations_weight file contains the weight of the throughput flowing through port

id: node ID of port
name: node name of port (port name_country)
iso3: iso3 of country go port
geometry: geometry (lat/lon of port centroid)
lat: latitude of port centroid
lon: longitude of port centroid
export: export flow through port (in tonnes)
import: import flow through port (in tonnes)
trans: transhipment flow through port (in tonnes)
throughput: total throughput through port (import+export+trans)(in tonnes)

----
The port_foreign_domestic_trade file contains the value for throughput, split out in a domestic (flows coming from or going to domestic economy) or foreign (flows coming from or going to foreign economy) part. 

id: node ID of port
name: node name of port (port name_country)
iso3: iso3 of country go port
geometry: geometry (lat/lon of port centroid)
Continent_Code: continent code of iso3
export: export flow through port (in USD)
import: import flow through port (in USD)
v_trans_all: transhipment flow through port (in USD)
throughput: total throughput through port (import+export+v_trans_all)(in USD)
v_im_dom: domestic import flow through port (in USD)
v_im_for: foreign import flow through port (in USD)
v_ex_dom: domestic export flow through port (in USD)
v_ex_for: foreign export flow through port (in USD)
v_trans_dom: domestic transhipment flow through port (in USD)
v_trans_for: foreign transhipment flow through port (in USD)
ratio_for_import: fraction of imports being foreign (-)
ratio_for_export: fraction of exports being foreign (-)
ratio_for_throughput: fraction of throughput being foreign (-)
throughput_foreign: value of foreign throughput
-------

