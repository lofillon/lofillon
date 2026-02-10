# ComplexNetworksAnalysis\_SCRM

Ce dépôt contient le rapport (`paper.tex` / `paper.pdf`) et les scripts Python utilisés pour comparer des métriques de criticité (centralités classiques vs métriques basées sur la distance de résistance) dans un cadre SCRM, sur graphes jouets et sur un cas empirique portuaire.

## Données (dataset empirique)

Le fichier de données brut utilisé pour construire le graphe empirique est volumineux et n'est **pas versionné** dans GitHub (limite 100 MB).

- **Dataset** : *Global port supply-chains* (Mendeley Data), version 1 (2022)
- **DOI** : `10.17632/vzzy3b9gg4.1`
- **Lien** : `https://data.mendeley.com/datasets/vzzy3b9gg4/1`
- **Licence** : CC BY 4.0 (voir la page du dataset)

### Où placer le fichier localement

Après téléchargement, placer le fichier :

- `Global port supply-chains/Port_to_port_network/port_trade_network.csv`

Le script `run_ports_tables.py` s'attend à trouver ce chemin (ou accepte un chemin fourni en argument CLI).

## Exécution rapide

- **Réseau réel (ports)** :

```bash
python3 run_ports_tables.py --csv "Global port supply-chains/Port_to_port_network/port_trade_network.csv"
```

- **Graphes jouets (pipeline aligné réel)** :

```bash
python3 run_toys_tables.py
```


