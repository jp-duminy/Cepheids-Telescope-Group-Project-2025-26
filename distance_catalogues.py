"""

Cepheid distance values with corresponding primary sources.

It is most sensible to use parallax measurements as then we are calibrating the distance ladder, like Henrietta Swan Leavitt.

"""

# Gaia DR3 Tags

# 01 MW Cyg: 2055014277739104896
# 02 V520 Cyg: 2166861170366710272
# 03 VX Cyg: 1873250780732545920
# 04 VY Cyg: 1873112207907294848
# 05 CP Cep: 2198162651499971200
# 06 Z Lac: 2007201567928631296
# 07 V Lac: 2003938801532007808
# 08 SW Cas: 2010285491880986112
# 09 TU Cas: 394818721274314112
# 10 DL Cas: 428620663657823232
# 11 V636 Cas: 512524361613040640




cepheid_distances = {
    "01": {"name": "MW Cyg", "distance": 1466},
    "02": {"name": "V520 Cyg", "distance":1849 },
    "03": {"name": "VX Cyg", "distance": 3322},
    "04": {"name": "VY Cyg", "distance": 1881 },
    "05": {"name": "CP Cep", "distance": 4604},
    "06": {"name": "Z Lac", "distance": 1882 },
    "07": {"name": "V Lac", "distance": 1631 },
    "08": {"name": "SW Cas", "distance": 2008 },
    "09": {"name": "TU Cas", "distance": 808 },
    "10": {"name": "DL Cas", "distance": 1688},
    "11": {"name": "V636 Cas", "distance": 605 },
}

cepheid_simbad_distances = {
    "01": {"name": "MW Cyg", "distance": 1949.318}, # parallax: Gaia DR3 (2020)
    "02": {"name": "V520 Cyg", "distance":2439.02}, # parallax: Gaia DR3 (2020)
    "03": {"name": "VX Cyg", "distance": 3474.635}, # parallax: Gaia DR3 (2020)
    "04": {"name": "VY Cyg", "distance": 2187.23 }, # parallax: Gaia DR3 (2020)
    "05": {"name": "CP Cep", "distance": 3946.330}, # parallax: Gaia DR3 (2020)
    "06": {"name": "Z Lac", "distance": 2055.076 }, # parallax: Gaia DR3 (2020)
    "07": {"name": "V Lac", "distance": 2116.4 }, # parallax: Gaia DR3 (2020)
    "08": {"name": "SW Cas", "distance": 2292.001}, # parallax: Gaia DR3 (2020)
    "09": {"name": "TU Cas", "distance": 1008.88 }, # parallax: Gaia DR3 (2020)
    "10": {"name": "DL Cas", "distance": 1808.97}, # parallax: Gaia DR3 (2020)
    "11": {"name": "V636 Cas", "distance": 743.55 }, # parallax: Gaia DR3 (2020)
}

# median of photogeometric distance posterior
# this is calibrated from Gaia DR3 (Biller-Jones et al. (may have misspelt))

cepheid_vizier_distances = { 
    "01": {"name": "MW Cyg", "distance": 1856.94629000},
    "02": {"name": "V520 Cyg", "distance": 2272.18042000 },
    "03": {"name": "VX Cyg", "distance": 3114.65356000},
    "04": {"name": "VY Cyg", "distance": 2080.08105000},
    "05": {"name": "CP Cep", "distance": 3641.31836000},
    "06": {"name": "Z Lac", "distance": 1972.44800000 },
    "07": {"name": "V Lac", "distance": 2022.23596000},
    "08": {"name": "SW Cas", "distance": 2167.97705000},
    "09": {"name": "TU Cas", "distance": 977.19494600},
    "10": {"name": "DL Cas", "distance": 1724.81311000},
    "11": {"name": "V636 Cas", "distance": 733.21038800},
}