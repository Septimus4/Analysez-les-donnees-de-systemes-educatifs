# 📊 Online Education Market Insights

This project analyzes global education, demographic, and economic indicators to identify high-potential countries for expanding online post-secondary education.

It uses World Bank EdStats and geospatial data to:
- Filter and rank countries by potential
- Visualize data coverage and indicator correlations
- Generate insightful plots and maps

---

## 📥 Required Datasets

Before running the analysis, download and place the following datasets in the project root:

### 1. World Bank EdStats

Download from: [https://datatopics.worldbank.org/education/](https://datatopics.worldbank.org/education/)

Place the following files in `./Projet+Python_Dataset_Edstats_csv/`:

* `EdStatsData.csv`
* `EdStatsCountry.csv`
* `EdStatsSeries.csv`

### 2. Natural Earth World Shapefile

Download from: [https://www.naturalearthdata.com/](https://www.naturalearthdata.com/)

Place the shapefile components in `./ne_110m_admin_0_countries/`:

* `ne_110m_admin_0_countries.shp`
* `ne_110m_admin_0_countries.dbf`, `.shx`, `.prj`, etc.

---

## 📊 Visual Outputs

The analysis generates:

* Top 20 country rankings
* Score correlation heatmap
* Geospatial potential map
* Score breakdown vs. raw indicator values
