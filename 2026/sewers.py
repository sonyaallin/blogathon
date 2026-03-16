import geopandas as gpd
import pandas as pd
import folium
from folium.features import DivIcon
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
import contextily as ctx
import os

if __name__ == "__main__":

    use_folium = False  # True = Folium interactive map, False = Matplotlib static map

    # --------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------

    wards = gpd.read_file("data/City_Wards_Data.geojson")
    flooding = pd.read_csv("data/floodings.csv")
    ecoli = gpd.read_file("data/toronto-beaches-water-quality - 4326.geojson")

    manholes = gpd.read_file("data/Sewer Manholes - 4326/Sewer Manholes - 4326.shp")
    sewer_mains = gpd.read_file("data/Sewer Pressurized Main - 4326/Sewer Pressurized Main - 4326.shp")

    # Match CRS
    manholes = manholes.to_crs(wards.crs)
    sewer_mains = sewer_mains.to_crs(wards.crs)

    # Print dataset columns
    print(ecoli.columns)
    print(flooding.columns)
    print(wards.columns)

    # --------------------------------------------------
    # SPATIAL ANALYSIS
    # --------------------------------------------------

    # Count manholes per ward
    manholes_in_wards = gpd.sjoin(manholes, wards, how="left", predicate="within")
    manhole_counts = (
        manholes_in_wards.groupby("_id")
        .size()
        .reset_index(name="manhole_count")
    )
    print(manhole_counts)

    # Count pressurized sewer mains per ward
    mains_in_wards = gpd.sjoin(sewer_mains, wards, how="left", predicate="within")
    mains_counts = (
        mains_in_wards.groupby("_id")
        .size()
        .reset_index(name="main_count")
    )
    print(mains_counts)

    # --------------------------------------------------
    # DATA MERGING
    # --------------------------------------------------

    merged = wards.merge(flooding, left_on="_id", right_on="Ward")
    merged = merged.merge(mains_counts, on="_id", how="left")
    merged = merged.merge(manhole_counts, on="_id", how="left")

    merged["manhole_count"] = merged["manhole_count"].fillna(0).astype(int)
    merged["main_count"] = merged["main_count"].fillna(0).astype(int)

    # --------------------------------------------------
    # CRS FOR VISUALIZATION
    # --------------------------------------------------

    if use_folium:
        merged = merged.to_crs(epsg=4326)
        manholes = manholes.to_crs(epsg=4326)
        sewer_mains = sewer_mains.to_crs(epsg=4326)
        ecoli = ecoli.to_crs(epsg=4326)
    else:
        merged = merged.to_crs(epsg=3857)
        manholes = manholes.to_crs(epsg=3857)
        sewer_mains = sewer_mains.to_crs(epsg=3857)
        ecoli = ecoli.to_crs(epsg=3857)

    # ==================================================
    # FOLIUM INTERACTIVE MAP
    # ==================================================

    if use_folium:

        TOLERANCE = 0.0001

        merged_display = merged[["_id", "Total", "main_count", "manhole_count", "geometry"]].copy()
        merged_display["geometry"] = merged_display.geometry.simplify(TOLERANCE)

        sewer_display = sewer_mains[["geometry"]].copy()
        sewer_display["geometry"] = sewer_display.geometry.simplify(TOLERANCE)

        # Map center (no deprecation warning)
        center = merged_display.geometry.union_all().centroid

        m = folium.Map(
            location=[center.y, center.x],
            zoom_start=11,
            tiles="CartoDB positron"
        )

        # Choropleth of flooding incidents
        folium.Choropleth(
            geo_data=merged_display,
            data=merged_display,
            columns=["_id", "Total"],
            key_on="feature.properties._id",
            fill_color="OrRd",
            fill_opacity=0.7,
            line_opacity=0.4,
            legend_name="Flooding Incidents by Ward",
        ).add_to(m)

        # Sewer mains layer
        folium.GeoJson(
            sewer_display,
            name="Sewer Mains",
            style_function=lambda x: {
                "color": "blue",
                "weight": 1,
                "opacity": 0.4,
            }
        ).add_to(m)

        # --------------------------------------------------
        # Compute centroids safely (projected CRS)
        # --------------------------------------------------

        centroids = merged_display.to_crs(3857).geometry.centroid
        centroids = gpd.GeoSeries(centroids, crs=3857).to_crs(4326)

        merged_display["centroid"] = centroids

        # Ward labels
        for _, row in merged_display.iterrows():

            centroid = row.centroid
            label = int(row["main_count"])

            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=DivIcon(
                    html=f'<div style="font-size:10pt;font-weight:bold;color:black">{label}</div>'
                )
            ).add_to(m)

        # --------------------------------------------------
        # E. coli sampling sites
        # --------------------------------------------------

        threshold = 200
        ecoli_concerning = ecoli[ecoli["eColi"] > threshold]

        for _, row in ecoli_concerning.iterrows():

            geom = row.geometry

            if geom.geom_type == "Point":
                points = [geom]
            elif geom.geom_type == "MultiPoint":
                points = list(geom.geoms)
            else:
                continue

            for point in points:
                folium.CircleMarker(
                    location=[point.y, point.x],
                    radius=6,
                    color="purple",
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"E. coli: {row['eColi']}"
                ).add_to(m)

        m.save("interactive_map.html")
        print("Map saved as interactive_map.html")

    # ==================================================
    # MATPLOTLIB STATIC MAP
    # ==================================================

    else:

        fig, ax = plt.subplots(figsize=(14, 12))

        merged.plot(
            column="Total",
            ax=ax,
            cmap="OrRd",
            legend=True,
            legend_kwds={"label": "Flooding Incidents by Ward"},
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7
        )

        sewer_mains.plot(
            ax=ax,
            color="blue",
            linewidth=0.5,
            alpha=0.3
        )

        for _, row in merged.iterrows():

            centroid = row.geometry.centroid

            ax.annotate(
                text=str(row["main_count"]),
                xy=(centroid.x, centroid.y),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                path_effects=[path_effects.withStroke(linewidth=2, foreground="white")]
            )

        ecoli.plot(
            ax=ax,
            column="eColi",
            cmap="cool",
            markersize=50,
            alpha=0.8,
            legend=True,
            legend_kwds={"label": "E. coli Count"},
        )

        os.environ["USER_AGENT"] = "my-geopandas-map-script"
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

        ax.set_title("Basement Flooding Incidents and Sewer Main Counts by Ward")

        plt.show()