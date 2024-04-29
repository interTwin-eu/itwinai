import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import matplotlib.colors as colors


def plot_sampler(da_bkg, meta, meta_valid):
    
    vv = da_bkg
    
    static_meta = meta.get("static")
    static_meta_valid = meta_valid.get("static")
    
    vv = vv.assign_coords({"gridcell":(("lat", "lon"), static_meta.idx_orig_2d)})
    
    vv = vv.assign_coords({"gridcell_valid":(("lat", "lon"), static_meta_valid.idx_orig_2d)})
    
    tmp = np.zeros(vv.shape).astype(np.bool_)
    for i in static_meta.idx_sampled_1d:
        tmp[vv.gridcell == i] = True
    
    tmp_valid = np.zeros(vv.shape).astype(np.bool_)
    for i in static_meta_valid.idx_sampled_1d:
        tmp_valid[vv.gridcell_valid == i] = True
    
    df = vv.where(tmp).to_dataframe().dropna().reset_index()
    
    df_valid = vv.where(tmp_valid).to_dataframe().dropna().reset_index()
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.lon, y=df.lat), crs=4326)
    
    gdf_valid = gpd.GeoDataFrame(df_valid, geometry=gpd.points_from_xy(x=df_valid.lon, y=df_valid.lat), crs=4326)
    
    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    da_bkg.plot(ax = ax, add_colorbar=False, alpha = 0.5, cmap="terrain")
    gdf.plot(ax=ax, color="red", markersize=10, label="training")
    gdf_valid.plot(ax=ax, color="black", markersize=10, label = "validation")
    plt.legend()
    #ax.set_xlim([6, 7.5])
    #ax.set_ylim([45.5, 46.5])
    
    
    return fig,ax


def compute_pbias(y: xr.DataArray, yhat, dim="time"):
    if isinstance(y, xr.DataArray) or isinstance(yhat, xr.DataArray):
        return 100*(( yhat - y).sum(dim=dim, skipna=False) / y.sum(dim=dim,skipna=False))
    else:
        return 100* np.sum(yhat -y, axis=2) / np.sum(y, axis=2)    

def compute_bias(y: xr.DataArray, yhat, dim="time"):
    if isinstance(y, xr.DataArray) or isinstance(yhat, xr.DataArray):
        return ( yhat - y).sum(dim=dim, skipna=False)
    else:
        return np.sum(yhat - y, axis=2)    


def map_pearson(y: xr.DataArray, yhat, dim="time"):
    p = xr.corr(y, yhat, dim=dim)
    fig, ax = plt.subplots(1,1)
    i = ax.imshow(p, cmap="RdBu", norm=colors.CenteredNorm())
    fig.colorbar(i, ax=ax)

def map_pbias(y: xr.DataArray, yhat, dim="time"):
    pbias = compute_pbias(y, yhat, dim)
    fig, ax = plt.subplots(1,1)
    i = ax.imshow(pbias, cmap="RdBu",  norm=colors.CenteredNorm())
    fig.colorbar(i, ax=ax)

def map_bias(y: xr.DataArray, yhat, dim ="time" ):
    bias = compute_bias(y, yhat, dim)
    fig, ax = plt.subplots(1,1)
    i = ax.imshow(bias, cmap="RdBu",  norm=colors.CenteredNorm())
    fig.colorbar(i, ax=ax)

def map_at_timesteps(y: xr.DataArray, yhat: xr.DataArray, dates = None):
    ts = dates if dates else y.time.dt.date.values 
    for t in dates:
        fig, ax = plt.subplots(1,2, figsize= (20,15))
        fig.subplots_adjust(hspace=0.3)
        l1 = ax[0].imshow(yhat.sel(time=t))
        ax[0].set_title("LSTM", fontsize=28)
        fig.colorbar(l1, ax=ax[0],shrink=0.5)
        l2 = ax[1].imshow(y.sel(time=t))
        ax[1].set_title("wflow", fontsize=28)
        fig.colorbar(l2, ax=ax[1],shrink=0.5)
        fig.suptitle(t, y = 0.8, fontsize=20, fontweight="bold")
        fig.tight_layout()
        
        
        
def ts_compare(y: xr.DataArray, yhat, lat= [], lon = []):
    for ilat,ilon in zip(lat, lon):
        ax_dict = plt.figure(layout="constrained", figsize=(20,6)).subplot_mosaic(
        """
        AC
        BC
        """,
        width_ratios=[4, 1]
        )
        iy = y.sel(lat = ilat,lon = ilon, method="nearest")
        iyhat = yhat.sel(lat = ilat,lon = ilon, method="nearest") 
        ax_dict["A"].plot(iyhat, label ="lstm")
        ax_dict["A"].plot(iy, label="wflow")
        ax_dict["A"].legend()
        ax_dict["B"].scatter(iy,iyhat, s=1)
        xmin = np.nanmin( np.concatenate([iy, iyhat] )) - 0.05
        xmax = np.nanmax( np.concatenate([iy, iyhat] )) + 0.05
        ax_dict["B"].set_xlim(xmin, xmax)
        ax_dict["B"].set_ylim(xmin, xmax)
        ax_dict["B"].axline((0, 0), (1, 1), color="black", linestyle="dashed")
        ax_dict["B"].set_ylabel("lstm")
        ax_dict["B"].set_xlabel("wflow")
        df = gpd.GeoDataFrame([],geometry=gpd.points_from_xy(x=[ilon], y=[ilat]))
        y.mean("time").plot(ax=ax_dict["C"], add_colorbar=False)
        df.plot(ax=ax_dict["C"], markersize=20, color="red")
        plt.title(f"lat, lon:  ({ ilat}, {ilon})")