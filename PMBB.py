"""
Simple CSV data explorer for PMBB imaging data analysis to better understand
clinical AI implementation.

Author(s):
    Allison Chae
    Michael S Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import argparse
from collections import defaultdict
import inspect
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence, Union


class PMBBImagingExplorer:
    def __init__(
        self,
        PACS_data: Union[Path, str],
        cache_dir: Optional[Union[Path, str]] = "./cache",
        font_size: int = 18,
        use_sans_serif: bool = True
    ):
        """
        Args:
            PACS_data: filepath to encoded PACS CSV file on PMBB imaging data.
            cache_dir: optional directory path for caching processed data.
                If None, no caching is performed.
            font_size: default font size. Default 18.
            use_sans_serif: whether to use Sans Serif font style.
        """
        self.mod_key = "Mod"
        self.date_key = "ProcedureDTTM"
        self.pid_key = "x2"
        self.size_key = "StudySz"

        self.cache = cache_dir
        if self.cache is not None and not os.path.isdir(self.cache):
            os.mkdir(self.cache)
        self.font_size = font_size
        self.use_sans_serif = use_sans_serif
        self.plot_config()
        self.PACS_data = PACS_data
        self.data = pd.read_csv(self.PACS_data, converters={
            "OrderNumber": str,
            self.date_key: str
        })
        self.data.dropna(inplace=True)
        self.data.drop(columns=["Unnamed: 0", "ImgCt"], inplace=True)
        # Convert study dates to datetime objects.
        self.data[self.date_key] = pd.to_datetime(
            self.data[self.date_key], format="%m/%d/%y %I:%M %p"
        )
        # Convert study sizes to integer number of bytes.
        self.data[self.size_key] = self.data[self.size_key].astype(int)
        # Extract the anonymized patient ID's.
        self.patients = set((self.data[self.pid_key].astype(int)))
        # Extract the unique imaging modalities.
        self.modalities = set(self.data[self.mod_key].str.upper())
        self.modalities = self.modalities & set(self.DICOM_modalities())

    def plot_config(self) -> None:
        """
        Plot configuration variables.
        Input:
            None.
        Returns:
            None.
        """
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        if self.use_sans_serif:
            matplotlib.rcParams['font.family'] = "Arial"
        else:
            matplotlib.rcParams["font.family"] = "STIXGeneral"
        matplotlib.rcParams.update({"font.size": self.font_size})
        self.colors = [
            "#1A476F",
            "#90353B",
            "#55752F",
            "#E37E00",
            "#6E8E84",
            "#C10534",
            "#938DD2",
            "#CAC27E",
            "#A0522D",
            "#7B92A8",
            "#2D6D66",
            "#9C8847",
            "#BFA19C",
            "#FFD200",
            "#89969B"
        ]

    def DICOM_modalities(self) -> Sequence[str]:
        """
        Returns a list of DICOM modality abbreviations.
        Input:
            None.
        Returns:
            A list of DICOM modality abbreviations.
        """
        return [
            "XR", "CT", "MR", "NM", "US", "BI", "ES", "PT", "PET", "XA", "RF",
            "TG", "DX", "MG", "IO", "PX", "GM", "SM", "XC", "ECG", "OP", "OPT"
        ]

    def find_cache(self, name: str) -> Optional[Any]:
        """
        Finds the .pkl file with the specified name within the cache directory.
        Returns the deserialized object if found, otherwise returns None.
        Input:
            name: name of the .pkl file in the cache directory to query.
        Returns:
            Data within the pickle file (if found).
        """
        name = name.lower()
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        data = None
        cache_file = os.path.join(self.cache, name)
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
        return data

    def save_cache(self, name: str, data: Any) -> None:
        """
        Saves the input data to the .pkl file with the specified name within
        the cache directory.
        Input:
            name: name of the .pkl file in the cache directory to save to.
            data: data to serialize and save to cache.
        Returns:
            None.
        """
        name = name.lower()
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        cache_file = os.path.join(self.cache, name)
        with open(cache_file, "w+b") as f:
            pickle.dump(data, f)
        return

    def repeat_studies_by_modality(
        self,
        savepath: Optional[Union[Path, str]] = None,
        do_plot: bool = True,
        max_modalities: int = 10,
        max_repeats: int = 10,
        label: Optional[str] = None
    ) -> Dict[str, Dict[int, int]]:
        """
        Calculates and plots the number of repeat imaging studies on a per
        patient, per imaging modality basis.
        Input:
            savepath: optional filepath to save the plot to. Default not saved.
            do_plot: whether or not to plot the data. Default True.
            max_modalities: maximum number of modalities to plot. Default 10.
                If -1, then all modalities are plotted.
            max_repeats: maximum number of repeat imaging studies to plot.
                Default 10.
            label: optional label for the plot. Default None.
        Returns:
            A dictionary mapping imaging modality to another dictionary mapping
            number of repeats to frequency.
        """
        modality_to_counts_to_freqs = {}
        used_cache = False
        current_func_name = inspect.stack()[0][3]
        if self.cache is not None:
            modality_to_counts_to_freqs = self.find_cache(current_func_name)
            used_cache = modality_to_counts_to_freqs is not None
            if not used_cache:
                modality_to_counts_to_freqs = {}
        if not used_cache:
            for mod in tqdm(
                self.modalities, desc="Repeat Studies by Modality"
            ):
                modality_data = list(
                    self.data[self.data[self.mod_key] == mod][self.pid_key]
                )
                counts = [
                    modality_data.count(_id) for _id in self.patients
                ]
                counts_to_freqs = defaultdict(int)
                for count in counts:
                    counts_to_freqs[count] += 1
                modality_to_counts_to_freqs[mod] = counts_to_freqs

        if do_plot:
            modality_volumes = self.volume_by_modality(
                do_plot=False, max_modalities=max_modalities
            )
            plt.figure(figsize=(10, 8))
            for (mod, volume), c in zip(modality_volumes.items(), self.colors):
                if mod not in modality_to_counts_to_freqs:
                    continue
                counts = sorted(list(modality_to_counts_to_freqs[mod].keys()))
                freqs = [modality_to_counts_to_freqs[mod][c] for c in counts]
                cdf = np.flip(np.cumsum(np.flip(np.array(freqs))))
                if len(cdf) <= 1:
                    continue
                if len(counts) > max_repeats:
                    counts = counts[:max_repeats + 1]
                    cdf = cdf[:max_repeats + 1]
                plt.plot(
                    counts[1:],
                    cdf[1:] / cdf[1],
                    color=c,
                    label=mod,
                    linewidth=3
                )
            plt.legend(loc="upper right")
            plt.xlabel("Number of Imaging Studies")
            plt.ylabel("1 - CDF")
            if label is not None and len(label) > 0:
                plt.annotate(
                    label,
                    xy=(0.0, 1.025),
                    xycoords="axes fraction",
                    fontsize=24,
                    weight="bold"
                )
            if savepath is None:
                plt.show()
            else:
                plt.savefig(
                    savepath,
                    dpi=600,
                    transparent=True,
                    bbox_inches="tight"
                )
            plt.close()

        if not used_cache and self.cache is not None:
            self.save_cache(current_func_name, modality_to_counts_to_freqs)

        return modality_to_counts_to_freqs

    def volume_by_modality(
        self,
        savepath: Optional[Union[Path, str]] = None,
        do_plot: bool = True,
        max_modalities: int = -1,
        label: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Calculates and plots the total volume of imaging studies stratified
        by imaging modality.
        Input:
            savepath: optional filepath to save the plot to. Default not saved.
            do_plot: whether or not to plot the data. Default True.
            max_modalities: maximum number of modalities to plot. Default 10.
                If -1, then all modalities are plotted.
            label: optional label for the plot. Default None.
        Returns:
            A dictionary mapping imaging modality to volume. Note that if
            `max_modalities` is specified, then the number of returned key-
            value pairs is bounded from above by `max_modalities`.
        """
        modality_to_volume = {}
        used_cache = False
        current_func_name = inspect.stack()[0][3]
        if self.cache is not None:
            modality_to_volume = self.find_cache(current_func_name)
            used_cache = modality_to_volume is not None
            if not used_cache:
                modality_to_volume = {}
        if not used_cache:
            for mod in tqdm(self.modalities, desc="Volume by Modality"):
                volume, _ = self.data[self.data[self.mod_key] == mod].shape
                modality_to_volume[mod] = volume

        if do_plot:
            plt.figure(figsize=(10, 8))
            keys = list(modality_to_volume.keys())
            keys = sorted(
                keys, key=lambda k: modality_to_volume[k], reverse=True
            )
            if max_modalities > -1 and max_modalities < len(keys):
                keys = keys[:max_modalities]
            counts = [modality_to_volume[k] / len(self.patients) for k in keys]
            plt.bar(
                keys,
                counts,
                color=self.colors[:len(keys)],
                edgecolor="black",
                linewidth=3
            )
            plt.xlabel("Imaging Study")
            plt.ylabel("Number of Studies per PMBB Capita")
            left_ax = plt.gca()
            right_ax = left_ax.twinx()
            mn, mx = left_ax.get_ylim()
            right_ax.set_ylim(
                mn * len(self.patients) / 1e3, mx * len(self.patients) / 1e3
            )
            right_ax.set_ylabel("Total Number of PMBB Studies (Thousands)")
            if label is not None and len(label) > 0:
                plt.annotate(
                    label,
                    xy=(0.0, 1.025),
                    xycoords="axes fraction",
                    fontsize=24,
                    weight="bold"
                )
            if savepath is None:
                plt.show()
            else:
                plt.savefig(
                    savepath,
                    dpi=600,
                    transparent=True,
                    bbox_inches="tight"
                )
            plt.close()

        if not used_cache and self.cache is not None:
            self.save_cache(current_func_name, modality_to_volume)

        keys = list(modality_to_volume.keys())
        keys = sorted(
            keys, key=lambda k: modality_to_volume[k], reverse=True
        )
        if max_modalities > -1 and max_modalities < len(keys):
            keys = keys[:max_modalities]
            modality_to_volume = {
                k: modality_to_volume[k] for k in keys
            }
        return modality_to_volume

    def delta_time_by_patient_modality(
        self,
        savepath: Optional[Union[Path, str]] = None,
        do_plot: bool = True,
        max_modalities: int = -1
    ) -> Dict[str, Dict[int, Sequence[int]]]:
        """
        Calculates and plots the time between sequential imaging studies by
        patient and by modality.
        Input:
            savepath: optional filepath to save the plot to. Default not saved.
            do_plot: whether or not to plot the data. Default True.
            max_modalities: maximum number of modalities to plot. Default -1.
                If -1, then all modalities are plotted.
        Returns:
            A dictionary mapping imaging modality to another dictionary mapping
            patient anonymized id to a list of delta time (in days) between
            imaging modalities. Note that all modalities are returned
            independent of the `max_modalities` input argument.
        """
        modality_to_patient_to_deltatimes = {}
        used_cache = False
        current_func_name = inspect.stack()[0][3]
        if self.cache is not None:
            modality_to_patient_to_deltatimes = self.find_cache(
                current_func_name
            )
            used_cache = modality_to_patient_to_deltatimes is not None
            if not used_cache:
                modality_to_patient_to_deltatimes = {}
        if not used_cache:
            for mod in tqdm(self.modalities, desc="Delta Times by Modality"):
                modality_data = self.data[self.data[self.mod_key] == mod]
                patient_to_deltatimes = {}
                for _id in self.patients:
                    patient_modality_data = modality_data[
                        modality_data[self.pid_key] == _id
                    ]
                    patient_modality_data = np.array(
                        sorted(
                            list(patient_modality_data[self.date_key])
                        )
                    )
                    patient_to_deltatimes[_id] = (
                        patient_modality_data[1:] - patient_modality_data[:-1]
                    )
                modality_to_patient_to_deltatimes[mod] = patient_to_deltatimes

        if do_plot:
            pass  # TODO: Still need to implement and figure out how to plot.

        if not used_cache and self.cache is not None:
            self.save_cache(
                current_func_name, modality_to_patient_to_deltatimes
            )

        return modality_to_patient_to_deltatimes

    def delta_time_dist_by_modality(
        self,
        savepath: Optional[Union[Path, str]] = None,
        do_plot: bool = True,
        max_modalities: int = 10,
        max_dt: int = 365,
        label: Optional[str] = None
    ) -> Dict[str, Sequence[int]]:
        """
        Calculates and plots the distribution of delta times between
        sequential imaging studies by modality.
        Input:
            savepath: optional filepath to save the plot to. Default not saved.
            do_plot: whether or not to plot the data. Default True.
            max_modalities: maximum number of modalities to plot. Default 10.
                If -1, then all modalities are plotted.
            max_dt: maximum range of delta times to plot. Default 365 days.
                If -1, then the entire range is plotted.
            label: optional label for the plot. Default None.
        Returns:
            A dictionary mapping imaging modality to all the delta times (in
            days) between imaging modalities by patient. Note that if
            `max_modalities` is specified, then the number of returned key-
            value pairs is bounded from above by `max_modalities`.
        """
        delta_times = self.delta_time_by_patient_modality(do_plot=False)
        delta_times = {
            mod: [
                [int(t.days) for t in dts] for _id, dts in values.items()
            ]
            for mod, values in delta_times.items()
        }
        delta_times = {
            mod: np.array(list(itertools.chain.from_iterable(dts)))
            for mod, dts in delta_times.items()
        }

        if do_plot:
            volume = self.volume_by_modality(
                do_plot=False, max_modalities=max_modalities
            )
            keys = list(volume.keys())
            if max_modalities > -1 and max_modalities < len(keys):
                keys = sorted(
                    keys, key=lambda k: volume[k], reverse=True
                )
                keys = keys[:max_modalities]
            delta_times = {
                k: delta_times[k] for k in keys[::-1]
            }

            plt.figure(figsize=(10, 8))
            for (mod, dts), co in zip(delta_times.items(), self.colors):
                if len(dts) == 0:
                    continue
                plt.hist(
                    dts,
                    bins=100,
                    weights=np.ones_like(dts) / len(dts),
                    range=[0, max_dt % len(dts)],
                    label=mod,
                    facecolor=co,
                    edgecolor="black",
                    alpha=0.75,
                    linewidth=1,
                )
            plt.legend(loc="upper right")
            plt.xlabel(
                r"$\Delta$Time Between Sequential Imaging Studies (Days)"
            )
            plt.ylabel("Frequency")
            plt.yscale("log")
            if label is not None and len(label) > 0:
                plt.annotate(
                    label,
                    xy=(0.0, 1.025),
                    xycoords="axes fraction",
                    fontsize=24,
                    weight="bold"
                )
            if savepath is None:
                plt.show()
            else:
                plt.savefig(
                    savepath,
                    dpi=600,
                    transparent=True,
                    bbox_inches="tight"
                )
            plt.close()

        return delta_times

    def total_imaging_size(self) -> int:
        """
        Calculates the total amount of imaging in the PMBB in TB.
        Input:
            None.
        Returns:
            The total size of imaging studies in the PMBB in TB.
        """
        BYTES_TO_TERABYTES = 1_099_511_627_776
        return np.sum(self.data[self.size_key]) / BYTES_TO_TERABYTES

    def imaging_by_year(
        self,
        savepath: Optional[Union[Path, str]] = None,
        do_plot: bool = True,
        max_modalities: int = 10,
        min_year: int = 1995,
        max_year: int = 2015,
        all_modalities: bool = False,
        label: Optional[str] = None
    ) -> Dict[str, Dict[int, int]]:
        """
        Calculates and plots the number of imaging studies per year by
        imaging modality in the PMBB.
        Input:
            savepath: optional filepath to save the plot to. Default not saved.
            do_plot: whether or not to plot the data. Default True.
            max_modalities: maximum number of modalities to plot. Default 10.
                If -1, then all modalities are plotted.
            min_year: minimum year to plot on the x-axis. Default 1995.
                If -1, then all available years are plotted.
            max_year: maximum year to plot on the x-axis. Default 2015.
                If -1, then all available years are plotted.
            all_modalities: whether to plot all imaging studies by year,
                irregardless of modality. If True, the `max_modalities`
                parameter is ignored.
            label: optional label for the plot. Default None.
        Returns:
            A dictionary mapping imaging modality to another dictionary
            mapping year to total number of imaging studies of that modality
            in that year. Note that all modalities are returned irregardless
            if the `max_modalities` argument value.
        """
        if all_modalities:
            year_data = pd.DatetimeIndex(self.data[self.date_key]).year
            year_data = year_data.to_numpy()
            all_years = sorted(list(set(list(year_data))))
            year_to_volume = {}
            for y in all_years:
                year_to_volume[y] = np.count_nonzero(year_data == y)
            if do_plot:
                plot_years = [y for y in all_years if y >= min_year]
                if max_year >= min_year:
                    plot_years = [y for y in plot_years if y <= max_year]
                plot_volumes = [
                    year_to_volume[y] / len(self.patients) for y in plot_years
                ]
                plt.plot(plot_years, plot_volumes, color="black", linewidth=3)
                plt.xlabel("Year")
                plt.ylabel("Number of Imaging Studies per PMBB Capita")
                if savepath is None:
                    plt.show()
                else:
                    plt.savefig(
                        savepath,
                        dpi=600,
                        transparent=True,
                        bbox_inches="tight"
                    )
                plt.close()
            return year_to_volume

        modality_to_year_to_volume = {}
        used_cache = False
        current_func_name = inspect.stack()[0][3]
        if self.cache is not None:
            modality_to_year_to_volume = self.find_cache(
                current_func_name
            )
            used_cache = modality_to_year_to_volume is not None
            if not used_cache:
                modality_to_year_to_volume = {}
        if not used_cache:
            for mod in tqdm(self.modalities, desc="Imaging Volume by Year"):
                year_to_volume = defaultdict(int)
                mod_data = self.data[self.data[self.mod_key] == mod]
                year_data = pd.DatetimeIndex(mod_data[self.date_key]).year
                year_data = year_data.to_numpy()
                patient_ids = mod_data[self.pid_key].to_numpy()
                unique_pids = set(list(patient_ids))
                for _id in unique_pids:
                    year_to_volume[
                        np.min(year_data[np.where(patient_ids == _id)[0]])
                    ] += 1
                modality_to_year_to_volume[mod] = year_to_volume

        if do_plot:
            volume = self.volume_by_modality(
                do_plot=False, max_modalities=max_modalities
            )
            keys = list(volume.keys())
            if max_modalities > -1 and max_modalities < len(keys):
                keys = sorted(
                    keys, key=lambda k: volume[k], reverse=True
                )
                keys = keys[:max_modalities]
            modalities_to_plot = {
                k: modality_to_year_to_volume[k] for k in keys
            }

            plt.figure(figsize=(10, 8))
            for (mod, year_to_volume), co in zip(
                modalities_to_plot.items(), self.colors
            ):
                years = sorted(list(year_to_volume.keys()))
                years = [y for y in years if y >= min_year]
                if max_year >= min_year:
                    years = [y for y in years if y <= max_year]
                enrollment = [year_to_volume[y] / 1_000 for y in years]
                plt.plot(years, enrollment, label=mod, color=co, linewidth=3)
            plt.xlabel("Year")
            plt.xticks(np.arange(start=min_year, stop=(max_year + 1), step=5))
            plt.ylabel("Thousands of Imaging Studies")
            plt.legend(loc="upper left")
            if label is not None and len(label) > 0:
                plt.annotate(
                    label,
                    xy=(0.0, 1.025),
                    xycoords="axes fraction",
                    fontsize=24,
                    weight="bold"
                )
            if savepath is None:
                plt.show()
            else:
                plt.savefig(
                    savepath,
                    dpi=600,
                    transparent=True,
                    bbox_inches="tight"
                )
            plt.close()

        if not used_cache and self.cache is not None:
            self.save_cache(
                current_func_name, modality_to_year_to_volume
            )

        return modality_to_year_to_volume


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PMBB PACS Data Explorer")

    parser.add_argument(
        "--PACS_datapath",
        type=str,
        default=os.path.join("./data", "pacs_merge_encoded.csv"),
        help="Path to encoded PACS CSV data file on PMBB imaging data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./docs",
        help="Local directory to save generated plots to. Default ./docs"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache path to save and load processed data for faster usage."
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=18,
        help="Font size for plotting. Default 18."
    )
    parser.add_argument(
        "--use_sans_serif",
        action="store_true",
        help="Whether to use Sans Serif font for plotting. Default False."
    )

    return parser.parse_args()


def main():
    args = build_args()
    cache_dir = args.cache_dir
    if cache_dir.title() == "None" or len(cache_dir) == 0:
        cache_dir = None
    explorer = PMBBImagingExplorer(
        PACS_data=args.PACS_datapath,
        cache_dir=cache_dir,
        font_size=args.font_size,
        use_sans_serif=args.use_sans_serif
    )

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    explorer.volume_by_modality(
        savepath=os.path.join(args.output_dir, "volume_by_modality.png"),
        max_modalities=10,
        label="(a)"
    )
    explorer.imaging_by_year(
        savepath=os.path.join(args.output_dir, "imaging_by_year.png"),
        max_modalities=10,
        label="(b)"
    )
    explorer.repeat_studies_by_modality(
        savepath=os.path.join(
            args.output_dir, "repeat_studies_by_modality.png"
        ),
        max_modalities=10,
        label="(c)"
    )
    explorer.delta_time_dist_by_modality(
        savepath=os.path.join(
            args.output_dir, "delta_time_dist_by_modality.png"
        ),
        max_modalities=4,
        max_dt=1000,
        label="(d)"
    )


if __name__ == "__main__":
    main()
