"""
IDP analysis using PMBB data and PCA/t-SNE.

Author(s):
    Allison Chae
    Michael S Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
from typing import Optional, Sequence, Tuple, Union


class IDPExplorer:
    def __init__(
        self,
        idp_datapaths: Sequence[Union[Path, str]],
        icd_datapath: Union[Path, str],
        seed: Optional[int] = 42
    ):
        """
        Args:
            idp_datapaths: filepaths to CSV files with PMBB IDP data.
            icd_datapath: filepath to file with PMBB ICD-9 code data.
            seed: optional random seed. Default 42.
        """
        self.idp_datapaths = idp_datapaths
        self.icd_datapath = icd_datapath
        self._id = "PMBB_ID"
        self.diagnoses_key = "Diagnoses"
        self.valid_idp_columns = [
            self._id,
            "LIVER_METRIC_VOLUME",
            "LIVER_MEAN_HU",
            "SPLEEN_METRIC_VOLUME",
            "SPLEEN_MEAN_HU",
            "SUBQ_METRIC_VOLUME",
            "VISCERAL_METRIC_VOLUME"
        ]
        self.icd_categories = {
            "INFECTIOUS": [1, 140],
            "NEOPLASMS": [140, 240],
            "METABOLIC": [240, 280],
            "VASCULAR": [280, 290],
            "PSYCHIATRIC": [290, 320],
            "NERVOUS": [320, 390],
            "CIRCULATORY": [390, 460],
            "RESPIRATORY": [460, 520],
            "DIGESTIVE": [520, 580],
            "GENITOURINARY": [580, 680],
            "SKIN": [680, 710],
            "MSK": [710, 740],
            "CONGENITAL": [740, 760],
            "OTHER": [760, 1000]
        }
        self.icds_references = {
            "Obesity": [278, 279],
            "Obstructive Sleep Apnea": [327.23, 328.24],
            "Hypertension": [401, 402],
            "NAFLD": [571, 572],
            "Diabetes": [250, 251],
            "Genitourinary Diseases": [580, 680],
            "Chronic Kidney Disease": [585, 586]
        }
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.plot_config()
        self.data, self.labels = self.fimport()

    def fimport(self) -> Tuple[pd.DataFrame]:
        """
        Imports data from `self.datapaths` as `pandas` dataframe objects.
        Input:
            None.
        Returns:
            data: imported IDP data.
            diagnoses: imported ICD-9 code data.
        """
        data = None

        # Import IDP data.
        for fn in self.idp_datapaths:
            df = pd.read_csv(fn)
            # Only keep the valid IDP columns.
            df = df[list(set(df.columns) & set(self.valid_idp_columns))]
            # Remove any rows with negative IDP values.
            for col, dtype in zip(df.columns, df.dtypes):
                if not pd.api.types.is_numeric_dtype(dtype):
                    continue
                df = df[df[col] > 0]
            # If there are duplicate IDPs for a patient, just keep the mean
            # of that patient's IDP values.
            df = df.groupby(self._id).mean().reset_index()

            if data is None:
                data = df
                continue
            data = pd.merge(data, df, on=self._id)

        # Import ICD-9 data.
        df = pd.read_csv(self.icd_datapath, sep="\t")
        cols = df.columns.drop(self._id)
        def nonzero(row, cols): return [float(y) for y in cols[~(row == 0)]]
        diagnoses = df.drop([self._id], axis=1).apply(
            lambda x: nonzero(x, cols), axis=1
        )
        diagnoses = diagnoses.to_frame(name=self.diagnoses_key)
        diagnoses[self._id] = df[self._id]

        return data, diagnoses

    def visualize(
        self,
        savedir: Optional[Union[Path, str]] = None,
        n_components: int = 1,
        dim_reduction_method: str = "PCA",
        use_labels: bool = True,
        verbose: bool = True
    ) -> None:
        """
        Plots IDP principal component(s) as a function of different diagnoses.
        Input:
            savedir: directory to save plots to. Default no plots saved.
            n_components: number of principal components to plot. One of
                [1, 2]. Default 1.
            dim_reduction_method: dimensionality reduction method. One of
                [`PCA`, `TSNE`]. Default `PCA`.
            use_labels: whether to plot labels using (a), (b), (c), etc.
            verbose: flag for verbose outputs.
        Returns:
            None.
        """
        idp_X = self.data.drop([self._id], axis=1).to_numpy()
        idp_Z = StandardScaler().fit_transform(idp_X)
        n_components = min(max(n_components, 1), 2)
        if dim_reduction_method.upper() == "TSNE":
            idp_embedding = TSNE(
               n_components=n_components,
               learning_rate="auto",
               init="pca",
               perplexity=50
            ).fit_transform(idp_Z)
        elif dim_reduction_method.upper() == "PCA":
            pca = PCA(n_components=n_components)
            idp_embedding = pca.fit_transform(idp_Z)
            print("Explained Variance (%):", pca.explained_variance_ratio_)
        else:
            raise ValueError(
                f"Unrecognized dim reduction method {dim_reduction_method}"
            )

        label = 97  # Unicode for `a`.
        for (diagnosis, (icd_low, icd_high)), co in zip(
            self.icds_references.items(), self.colors
        ):
            positive = set([])
            negative = set([])
            for labels, _id in zip(
                self.labels[self.diagnoses_key], self.labels[self._id]
            ):
                labels = np.array(labels)
                if np.any((icd_low <= labels) & (labels < icd_high)):
                    if _id in negative:
                        negative.remove(_id)
                    positive.add(_id)
                else:
                    negative.add(_id)
            positive = np.array(list(positive))
            negative = np.array(list(negative))
            positive = np.where(
                np.in1d(self.data[self._id].to_numpy(), positive)
            )
            negative = np.where(
                np.in1d(self.data[self._id].to_numpy(), negative)
            )

            plt.figure(figsize=(10, 5 * n_components))
            if n_components == 2:
                plt.scatter(
                    idp_embedding[negative, 0],
                    idp_embedding[negative, -1],
                    alpha=0.5,
                    color="#8f8f8f",
                    label="_nolegend_"
                )
                plt.scatter(
                    idp_embedding[positive, 0],
                    idp_embedding[positive, -1],
                    alpha=0.5,
                    color=co,
                    label=diagnosis
                )
                if dim_reduction_method.upper() == "TSNE":
                    plt.xlabel("IDP t-SNE Dimension 1")
                    plt.ylabel("IDP t-SNE Dimension 2")
                else:
                    plt.xlabel("IDP PCA Dimension 1")
                    plt.ylabel("IDP PCA Dimension 2")
                plt.legend(loc="lower right")
            elif n_components == 1:
                idp_embedding = np.squeeze(idp_embedding)
                num_pos, num_neg = len(positive[0]), len(negative[0])
                if verbose:
                    print(
                        diagnosis + ":",
                        num_pos,
                        "[positive] /",
                        num_neg,
                        "[negative]"
                    )
                plt.hist(
                    idp_embedding[negative],
                    color="#8f8f8f",
                    edgecolor="black",
                    label="_nolegend",
                    alpha=0.6,
                    bins=50,
                    weights=np.ones_like(idp_embedding[negative]) / num_neg
                )
                plt.hist(
                    idp_embedding[positive],
                    color=co,
                    edgecolor="black",
                    label=diagnosis,
                    alpha=0.7,
                    bins=50,
                    weights=np.ones_like(idp_embedding[positive]) / num_pos
                )
                p_value = ks_2samp(
                    idp_embedding[positive], idp_embedding[negative]
                ).pvalue
                plt.annotate(
                    self.p_value_str(p_value),
                    xy=(0.025, 0.95),
                    xycoords="axes fraction",
                    color="black",
                    horizontalalignment="left",
                    verticalalignment="top"
                )
                if dim_reduction_method.upper() == "TSNE":
                    plt.xlabel("IDP t-SNE Dimension 1")
                else:
                    plt.xlabel("IDP PCA Dimension 1")
                plt.ylabel("Frequency")
                plt.legend(loc="upper right")

            if use_labels:
                plt.annotate(
                    "(" + chr(label) + ")",
                    xy=(0.0, 1.05),
                    xycoords="axes fraction",
                    fontsize=24,
                    weight="bold"
                )
                label += 1
            if savedir is None or len(savedir) == 0:
                plt.show()
            else:
                plt.savefig(
                    os.path.join(
                        savedir, diagnosis.lower().replace(" ", "_") + ".png"
                    ),
                    transparent=True,
                    dpi=600,
                    bbox_inches="tight"
                )
            plt.close()

    def plot_config(self) -> None:
        """
        Plot configuration variables.
        Input:
            None.
        Returns:
            None.
        """
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams['font.family'] = "Arial"
        matplotlib.rcParams.update({"font.size": 20})
        self.colors = [
            "#55752F",
            "#E37E00",
            "#6E8E84",
            "#90353B",
            "#1A476F",
            "#938DD2",
            "#C10534",
            "#CAC27E",
            "#A0522D",
            "#7B92A8",
            "#2D6D66",
            "#9C8847",
            "#BFA19C",
            "#FFD200",
            "#89969B"
        ]

    def p_value_str(self, p: float) -> str:
        """
        Converts a p value into an string annotation for plots.
        Input:
            p: input p value (assumed to be less than 1).
        Returns:
            Formatted p value string.
        """
        if p >= 0.01:
            return fr"$p = {p:.3f}$"
        if p > 1 or p <= 0:
            raise ValueError("p values should be between 0 and 1, got {p}.")
        decimal_power = 0
        while p < 1:
            p = p * 10
            decimal_power -= 1
        return fr"$p = {p:.3f} \times 10^{{{decimal_power}}}$"


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PMBB IDP Explorer")

    datadir = "./data"
    parser.add_argument(
        "--idp_datapaths",
        type=str,
        nargs="+",
        default=[
            os.path.join(datadir, "steatosis_run_2_merge.csv"),
            os.path.join(datadir, "visceral_merge_log_run_11_1_20.csv")
        ],
        help="Filepaths to CSV files with PMBB IDP data."
    )
    parser.add_argument(
        "--icd_datapath",
        type=str,
        default=os.path.join(
            datadir,
            "PMBB-Release-2020-2.2_phenotype_PheCode-matrix-ct-studies.txt"
        ),
        help="Filepath to file with PMBB ICD-9 diagnosis code data."
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default=None,
        help="Directory to save plots to. Default plots are not saved."
    )
    parser.add_argument(
        "--dim_reduction_method",
        type=str,
        default="PCA",
        choices=["PCA", "TSNE"],
        help="Dimensionality reduction method. One of [`PCA`, `TSNE`]."
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of principal components to plot. One of [`1`, `2`]."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed. Default 42."
    )

    return parser.parse_args()


def main():
    args = build_args()
    explorer = IDPExplorer(
        idp_datapaths=args.idp_datapaths,
        icd_datapath=args.icd_datapath,
        seed=args.seed
    )
    if args.savedir is not None and len(args.savedir) > 0:
        if not os.path.isdir(args.savedir):
            os.mkdir(args.savedir)
    explorer.visualize(
        savedir=args.savedir,
        n_components=args.n_components,
        dim_reduction_method=args.dim_reduction_method,
        use_labels=True,
        verbose=True
    )


if __name__ == "__main__":
    main()
