import pandas as pd
from pathlib import Path

#
# convert photometry data per night -> data per cepheid
#
# can leave as apparent magnitudes, P-L functions convert these to absolute.
#



def prepare_period_fit_data(photometry_dir, cepheid_id):
    """
    Finds the photometry data for each individual cepheid.
    """
    all_data = []
    for csv_file in sorted(Path(photometry_dir).glob("photometry_*.csv")):
        df = pd.read_csv(csv_file)
        df["ID"] = df["ID"].apply(lambda x: f"{int(x):02d}")
        cep_rows = df[df["ID"] == cepheid_id]
        all_data.append(cep_rows)

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values("ISOT")
    return combined

def export_all_cepheid_csvs(photometry_dir, output_dir, cepheid_ids):
    """
    Exports each individual cepheid as its own .csv
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for cep_id in cepheid_ids:
        try:
            combined = prepare_period_fit_data(photometry_dir, cep_id)

            if combined.empty:
                print(f"No data found for Cepheid {cep_id}, skipping")
                continue

            name = combined["Name"].iloc[0].replace(" ", "_")
            filename = f"{output_dir}/cepheid_{cep_id}_{name}.csv"
            combined.to_csv(filename, index=False)
            print(f"Saved {filename} ({len(combined)} observations)")

        except Exception as e:
            print(f"Cepheid {cep_id} failed: {e}")
            continue

if __name__ == "__main__":
    photometry_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Photometry"
    output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Analysis/RawData"
    df = pd.read_csv(Path(photometry_dir) / "photometry_2025-10-06.csv")
    print(df["ID"].dtype)
    print(df["ID"].unique())
    cepheid_ids = ["01", "02", "03", "04", "05", "06",
                   "07", "08", "09", "10", "11"]

    export_all_cepheid_csvs(photometry_dir, output_dir, cepheid_ids)