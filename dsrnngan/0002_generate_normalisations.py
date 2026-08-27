from data import gen_fcst_norm
import os
import read_config

years = [2018, 2019, 2020, 2021]

def do_normalisations(years):
    data_paths = read_config.get_data_paths()
    norm_folder = data_paths["GENERAL"]["NORMALISATION_PATH"]

    os.makedirs(norm_folder, exist_ok=True)

    for year in years:
        print(f'doing year {year}')
        gen_fcst_norm(year)
        print(f'Finished year {year}')

    print('Done')

if __name__ == "__main__":
    do_normalisations(years)