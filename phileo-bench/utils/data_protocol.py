# Standard Library
import os
from glob import glob
import pandas as pd

# External Libraries
import buteo as beo
import numpy as np
import random
import json
from datetime import date
#

from utils.training_utils import MultiArray_1D
from pathlib import Path


random.seed(97)
np.random.seed(1234) # also affect pandas

REGIONS_DOWNSTREAM_DATA = ['denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1',
                           'israel-1', 'israel-2', 'japan', 'nigeria', 'north-america', 'senegal', 'south-america',
                           'tanzanipa-1', 'tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1']

REGIONS_BUCKETS = {'europe': ['europe','denmark-1','denmark-2'],
                   'east-africa':['east-africa','tanzania-1','tanzania-2','tanzania-3','tanzania-4','tanzania-5','uganda-1'],
                   'northwest-africa':['eq-guinea','ghana-1','egypt-1','israel-1','israel-2','nigeria','senegal'],
                   'north-america':['north-america'],
                   'south-america':['south-america'],
                   'japan':['japan']}
# REGIONS_BUCKETS = {'japan':['japan']}

REGIONS = REGIONS_DOWNSTREAM_DATA
LABELS = ['label_roads', 'label_kg', 'label_building', 'label_lc', 'label_coords']


def sanity_check_labels_exist(x_files, y_files):
    """
    checks that s2 and label numpy files are consistent

    :param x_files:
    :param y_files:
    :return:
    """
    existing_x = []
    existing_y = []
    counter_missing = 0

    assert len(x_files) == len(y_files)
    for x_path, y_path in zip(x_files, y_files):

        exists = os.path.exists(y_path)
        if exists:
            existing_x.append(x_path)
            existing_y.append(y_path)
        else:
            counter_missing += 1

    if counter_missing > 0:
        print(f'WARNING: {counter_missing} label(s) not found')
        missing = [y_f for y_f in y_files if y_f not in existing_y]
        print(f'Showing up to 5 missing files: {missing[:5]}')

    return existing_x, existing_y


def get_testset(folder: str,
                regions: list = None,
                y: str = 'building'):

    """
    Loads a pre-defined test set data from specified geographic regions.
    :param folder: dataset source folder
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :return: test MultiArrays
    """
    x_test_files = []

    if regions is None:
        regions = REGIONS
    else:
        for r in regions:
            assert r in REGIONS, f"region {r} not found"

    print("Testing regions:", regions)
    for region in regions:
        # get test samples of region
        x_test_files = x_test_files + sorted(glob(os.path.join(folder, f"{region}*test_s2.npy")))
    y_test_files = [f_name.replace('s2', f'label_{y}') for f_name in x_test_files]
    x_test_files, y_test_files = sanity_check_labels_exist(x_test_files, y_test_files)

    x_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_test_files])
    y_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_test_files])

    assert len(x_test) == len(y_test), "Lengths of x and y do not match."

    return x_test, y_test

def protocol_minifoundation(folder: str, y:str):
    """
    Loads all the data from the data folder.
    """

    x_train = sorted(glob(os.path.join(folder, f"*/*train_s2.npy")))
    y_train = [f_name.replace('s2', f'label_{y}') for f_name in x_train]

    x_val = []
    y_val = []
    for i in range(int(len(x_train)*0.05)):
        j = random.randint(0, len(x_train)-1)
        x_val.append(x_train[j])
        y_val.append(y_train[j])
        del x_train[j]; del y_train[j]

    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train], shuffle=True)
    y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train], shuffle=True)
    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val], shuffle=True)
    y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val], shuffle=True)

    return x_train, y_train, x_val, y_val

def protocol_split(folder: str,
                   split_percentage: float = 0.1,
                   regions: list = None,
                   y: str = 'building'):
    """
    Loads a percentage of the data from specified geographic regions.
    :param folder: dataset source folder
    :param split_percentage: percentage of data to sample from each region
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :return: train, val MultiArrays
    """

    if regions is None:
        regions = list(REGIONS_BUCKETS.keys())
    else:
        for r in regions:
            assert r in list(REGIONS_BUCKETS.keys()), f"region {r} not found. Possible regions are {list(REGIONS_BUCKETS.keys())}"

    assert 0 < split_percentage <= 1, "split percentage out of range (0 - 1)"

    df = pd.read_csv(glob(os.path.join(folder, f"*.csv"))[0])
    df = df.sort_values(by=['samples'])

    x_train_files = []
    shots_per_region = {'total':0}
    # egions =[subregion for r in regions for subregion in REGIONS_BUCKETS[r]]
    print("Train p-split regions:", regions)
    for region in regions:
        mask = [False]*len(df)
        for subregion in REGIONS_BUCKETS[region]:
            submask = [subregion in f for f in df.iloc[:, 0]]
            mask = [any(tuple) for tuple in zip(mask, submask)]
        mask = [region in f for f in df.iloc[:, 0]]
        df_temp = df[mask].sample(frac=1).copy().reset_index(drop=True)
        # skip iteration if Region does not belong to current dataset
        if df_temp.shape[0] == 0:
            continue

        df_temp['cumsum'] = df_temp['samples'].cumsum()

        # find row with closest value to the required number of samples
        idx_closest = df_temp.iloc[
            (df_temp['cumsum'] - int(df_temp['samples'].sum() * split_percentage)).abs().argsort()[:1]].index.values[0]
        x_train_files = x_train_files + list(df_temp.iloc[:idx_closest, 0])
        
        shots_per_region[region] = df_temp['cumsum'].values[idx_closest]

    shots_per_region['total'] = sum(shots_per_region.values())
    x_train_files = [os.path.join(folder, f_name) for f_name in x_train_files]
    y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]
    x_val_files = [f_name.replace('train', 'val') for f_name in x_train_files]
    y_val_files = [f_name.replace('train', 'val') for f_name in y_train_files]


    # checks that s2 and label numpy files are consistent
    x_train_files, y_train_files = sanity_check_labels_exist(x_train_files, y_train_files)
    x_val_files, y_val_files = sanity_check_labels_exist(x_val_files, y_val_files)


    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
    y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])

    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
    y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])

    assert len(x_train) == len(y_train)  and len(x_val) == len(
        y_val), "Lengths of x and y do not match."

    return x_train, y_train, x_val, y_val


def check_region_validity(folder, regions, y):
    # import pdb; pdb.set_trace()
    l = []
    for i, region in enumerate(regions):
        x_train_files = []
        for sub_regions in REGIONS_BUCKETS[region]: 
            x_train_files += sorted(glob(os.path.join(folder, f"{sub_regions}*train_s2.npy")))

        
        # generate multi array for region
        # x_train_files = sorted(glob(os.path.join(folder, f"{region}*train_s2.npy")))
        y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]

        # checks that s2 and label numpy files are consistent
        x_train_files, y_train_files = sanity_check_labels_exist(x_train_files, y_train_files)
        if x_train_files:
            l.append(region)

    # import pdb; pdb.set_trace()
    return l


def protocol_fewshot(folder: str,
                     dst: str,
                     n: int = 10,
                     val_ratio: float = 0.2,
                     regions: list = None,
                     y: str = 'building',
                     resample: bool = False,
                     ):

    """
    Loads n-samples data from specified geographic regions.
    :param folder: dataset source folder
    :param dst: save folder
    :param n: number of samples
    :param val_ratio: ratio of validation set
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :return: train, val MultiArrays
    """
    if os. path. exists(f'{dst}/{n}_shot_{y}/{n}shot_train_s2.npy'):
        train_X_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_train_s2.npy', mmap_mode='r')
        train_y_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_train_label_{y}.npy', mmap_mode='r')
        val_X_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_val_s2.npy', mmap_mode='r')
        val_y_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_val_label_{y}.npy', mmap_mode='r')
    else:

        if regions is None:
            regions = list(REGIONS_BUCKETS.keys())
        else:
            for r in regions:
                assert r in list(REGIONS_BUCKETS.keys()), f"region {r} not found. Possible regions are {list(REGIONS_BUCKETS.keys())}"
        regions = check_region_validity(folder, regions, y)

        f_x = glob(os.path.join(folder, f"{regions[0]}*test_s2.npy"))[0]
        ref_x = np.load(f_x, mmap_mode='r')
        f_y = glob(os.path.join(folder, f"{regions[0]}*test_label_{y}.npy"))[0]
        ref_y = np.load(f_y, mmap_mode='r')

        d_size = n*len(regions)
        d_size_val = int(np.ceil(n*val_ratio)*len(regions))

        train_X_temp = np.zeros_like(a=ref_x, shape=(d_size, ref_x.shape[1], ref_x.shape[2], ref_x.shape[3]))
        val_X_temp = np.zeros_like(a=ref_x, shape=(d_size_val, ref_x.shape[1], ref_x.shape[2], ref_x.shape[3]))
        train_y_temp = np.zeros_like(a=ref_y, shape=(d_size, ref_y.shape[1], ref_y.shape[2], ref_y.shape[3]))
        val_y_temp = np.zeros_like(a=ref_y, shape=(d_size_val, ref_y.shape[1], ref_y.shape[2], ref_y.shape[3]))
        del ref_x ; del ref_y

        print("Train n-shot regions:", regions)
        for i, region in enumerate(regions):
            # generate multi array for region
            x_train_files = []
            for sub_regions in REGIONS_BUCKETS[region]: 
                x_train_files += sorted(glob(os.path.join(folder, f"{sub_regions}*train_s2.npy")))
            y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]
            x_val_files = [f_name.replace('train', 'val') for f_name in x_train_files]
            y_val_files = [f_name.replace('train', 'val') for f_name in y_train_files]

            # checks that s2 and label numpy files are consistent
            x_train_files, y_train_files = sanity_check_labels_exist(x_train_files, y_train_files)
            x_val_files, y_val_files = sanity_check_labels_exist(x_val_files, y_val_files)

            x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
            y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])
            x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
            y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])

            if n < len(x_train):
                train_indexes = random.sample(range(0, len(x_train)), n)

                for j, idx in enumerate(train_indexes):
                    train_X_temp[(n*i)+j] = x_train[idx]
                    train_y_temp[(n * i) + j] = y_train[idx]

            else:
                # resample if n > than regions number of samples
                for j in range(0, len(x_train)):
                    train_X_temp[(n * i)+j] = x_train[j]
                    train_y_temp[(n * i)+j] = y_train[j]

                if resample:
                    train_indexes = random.choices(range(0, len(x_train)), k=(n - len(x_train)))
                    for j, idx in enumerate(train_indexes):
                        train_X_temp[(n * i)+len(x_train)+j] = x_train[idx]
                        train_y_temp[(n * i)+len(x_train) + j] = y_train[idx]

            if int(np.ceil(n * val_ratio)) < len(x_val):

                val_indexes = random.sample(range(0, len(x_val)), int(np.ceil(n * val_ratio)))

                for j, idx in enumerate(val_indexes):
                    val_X_temp[(int(np.ceil(n * val_ratio)) * i) + j] = x_val[idx]
                    val_y_temp[(int(np.ceil(n * val_ratio)) * i) + j] = y_val[idx]

            else:
                # resample if n > than regions number of samples
                for j in range(0, len(x_val)):
                    val_X_temp[(int(np.ceil(n * val_ratio)))+j] = x_val[j]
                    val_y_temp[(int(np.ceil(n * val_ratio)))+j] = y_val[j]
                if resample:
                    val_indexes = random.choices(range(0, len(x_val)), k=((int(np.ceil(n * val_ratio))) - len(x_val)))
                    for j, idx in enumerate(val_indexes):
                        val_X_temp[(int(np.ceil(n * val_ratio)))+len(x_val)+j] = x_val[idx]
                        val_y_temp[(int(np.ceil(n * val_ratio)))+len(x_val) + j] = y_val[idx]

            del x_train; del y_train; del x_val; del y_val

        os.makedirs(f'{dst}/{n}_shot_{y}', exist_ok=True)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_train_s2.npy', train_X_temp)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_train_label_{y}.npy', train_y_temp)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_val_s2.npy', val_X_temp)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_val_label_{y}.npy', val_y_temp)
    return train_X_temp, train_y_temp, val_X_temp, val_y_temp


def protocol_fewshot_memmapped(folder: str,
                     dst: str,
                     n: int = 10,
                     val_ratio: float = 0.2,
                     regions: list = None,
                     y: str = 'building',
                     data_selection: str = 'strict',
                     name: str = '128_10m'
                     ):

    """
    Loads n-samples data from specified geographic regions.
    :param folder: dataset source folder
    :param dst: save folder
    :param n: number of samples
    :param val_ratio: ratio of validation set
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :param data_selection: choose from 'strict' (take train/val selection from predefined selection), 'create' (use train/val selection if exists, else create it), 'random' (create train/val selection randomly)
    :return: train, val MultiArrays
    """

    
    if regions is None:
        regions = list(REGIONS_BUCKETS.keys())
        # import pdb ; pdb.set_trace()
    else:
        # import pdb ; pdb.set_trace()
        for r in regions:
            assert r in list(REGIONS_BUCKETS.keys()), f"region {r} not found. Possible regions are {list(REGIONS_BUCKETS.keys())}"
    regions = check_region_validity(folder, regions, y)

    assert data_selection in ['strict','create','random']

    samples_loaded = False
    if data_selection != 'random':
        indices_path = glob(f"indices/indices_*_{name}_{y}_{n}.json")
        
        if len(indices_path) == 0:
            if data_selection == 'create':
                samples_dict = {}
                print(f'creating train/val selection for task {y}, nshot={n}')
            else:
                raise ValueError('No file found for nshot sample selection while data_selection="strict". If you want to create fixed indices on the fly or use random train/val samples consider setting data_selction to "create" or "random"')
        
        elif len(indices_path) > 1:
            raise ValueError('Multiple files found for nshot sample selection')
        
        else:
            samples_loaded = True
            print('Loading predefined train/val selection')
            with open(indices_path[0], 'r') as f:
                samples_dict = json.load(f)
    
    x_train_samples = []
    y_train_samples = []
    x_val_samples = []
    y_val_samples = []
    # import pdb ; pdb.set_trace()
    for i, region in enumerate(regions):
        print(i,region)

        # generate multi array for region
        x_train_files = []
        for sub_regions in REGIONS_BUCKETS[region]: 
            x_train_files += sorted(glob(os.path.join(folder, f"{sub_regions}*train_s2.npy")))
        y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]
        x_val_files = [f_name.replace('train', 'val') for f_name in x_train_files]
        y_val_files = [f_name.replace('train', 'val') for f_name in y_train_files]

        # import pdb ; pdb.set_trace()

        # checks that s2 and label numpy files are consistent
        x_train_files, y_train_files = sanity_check_labels_exist(x_train_files, y_train_files)
        x_val_files, y_val_files = sanity_check_labels_exist(x_val_files, y_val_files)

        x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
        y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])
        x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
        y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])

        n_train_samples = min(n, len(x_train))
        n_val_samples = min(int(np.ceil(n * val_ratio)), len(x_val))

        if samples_loaded:
            assert len(x_train) == samples_dict[region]['length_multi_array_train']
            assert len(x_val) == samples_dict[region]['length_multi_array_val']

            train_indices = samples_dict[region]['train_indices']
            val_indices = samples_dict[region]['val_indices']

            # assert len(train_indices) == n_train_samples
            # import pdb; pdb.set_trace()
            # assert len(val_indices) == n_val_samples

        else:
            train_indices= random.Random(12345).sample(range(0, len(x_train)), n_train_samples)
            val_indices  = random.Random(12345).sample(range(0, len(y_val)), n_val_samples)

            random_sampler = random.Random(156)
            if y =='roads' or y=='building': # make sure training data is representative of the task
                random_train_indices= random.Random(12345).sample(range(0, len(x_train)), len(x_train))
                random_val_indices  = random.Random(12345).sample(range(0, len(y_val)), len(y_val))
                train_indices = []
                val_indices = []
                for i in random_train_indices:
                    label = y_train[i]
                    if np.mean(label)>0.005:

                        train_indices.append(i)
                    else:
                        if random_sampler.random()>0.75:
                            train_indices.append(i)

                    if len(train_indices)==n_train_samples:
                        break

                for i in random_val_indices:
                    label = y_val[i]
                    if np.mean(label)>0.01:
                        val_indices.append(i)
                    else:
                        if random_sampler.random()>0.75:
                            val_indices.append(i)
                    if len(val_indices)==n_val_samples:
                        break
        
            samples_dict[region] = {'train_indices':train_indices, 'val_indices':val_indices, 'length_multi_array_train':len(x_train), 'length_multi_array_val':len(x_val)}

        x_train_samples += [x_train[i] for i in train_indices]
        y_train_samples += [y_train[i] for i in train_indices]

        x_val_samples += [x_val[i] for i in val_indices]
        y_val_samples += [y_val[i] for i in val_indices]

    if not samples_loaded and data_selection=='create':
        out_path = Path(__file__).resolve().parent.parent
        out_path = out_path / f'indices/indices_{date.today().strftime("%d%m%Y")}_{name}_{y}_{n}.json'
        print(f'No predefined train/val sampling was used. Saving current sampling schema in {out_path}')
        with open(out_path, 'w') as f:
            json.dump(samples_dict, f)

    return MultiArray_1D(x_train_samples), MultiArray_1D(y_train_samples), MultiArray_1D(x_val_samples), MultiArray_1D(y_val_samples)


if __name__ == '__main__':
    label =['roads', 'building', 'lc']
    n_shots = [1, 2, 5, 10, 50, 100, 150, 200, 500, 750, 1000]
    for l in label:
        for n in n_shots:
            x_train, y_train, x_val, y_val = protocol_fewshot('/phileo_data/downstream/downstream_dataset_patches_np/',
                                                              dst='/phileo_data/downstream/downstream_datasets_nshot/',
                                                              n=n,
                                                              y=l)